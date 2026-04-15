"""
retrieval/query_engine.py
─────────────────────────
Assembles the four-layer retrieval pipeline with per-request RBAC filtering.

  Layer 0 — RBAC pre-filter (Qdrant payload filter)   ← NEW
    Before the ANN search even starts, Qdrant is told: "only scan chunks whose
    `allowed_roles` array contains at least one of the user's expanded roles."
    This is enforced inside the vector database — no post-retrieval filtering,
    no risk of a forbidden chunk leaking into the context window.

  Layer 1 — Hybrid search (Qdrant)
    BM25 + dense vector fused with RRF, scoped to the RBAC-filtered subset.

  Layer 2 — AutoMerging Retriever (LlamaIndex)
    Swaps matched leaf chunks for their parent chunks to give the LLM
    richer context without widening the initial search.

  Layer 3 — Cohere Reranker
    Cross-encoder re-scores candidates. Top-N survive.

  Layer 4 — Response synthesizer
    Builds the final prompt and calls the LLM with streaming.

Why per-request engine instead of a singleton?
───────────────────────────────────────────────
The Qdrant pre-filter depends on the current user's role, which varies per
request. We solve this cleanly: the *index* is a singleton (expensive to
build), but we construct a lightweight retriever + engine wrapper per request
using build_query_engine_for_user(). The per-request cost is negligible —
it's just Python object instantiation, no I/O.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from loguru import logger

from auth.rbac import expand_roles
from config import settings


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert assistant for internal company documentation.

Your job:
- Answer questions based ONLY on the provided context documents.
- Always cite your sources using the format [Source: <filename>].
- If the answer is not in the context, say "I couldn't find that in the available
  documents" — do NOT make up an answer.
- Be concise and direct. Use bullet points for lists.
- If a question is ambiguous, clarify the most likely intent and answer that.

Tone: professional but approachable."""


# ── RBAC filter builder ───────────────────────────────────────────────────────
def _build_rbac_filter(user_role: str) -> MetadataFilters:
    """
    Build a LlamaIndex MetadataFilters object that Qdrant translates into a
    payload pre-filter.

    Logic:
        expand_roles("management") -> ["employee", "hr", "finance", "management"]

        Qdrant filter (pseudo-SQL):
            WHERE allowed_roles CONTAINS ANY OF
                  ["employee", "hr", "finance", "management"]

    This uses FilterOperator.IN which maps to Qdrant's MatchAny condition.
    The filter runs *before* the ANN search, meaning Qdrant never even
    scores chunks that the user is not allowed to see.

    Interview note: this is a pre-filter (not post-filter). Post-filtering
    would retrieve the chunks first and then discard them — wasteful and
    insecure. Pre-filtering is enforced at the vector database level.
    """
    accessible_roles = expand_roles(user_role)
    logger.debug(f"RBAC pre-filter: role={user_role!r} -> accessible={accessible_roles}")

    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="allowed_roles",
                value=accessible_roles,
                operator=FilterOperator.IN,   # "allowed_roles contains any of [list]"
            )
        ],
        condition=FilterCondition.AND,
    )


# ── Per-request engine builder ────────────────────────────────────────────────
def build_query_engine_for_user(
    index: VectorStoreIndex,
    user_role: str,
) -> RetrieverQueryEngine:
    """
    Build a query engine scoped to a specific user's role.

    This is called once per API request. It reuses the shared index
    (no re-embedding) but injects a fresh RBAC filter into the retriever.

    Args:
        index:     The VectorStoreIndex singleton loaded at startup.
        user_role: The role string extracted from the user's JWT
                   (e.g. "hr", "engineering", "management").

    Returns:
        A RetrieverQueryEngine that will only surface chunks the user
        is authorised to see.
    """
    storage_context = index.storage_context
    rbac_filter = _build_rbac_filter(user_role)

    # ── Layer 1: Hybrid retriever with RBAC pre-filter ────────────────────────
    base_retriever = index.as_retriever(
        similarity_top_k=settings.TOP_K_RETRIEVAL,
        vector_store_query_mode="hybrid",
        alpha=settings.HYBRID_ALPHA,
        filters=rbac_filter,            # RBAC pre-filter injected here
    )

    # ── Layer 2: AutoMerging Retriever ────────────────────────────────────────
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context,
        simple_ratio_thresh=0.5,
        verbose=False,
    )

    # ── Layer 3: Post-processors ──────────────────────────────────────────────
    postprocessors = []
    postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.35))

    if settings.COHERE_API_KEY:
        postprocessors.append(
            CohereRerank(
                api_key=settings.COHERE_API_KEY,
                top_n=settings.TOP_N_RERANK,
                model="rerank-english-v3.0",
            )
        )

    # ── Layer 4: Response synthesizer ────────────────────────────────────────
    synthesizer = get_response_synthesizer(
        response_mode="compact",
        streaming=True,
        text_qa_template=_build_qa_prompt(),
        verbose=False,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=postprocessors,
        response_synthesizer=synthesizer,
    )


def _build_qa_prompt():
    from llama_index.core import PromptTemplate

    qa_prompt_tmpl = (
        f"{SYSTEM_PROMPT}\n\n"
        "---------------------\n"
        "CONTEXT DOCUMENTS:\n"
        "{context_str}\n"
        "---------------------\n"
        "USER QUESTION: {query_str}\n\n"
        "ANSWER (cite sources using [Source: filename]):"
    )
    return PromptTemplate(qa_prompt_tmpl)


# ── Index singleton ────────────────────────────────────────────────────────────
_index_cache: VectorStoreIndex | None = None


def get_index() -> VectorStoreIndex:
    """Return the cached index. Raises if not yet initialised."""
    if _index_cache is None:
        raise RuntimeError(
            "Index not initialised. Call set_index() during application startup."
        )
    return _index_cache


def set_index(index: VectorStoreIndex) -> None:
    """Store the index singleton. Called once during FastAPI lifespan startup."""
    global _index_cache
    _index_cache = index
    logger.info("Index singleton set in query_engine module")


# ── Backwards-compat shim ─────────────────────────────────────────────────────
def get_query_engine(index: VectorStoreIndex | None = None) -> RetrieverQueryEngine:
    """
    Legacy helper used by eval scripts and tests.
    Defaults to admin role (sees everything).
    New code should call build_query_engine_for_user(index, role) directly.
    """
    idx = index or get_index()
    logger.warning(
        "get_query_engine() called without a user role — defaulting to 'admin'. "
        "Use build_query_engine_for_user(index, role) in production paths."
    )
    return build_query_engine_for_user(idx, "admin")
