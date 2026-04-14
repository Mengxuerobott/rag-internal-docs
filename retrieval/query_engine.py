"""
retrieval/query_engine.py
─────────────────────────
Assembles the three-layer retrieval pipeline:

  Layer 1 — Hybrid search (Qdrant)
    Combines dense vector similarity (semantic) with BM25 sparse vectors
    (keyword). Fused with Reciprocal Rank Fusion (RRF). This catches both
    "what is our vacation policy" (semantic) and "FMLA leave" (keyword/acronym).

  Layer 2 — AutoMerging Retriever (LlamaIndex)
    If enough sibling leaf chunks from the same parent are retrieved,
    swaps them out for the full parent chunk. Gives the LLM richer context
    without widening the initial search.

  Layer 3 — Cohere Reranker
    Cross-encoder reranker takes the top-K candidates and re-scores them
    with a more expensive model that reads (query, chunk) together.
    Dramatically reduces irrelevant chunks before they reach the LLM.

  Layer 4 — Response synthesizer
    Builds the final prompt (system + retrieved context + user question)
    and calls GPT-4o / Claude with streaming.

The pipeline is built once at startup and reused across all requests.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from loguru import logger

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


def build_query_engine(index: VectorStoreIndex) -> RetrieverQueryEngine:
    """
    Assemble the full retrieval pipeline from a pre-built VectorStoreIndex.

    Args:
        index: A VectorStoreIndex backed by Qdrant (built by ingestion/embedder.py).

    Returns:
        A RetrieverQueryEngine ready to accept .query() calls.
    """
    storage_context = index.storage_context

    # ── Layer 1: Hybrid retriever ─────────────────────────────────────────────
    # similarity_top_k: how many candidates to fetch from Qdrant before reranking.
    # More candidates → higher recall, but more tokens to the reranker.
    # 10 is a good default; raise to 20 for larger corpora.
    #
    # vector_store_query_mode="hybrid": tells Qdrant to run both a dense ANN
    # search and a BM25 sparse search, then fuse results with RRF.
    #
    # alpha: weighting between sparse (0.0) and dense (1.0). 0.5 is balanced.
    # Increase toward 1.0 for pure-prose docs; decrease toward 0.0 if your
    # docs are heavy with acronyms / product codes.
    base_retriever = index.as_retriever(
        similarity_top_k=settings.TOP_K_RETRIEVAL,
        vector_store_query_mode="hybrid",
        alpha=settings.HYBRID_ALPHA,
    )
    logger.debug(
        f"Hybrid retriever: top_k={settings.TOP_K_RETRIEVAL}, alpha={settings.HYBRID_ALPHA}"
    )

    # ── Layer 2: AutoMerging Retriever ────────────────────────────────────────
    # Wraps the base retriever. After fetching leaf nodes, checks whether a
    # majority of siblings from the same parent were also retrieved.
    # If yes, replaces the leaf cluster with the parent node (broader context).
    #
    # simple_ratio_thresh (default 0.5): if ≥50% of a parent's children are in
    # the retrieved set, merge up to the parent.
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context,
        simple_ratio_thresh=0.5,
        verbose=False,
    )

    # ── Layer 3: Post-processors ──────────────────────────────────────────────
    postprocessors = []

    # Similarity threshold filter: drop any node below this score before reranking.
    # Prevents low-quality matches from polluting the context window.
    postprocessors.append(
        SimilarityPostprocessor(similarity_cutoff=0.35)
    )

    # Cohere Reranker (most impactful quality improvement for ~zero latency cost).
    if settings.COHERE_API_KEY:
        reranker = CohereRerank(
            api_key=settings.COHERE_API_KEY,
            top_n=settings.TOP_N_RERANK,
            model="rerank-english-v3.0",
        )
        postprocessors.append(reranker)
        logger.info(f"Cohere reranker enabled — keeping top {settings.TOP_N_RERANK} nodes")
    else:
        logger.warning(
            "COHERE_API_KEY not set — skipping reranker. "
            "Retrieval quality will be lower. Get a free key at dashboard.cohere.com"
        )

    # ── Layer 4: Response synthesizer ────────────────────────────────────────
    # "compact" mode: packs retrieved chunks into the fewest LLM calls possible.
    # Alternative modes: "tree_summarize" (better for long docs), "refine"
    # (iterative, more expensive but higher quality for complex questions).
    synthesizer = get_response_synthesizer(
        response_mode="compact",
        streaming=True,
        text_qa_template=_build_qa_prompt(),
        verbose=False,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=postprocessors,
        response_synthesizer=synthesizer,
    )

    logger.info("Query engine built successfully")
    return query_engine


def _build_qa_prompt():
    """
    Build the LlamaIndex PromptTemplate used for the final QA call.
    Injects SYSTEM_PROMPT + retrieved context + user question.
    """
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


# ── Singleton cache ───────────────────────────────────────────────────────────
# The query engine is expensive to build (connects to Qdrant, validates models).
# Cache it as a module-level singleton so the FastAPI app only builds it once.
_engine_cache: RetrieverQueryEngine | None = None


def get_query_engine(index: VectorStoreIndex | None = None) -> RetrieverQueryEngine:
    """
    Return the cached query engine, building it on first call.
    Pass `index` on the first call; subsequent calls ignore it.
    """
    global _engine_cache
    if _engine_cache is None:
        if index is None:
            raise ValueError("Query engine not yet initialised — pass index on first call.")
        _engine_cache = build_query_engine(index)
    return _engine_cache
