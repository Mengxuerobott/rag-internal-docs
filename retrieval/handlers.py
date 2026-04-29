"""
retrieval/handlers.py
─────────────────────
Three concrete route handlers — each takes a RouteContext and returns a
RouteResult.  The router (retrieval/router.py) calls the correct one after
classifying the user's intent.

Handler 1 — SmallTalkHandler
  Input : free-form conversational message + conversation history
  Output: direct LLM reply, zero retrieval
  Cost  : one gpt-4o-mini call (~50ms, ~$0.00003)
  When  : greetings, capability questions, follow-up clarifications,
          anything that does not require document knowledge

Handler 2 — SummarizationHandler
  Input : "summarize <document name>" or "give me an overview of <doc>"
  Output: full-document summary, no vector search
  Cost  : one Qdrant scroll + one gpt-4o call (~500ms)
  When  : user asks for a holistic summary rather than a specific fact.
          The handler fetches ALL chunks for the target document directly
          (by doc_id or source filename) and summarises them in one LLM call.

Handler 3 — DeepRagHandler
  Input : substantive knowledge question requiring precise retrieval
  Output: answer + source citations from the full 4-layer pipeline
  Cost  : embedding lookup + Cohere rerank + gpt-4o (~1.2-1.8s)
  When  : everything that isn't small-talk or a summarisation request.

Each handler's RouteResult carries a `route_type` string so the API response
(and the Streamlit UI) can show the user which path was taken, which makes
the agentic routing behaviour observable and debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from auth.jwt_handler import CurrentUser
from auth.rbac import expand_roles
from config import settings


# ── Shared types ──────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    role: str          # "user" | "assistant"
    content: str


@dataclass
class RouteContext:
    """Everything a handler needs to produce an answer."""
    question:    str
    user:        CurrentUser
    history:     list[ConversationTurn] = field(default_factory=list)
    # For DeepRAG: the pre-built query engine (built per-request in api/main.py)
    query_engine: Optional[object] = None
    # For Summarization: target document name or doc_id extracted by the router
    target_doc:  Optional[str] = None


@dataclass
class RouteResult:
    """Unified output structure returned to the API layer."""
    answer:     str
    route_type: str                   # "small_talk" | "summarization" | "deep_rag"
    sources:    list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


# ── Shared helpers ────────────────────────────────────────────────────────────

def _openai() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _history_to_messages(history: list[ConversationTurn]) -> list[dict]:
    """Convert ConversationTurn list → OpenAI messages list."""
    return [{"role": t.role, "content": t.content} for t in history]


# ── Handler 1: SmallTalk ──────────────────────────────────────────────────────

_SMALL_TALK_SYSTEM = """\
You are a helpful assistant for an internal company document system.
You have access to company policies, HR documents, engineering guides,
finance policies, and legal documents.

For conversational messages, capability questions, or follow-ups that don't
require looking up a specific document, respond directly and helpfully.

If the user seems to be asking about a real document topic (leave policy, expense
reports, onboarding, security, etc.), let them know you can look that up for them
and suggest they ask a specific question.

Keep responses concise (2-4 sentences for greetings, up to a short paragraph
for capability questions). Be warm and professional."""


class SmallTalkHandler:
    """
    Handles conversational messages, capability questions, and chitchat.
    Zero retrieval — one direct LLM call with conversation history.

    This is the cheapest path in the router. A greeting that previously
    triggered a full vector search + reranker now costs one gpt-4o-mini call.
    """

    def handle(self, ctx: RouteContext) -> RouteResult:
        import time
        start = time.perf_counter()

        messages = [{"role": "system", "content": _SMALL_TALK_SYSTEM}]
        messages += _history_to_messages(ctx.history)
        messages.append({"role": "user", "content": ctx.question})

        client = _openai()
        response = client.chat.completions.create(
            model=settings.ROUTER_MODEL,     # gpt-4o-mini — cheap and fast
            messages=messages,
            max_tokens=300,
            temperature=0.7,                 # slightly higher for conversational warmth
        )

        answer = response.choices[0].message.content.strip()
        latency = (time.perf_counter() - start) * 1000

        logger.info(
            f"SmallTalk route: user={ctx.user.username!r} "
            f"latency={latency:.0f}ms"
        )

        return RouteResult(
            answer=answer,
            route_type="small_talk",
            sources=[],
            latency_ms=round(latency, 1),
        )


# ── Handler 2: Summarization ──────────────────────────────────────────────────

_SUMMARIZATION_SYSTEM = """\
You are a precise document summariser for an internal company knowledge system.
Produce a structured, comprehensive summary that a busy executive could read
in 2-3 minutes. Include:
  - Document purpose and audience
  - Key policies, rules, or procedures (bullet points)
  - Important dates, thresholds, or numbers
  - Any exceptions or edge cases
  - Who to contact for questions (if mentioned)

Write in clear professional prose with bullet points for lists.
Cite section headings when relevant."""

_SUMMARIZATION_USER_TMPL = """\
Please provide a comprehensive summary of the following document content.

Document: {doc_name}

Content:
{content}

Summary:"""


class SummarizationHandler:
    """
    Handles "summarize X" requests by fetching all chunks for a target
    document directly from Qdrant (no vector search) and summarising in
    a single LLM call.

    Why skip vector search for summarisation?
    ──────────────────────────────────────────
    Vector search is designed to find the *most relevant* passage for a query.
    For "summarise the expense policy", every passage IS relevant — we need the
    whole document. Fetching all chunks by doc_id/source with a Qdrant scroll
    is O(N) but N is small (typically 10-50 chunks per document) and avoids
    the reranker entirely.
    """

    def handle(self, ctx: RouteContext) -> RouteResult:
        import time
        start = time.perf_counter()

        # Fetch all chunks for this document from Qdrant
        chunks, doc_name = self._fetch_document_chunks(ctx)

        if not chunks:
            logger.warning(
                f"Summarization: no chunks found for target_doc={ctx.target_doc!r} "
                f"user={ctx.user.username!r}"
            )
            return RouteResult(
                answer=(
                    f"I couldn't find a document matching '{ctx.target_doc}' "
                    "in the knowledge base. Try asking about a specific topic instead, "
                    "or check the document list with /docs-list."
                ),
                route_type="summarization",
                sources=[],
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )

        # Concatenate chunks in order (approximate — Qdrant scroll doesn't
        # guarantee page order, but chunk text is self-contained enough for summary)
        combined_text = "\n\n---\n\n".join(c["text"] for c in chunks)

        # Truncate if the combined content would blow the context window
        # gpt-4o supports 128K tokens; 200K chars is a safe text limit
        if len(combined_text) > 200_000:
            combined_text = combined_text[:200_000] + "\n\n[Content truncated…]"

        client = _openai()
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,         # use the full model for summaries
            messages=[
                {"role": "system", "content": _SUMMARIZATION_SYSTEM},
                {
                    "role": "user",
                    "content": _SUMMARIZATION_USER_TMPL.format(
                        doc_name=doc_name,
                        content=combined_text,
                    ),
                },
            ],
            max_tokens=1500,
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()
        latency = (time.perf_counter() - start) * 1000

        logger.info(
            f"Summarization route: doc={doc_name!r} chunks={len(chunks)} "
            f"user={ctx.user.username!r} latency={latency:.0f}ms"
        )

        # Build source list (just the document itself — no scores for summarisation)
        sources = [{
            "source": doc_name,
            "department": chunks[0].get("department", "general") if chunks else "general",
            "allowed_roles": chunks[0].get("allowed_roles", []) if chunks else [],
            "text_snippet": f"Full document summary ({len(chunks)} chunks)",
            "score": 1.0,
        }]

        return RouteResult(
            answer=answer,
            route_type="summarization",
            sources=sources,
            latency_ms=round(latency, 1),
        )

    def _fetch_document_chunks(
        self,
        ctx: RouteContext,
    ) -> tuple[list[dict], str]:
        """
        Scroll Qdrant for all chunks matching target_doc, filtered by RBAC.

        Returns (chunks, resolved_doc_name) where chunks is a list of
        {text, department, allowed_roles, page_num} dicts.

        The RBAC filter is applied here too — a summarisation request doesn't
        bypass access control.
        """
        # QdrantClient, Filter, FieldCondition etc. imported at module level

        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )

        accessible_roles = expand_roles(ctx.user.role)
        target = ctx.target_doc or ""

        # Build a filter that matches either doc_id OR source filename,
        # combined with RBAC.  We try both because the router may extract
        # a human-readable name ("expense policy") while Qdrant stores the
        # filename ("expense_policy.pdf") or a doc_id hash.
        #
        # Strategy: try exact source match first, then fuzzy (MatchText).
        filters_to_try = []

        # Exact filename match (e.g. "expense_policy.pdf")
        filters_to_try.append(Filter(must=[
            FieldCondition(key="allowed_roles", match=MatchAny(any=accessible_roles)),
            FieldCondition(key="source", match=MatchValue(value=target)),
        ]))

        # Without .pdf extension
        clean_name = target.replace(".pdf", "").replace(".docx", "").replace(".md", "")
        if clean_name != target:
            filters_to_try.append(Filter(must=[
                FieldCondition(key="allowed_roles", match=MatchAny(any=accessible_roles)),
                FieldCondition(key="source", match=MatchValue(value=clean_name + ".md")),
            ]))

        # doc_id match (if a stable ID was passed)
        filters_to_try.append(Filter(must=[
            FieldCondition(key="allowed_roles", match=MatchAny(any=accessible_roles)),
            FieldCondition(key="doc_id", match=MatchValue(value=target)),
        ]))

        for qdrant_filter in filters_to_try:
            try:
                results, _ = client.scroll(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    scroll_filter=qdrant_filter,
                    with_payload=True,
                    limit=200,     # generous limit — most docs have < 100 chunks
                )
            except Exception as e:
                logger.warning(f"Qdrant scroll failed: {e}")
                continue

            if results:
                chunks = [
                    {
                        "text": p.payload.get("_node_content", "") or
                                p.payload.get("text", ""),
                        "department": p.payload.get("department", "general"),
                        "allowed_roles": p.payload.get("allowed_roles", []),
                        "page_num": p.payload.get("page_num", 0),
                    }
                    for p in results
                    if p.payload
                ]
                # Sort approximately by page number
                chunks.sort(key=lambda c: c.get("page_num") or 0)

                resolved_name = results[0].payload.get("source", target)
                logger.debug(
                    f"Summarization: found {len(chunks)} chunks for "
                    f"target={target!r} resolved={resolved_name!r}"
                )
                return chunks, resolved_name

        return [], target


# ── Handler 3: DeepRAG ────────────────────────────────────────────────────────

class DeepRagHandler:
    """
    Full 4-layer retrieval pipeline for complex knowledge questions.

    Hybrid search → AutoMerging → Cohere Rerank → GPT-4o

    This is the most expensive path (~1.2-1.8s) and is only triggered when
    the router determines the question needs precise document retrieval.
    """

    def handle(self, ctx: RouteContext) -> RouteResult:
        import time
        start = time.perf_counter()

        if ctx.query_engine is None:
            raise RuntimeError("DeepRagHandler requires ctx.query_engine to be set")

        response = ctx.query_engine.query(ctx.question)
        answer = str(response)
        latency = (time.perf_counter() - start) * 1000

        # Extract sources from LlamaIndex NodeWithScore objects
        sources = []
        seen: set[str] = set()
        for node in getattr(response, "source_nodes", []):
            meta = node.node.metadata
            src = meta.get("source", "unknown")
            if src in seen:
                continue
            seen.add(src)
            sources.append({
                "source": src,
                "department": meta.get("department", "general"),
                "allowed_roles": meta.get("allowed_roles", []),
                "text_snippet": node.node.text[:300].replace("\n", " "),
                "score": round(node.score or 0.0, 4),
            })

        sources.sort(key=lambda s: s["score"], reverse=True)

        logger.info(
            f"DeepRAG route: user={ctx.user.username!r} "
            f"sources={len(sources)} latency={latency:.0f}ms"
        )

        return RouteResult(
            answer=answer,
            route_type="deep_rag",
            sources=sources,
            latency_ms=round(latency, 1),
        )
