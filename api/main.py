"""
api/main.py
───────────
FastAPI backend — now with agentic routing at the front of every query.

Every POST /query and POST /query/stream request flows through route_query()
which classifies intent then dispatches to the cheapest appropriate handler:

  small_talk    → direct LLM reply, no retrieval  (~150ms)
  summarization → Qdrant scroll + LLM summary     (~500ms)
  deep_rag      → full 4-layer RAG pipeline       (~1500ms)

The route_type field in every response tells the caller which path was taken,
enabling the Streamlit UI to show a routing indicator.

Endpoints:
  POST /auth/token          — Get a JWT (demo login)
  GET  /auth/me             — Inspect current user + role
  GET  /auth/my-roles       — Show expanded RBAC roles
  POST /query               — Agentic query (auth required, RBAC enforced)
  POST /query/stream        — Streaming version (SSE)
  GET  /query/history       — Retrieve conversation history for current session
  DELETE /query/history     — Clear conversation history for current session
  POST /ingest              — Trigger re-ingestion (admin only)
  GET  /health              — Liveness check (public)
  GET  /docs-list           — List indexed docs visible to current user
  DELETE /collection        — Wipe Qdrant collection (admin only)
  POST /webhooks/confluence — Confluence event webhook
  POST /webhooks/sharepoint — SharePoint event webhook
  POST /webhooks/gdrive     — Google Drive event webhook

Demo credentials (password "secret" for all):
    alice=hr  bob=engineering  carol=finance  dave=management
    eve=employee  frank=legal  admin=admin
"""

import json
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
from pydantic import BaseModel, Field

from auth.jwt_handler import (
    CurrentUser,
    TokenResponse,
    get_current_user,
    login,
)
from auth.rbac import expand_roles
from config import settings
from ingestion.embedder import get_or_build_index, build_index
from retrieval.query_engine import (
    build_query_engine_for_user,
    get_index,
    set_index,
)
from retrieval.router import route_query, route_query_stream, get_memory
from webhooks.router import router as webhook_router


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG API — loading index...")
    start = time.perf_counter()
    try:
        index = get_or_build_index()
        set_index(index)
        elapsed = time.perf_counter() - start
        logger.info(f"Index ready in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        raise
    yield
    logger.info("Shutting down RAG API")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Internal Docs RAG API",
    description=(
        "Ask questions about your company's internal documents. "
        "All endpoints except /health and /auth/token require a Bearer JWT. "
        "Queries are automatically routed to the cheapest appropriate pipeline."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(webhook_router)


# ── Request / Response models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=2000,
        example="What is our parental leave policy?",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID. Defaults to username. "
                    "Pass a stable client-side ID to maintain history across page reloads.",
    )
    department_filter: Optional[str] = Field(
        default=None,
        description="Restrict deep-RAG search to a specific department.",
        example="hr",
    )


class SourceDoc(BaseModel):
    source: str
    department: str
    allowed_roles: list[str]
    text_snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    latency_ms: float
    user_role: str
    route_type: str = Field(
        description="Which pipeline handled this query: "
                    "small_talk | summarization | deep_rag"
    )


class IngestRequest(BaseModel):
    docs_dir: Optional[str] = None
    force_rebuild: bool = False


class HealthResponse(BaseModel):
    status: str
    index_ready: bool


# ── Helpers ────────────────────────────────────────────────────────────────────
def _require_admin(user: CurrentUser) -> None:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )


def _build_deep_rag_engine(user: CurrentUser, department_filter: Optional[str] = None):
    """
    Build the per-request RBAC-filtered deep-RAG engine.
    Called only when the router selects the DEEP_RAG path.
    For small_talk and summarization routes this is never called.
    """
    from llama_index.core.vector_stores.types import (
        MetadataFilter, MetadataFilters, FilterOperator, FilterCondition,
    )
    from retrieval.query_engine import SYSTEM_PROMPT

    index = get_index()

    if department_filter:
        accessible_roles = expand_roles(user.role)
        combined = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="allowed_roles",
                    value=accessible_roles,
                    operator=FilterOperator.IN,
                ),
                MetadataFilter(
                    key="department",
                    value=department_filter,
                    operator=FilterOperator.EQ,
                ),
            ],
            condition=FilterCondition.AND,
        )
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import AutoMergingRetriever
        from llama_index.core.postprocessor import SimilarityPostprocessor
        from llama_index.core.response_synthesizers import get_response_synthesizer
        from llama_index.core import PromptTemplate
        from llama_index.postprocessor.cohere_rerank import CohereRerank

        base_ret = index.as_retriever(
            similarity_top_k=settings.TOP_K_RETRIEVAL,
            vector_store_query_mode="hybrid",
            alpha=settings.HYBRID_ALPHA,
            filters=combined,
        )
        retriever = AutoMergingRetriever(
            base_ret, index.storage_context, simple_ratio_thresh=0.5, verbose=False
        )
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.35)]
        if settings.COHERE_API_KEY:
            postprocessors.append(CohereRerank(
                api_key=settings.COHERE_API_KEY,
                top_n=settings.TOP_N_RERANK,
                model="rerank-english-v3.0",
            ))
        qa_tmpl = PromptTemplate(
            f"{SYSTEM_PROMPT}\n\n"
            "---------------------\nCONTEXT DOCUMENTS:\n{context_str}\n"
            "---------------------\nUSER QUESTION: {query_str}\n\n"
            "ANSWER (cite sources using [Source: filename]):"
        )
        synthesizer = get_response_synthesizer(
            response_mode="compact", streaming=True,
            text_qa_template=qa_tmpl, verbose=False,
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=postprocessors,
            response_synthesizer=synthesizer,
        )

    return build_query_engine_for_user(index, user.role)


def _sources_to_model(sources: list[dict]) -> list[SourceDoc]:
    return [
        SourceDoc(
            source=s.get("source", "unknown"),
            department=s.get("department", "general"),
            allowed_roles=s.get("allowed_roles", []),
            text_snippet=s.get("text_snippet", ""),
            score=s.get("score", 0.0),
        )
        for s in sources
    ]


# ── Auth endpoints ─────────────────────────────────────────────────────────────
@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def token(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """Exchange username + password for a signed JWT."""
    return login(form)


@app.get("/auth/me", response_model=CurrentUser, tags=["auth"])
async def me(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    return user


@app.get("/auth/my-roles", tags=["auth"])
async def my_roles(user: CurrentUser = Depends(get_current_user)):
    return {
        "username": user.username,
        "role": user.role,
        "accessible_roles": expand_roles(user.role),
    }


# ── Query endpoints ────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query(
    req: QueryRequest,
    user: CurrentUser = Depends(get_current_user),
) -> QueryResponse:
    """
    Route the query through the intent classifier then dispatch to the
    cheapest appropriate pipeline.

    RBAC is enforced on ALL routes:
      - small_talk: no documents accessed, no RBAC needed
      - summarization: Qdrant scroll filtered by expand_roles(user.role)
      - deep_rag: Qdrant ANN pre-filtered by expand_roles(user.role)

    The route_type field in the response shows which path was taken.
    """
    # Build the deep-RAG engine up-front — the router decides whether to use it.
    # Cost: pure Python object instantiation, no I/O if router picks another route.
    # For small_talk / summarization the engine object is created but never called.
    try:
        query_engine = _build_deep_rag_engine(user, req.department_filter)
    except Exception as e:
        logger.error(f"Failed to build query engine: {e}")
        query_engine = None

    try:
        result = route_query(
            question=req.question,
            user=user,
            session_id=req.session_id,
            query_engine=query_engine,
        )

        logger.info(
            f"Query complete: route={result.route_type!r} "
            f"latency={result.latency_ms:.0f}ms "
            f"user={user.username!r}"
        )

        return QueryResponse(
            answer=result.answer,
            sources=_sources_to_model(result.sources),
            latency_ms=result.latency_ms,
            user_role=user.role,
            route_type=result.route_type,
        )

    except Exception as e:
        logger.error(f"Query failed for user={user.username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", tags=["query"])
async def query_stream(
    req: QueryRequest,
    user: CurrentUser = Depends(get_current_user),
) -> StreamingResponse:
    """
    Streaming version — yields SSE events:
      data: [ROUTE]<route_type>
      data: <token>...
      data: [SOURCES]{...}
      data: [DONE]

    The [ROUTE] event arrives before the first token so the UI can show
    a routing indicator immediately.
    """
    try:
        query_engine = _build_deep_rag_engine(user, req.department_filter)
    except Exception:
        query_engine = None

    async def generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in route_query_stream(
                question=req.question,
                user=user,
                session_id=req.session_id,
                query_engine=query_engine,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"data: [ERROR]{str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/query/history", tags=["query"])
async def conversation_history(
    session_id: Optional[str] = None,
    user: CurrentUser = Depends(get_current_user),
):
    """Return the conversation history for the current session."""
    effective_session = session_id or user.username
    history = get_memory().get(effective_session)
    return {
        "session_id": effective_session,
        "turns": len(history) // 2,
        "messages": [{"role": t.role, "content": t.content} for t in history],
    }


@app.delete("/query/history", tags=["query"])
async def clear_history(
    session_id: Optional[str] = None,
    user: CurrentUser = Depends(get_current_user),
):
    """Clear the conversation history for the current session."""
    effective_session = session_id or user.username
    get_memory().clear(effective_session)
    return {"status": "cleared", "session_id": effective_session}


# ── Admin endpoints ────────────────────────────────────────────────────────────
@app.post("/ingest", tags=["admin"])
async def ingest(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
):
    """Trigger document re-ingestion. Admin role required."""
    _require_admin(user)

    def _run():
        try:
            if req.force_rebuild:
                import shutil
                if os.path.exists(settings.INDEX_PERSIST_DIR):
                    shutil.rmtree(settings.INDEX_PERSIST_DIR)
            new_index = build_index(req.docs_dir)
            set_index(new_index)
            logger.info("Re-ingestion complete — index singleton updated")
        except Exception as e:
            logger.error(f"Re-ingestion failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "ingestion started", "force_rebuild": req.force_rebuild}


@app.delete("/collection", tags=["admin"])
async def delete_collection(user: CurrentUser = Depends(get_current_user)):
    """Wipe the entire Qdrant collection. Admin role required."""
    _require_admin(user)
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
    try:
        client.delete_collection(settings.QDRANT_COLLECTION_NAME)
        logger.warning(f"Collection deleted by {user.username}")
        return {"status": "deleted", "collection": settings.QDRANT_COLLECTION_NAME}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Public endpoints ───────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["public"])
async def health() -> HealthResponse:
    try:
        index = get_index()
        ready = index is not None
    except Exception:
        ready = False
    return HealthResponse(status="ok" if ready else "degraded", index_ready=ready)


@app.get("/docs-list", tags=["query"])
async def docs_list(user: CurrentUser = Depends(get_current_user)):
    """List documents the current user is authorised to see."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
    accessible = expand_roles(user.role)

    try:
        qdrant_filter = Filter(must=[
            FieldCondition(key="allowed_roles", match=MatchAny(any=accessible))
        ])
        results, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=qdrant_filter,
            with_payload=True,
            limit=1000,
        )
        sources: dict[str, dict] = {}
        for point in results:
            meta = point.payload or {}
            src = meta.get("source", "unknown")
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "department": meta.get("department", "general"),
                    "allowed_roles": meta.get("allowed_roles", []),
                    "ingested_at": meta.get("ingested_at", "unknown"),
                }
        return {
            "user_role": user.role,
            "accessible_roles": accessible,
            "count": len(sources),
            "documents": list(sources.values()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
