"""
api/main.py
───────────
FastAPI backend for the RAG system.

Endpoints:
  POST /auth/token        — Get a JWT (demo login)
  GET  /auth/me           — Inspect current user + role
  GET  /auth/my-roles     — Show expanded RBAC roles
  POST /query             — Ask a question (auth required, RBAC enforced)
  POST /query/stream      — Streaming version (SSE)
  POST /ingest            — Trigger re-ingestion (admin only)
  GET  /health            — Liveness check (public)
  GET  /docs-list         — List indexed docs visible to current user
  DELETE /collection      — Wipe Qdrant collection (admin only)
  POST /webhooks/confluence — Confluence event webhook
  POST /webhooks/sharepoint — SharePoint event webhook
  POST /webhooks/gdrive     — Google Drive event webhook

RBAC flow per request:
  1. FastAPI's OAuth2PasswordBearer reads Authorization: Bearer <jwt>.
  2. get_current_user() verifies the JWT and returns CurrentUser(role=...).
  3. build_query_engine_for_user(index, role) injects a Qdrant pre-filter
     that restricts the ANN search to chunks the user's role can access.
  4. The LLM only ever sees authorised chunks — no post-filtering leakage.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

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
        "All endpoints except /health and /auth/token require a Bearer JWT."
    ),
    version="3.0.0",
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
    session_id: Optional[str] = Field(default=None)
    department_filter: Optional[str] = Field(
        default=None,
        description="Restrict search to a specific department.",
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


def _extract_sources(source_nodes) -> list[SourceDoc]:
    sources, seen = [], set()
    for node in source_nodes:
        meta = node.node.metadata
        src = meta.get("source", "unknown")
        if src in seen:
            continue
        seen.add(src)
        sources.append(SourceDoc(
            source=src,
            department=meta.get("department", "general"),
            allowed_roles=meta.get("allowed_roles", []),
            text_snippet=node.node.text[:300].replace("\n", " "),
            score=round(node.score or 0.0, 4),
        ))
    return sorted(sources, key=lambda s: s.score, reverse=True)


def _build_engine_for(user: CurrentUser, department_filter: Optional[str] = None):
    """
    Build a per-request RBAC-filtered query engine.
    Optionally layers an additional department filter on top of the RBAC filter.
    """
    from llama_index.core.vector_stores.types import (
        MetadataFilter, MetadataFilters, FilterOperator, FilterCondition,
    )

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
        from retrieval.query_engine import SYSTEM_PROMPT

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
    Answer a question from the indexed documents.
    RBAC enforced: only chunks accessible to the user's role are considered.
    """
    engine = _build_engine_for(user, req.department_filter)
    start = time.perf_counter()

    try:
        response = engine.query(req.question)
        latency = (time.perf_counter() - start) * 1000
        logger.info(
            f"Query answered in {latency:.0f}ms — "
            f"user={user.username!r} role={user.role!r}"
        )
        return QueryResponse(
            answer=str(response),
            sources=_extract_sources(response.source_nodes),
            latency_ms=round(latency, 1),
            user_role=user.role,
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
    Streaming version of /query — yields SSE tokens then a sources event.
    RBAC is enforced identically to /query.

    SSE format:
        data: <token>\\n\\n
        ...
        data: [SOURCES]{...}\\n\\n
        data: [DONE]\\n\\n
    """
    engine = _build_engine_for(user, req.department_filter)

    async def token_generator() -> AsyncGenerator[str, None]:
        start = time.perf_counter()
        try:
            streaming_response = engine.query(req.question)
            for token in streaming_response.response_gen:
                yield f"data: {token}\n\n"

            sources = _extract_sources(streaming_response.source_nodes)
            sources_payload = json.dumps({"sources": [s.model_dump() for s in sources]})
            yield f"data: [SOURCES]{sources_payload}\n\n"

            latency = (time.perf_counter() - start) * 1000
            logger.info(
                f"Stream answered in {latency:.0f}ms — "
                f"user={user.username!r} role={user.role!r}"
            )
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"data: [ERROR]{str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
