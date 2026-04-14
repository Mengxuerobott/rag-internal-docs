"""
api/main.py
───────────
FastAPI backend for the RAG system.

Endpoints:
  POST /query          — Ask a question, returns a streaming response + sources
  POST /ingest         — Trigger re-ingestion (admin use)
  GET  /health         — Liveness check
  GET  /docs-list      — List all indexed documents
  DELETE /collection   — Wipe the Qdrant collection (admin use)

Startup behaviour:
  On first launch, calls get_or_build_index() which will:
  - Load the persisted index from disk if it exists, OR
  - Run the full ingestion pipeline (load → chunk → embed → store)

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from ingestion.embedder import get_or_build_index, build_index
from retrieval.query_engine import get_query_engine


# ── Lifespan: build index on startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build / load the index when the server starts up."""
    logger.info("Starting RAG API — loading index...")
    start = time.perf_counter()
    try:
        index = get_or_build_index()
        get_query_engine(index)          # warm the singleton cache
        elapsed = time.perf_counter() - start
        logger.info(f"Index ready in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        raise

    yield   # server runs here

    logger.info("Shutting down RAG API")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Internal Docs RAG API",
    description="Ask questions about your company's internal documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to answer from company documents.",
        example="What is our remote work policy?",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation tracking in LangSmith.",
    )
    filters: Optional[dict] = Field(
        default=None,
        description=(
            "Optional Qdrant metadata filters. "
            'Example: {"department": "hr"} to only search HR documents.'
        ),
    )


class SourceDoc(BaseModel):
    source: str
    department: str
    file_type: str
    text_snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    latency_ms: float


class IngestRequest(BaseModel):
    docs_dir: Optional[str] = None
    force_rebuild: bool = False


class HealthResponse(BaseModel):
    status: str
    index_ready: bool


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_sources(source_nodes) -> list[SourceDoc]:
    """Convert LlamaIndex NodeWithScore objects into clean SourceDoc dicts."""
    sources = []
    seen_sources = set()   # deduplicate by filename

    for node in source_nodes:
        meta = node.node.metadata
        src = meta.get("source", "unknown")
        if src in seen_sources:
            continue
        seen_sources.add(src)

        sources.append(SourceDoc(
            source=src,
            department=meta.get("department", "general"),
            file_type=meta.get("file_type", "unknown"),
            text_snippet=node.node.text[:300].replace("\n", " "),
            score=round(node.score or 0.0, 4),
        ))

    return sorted(sources, key=lambda s: s.score, reverse=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + readiness check."""
    try:
        engine = get_query_engine()
        ready = engine is not None
    except Exception:
        ready = False

    return HealthResponse(status="ok" if ready else "degraded", index_ready=ready)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Answer a question from the indexed documents (non-streaming).
    Returns the full answer + source citations + latency.
    """
    engine = get_query_engine()
    start = time.perf_counter()

    try:
        # Set metadata filters if provided
        # e.g. {"department": "hr"} → only search HR docs
        if req.filters:
            from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key=k, value=v)
                    for k, v in req.filters.items()
                ]
            )
            engine.retriever._filters = filters

        response = engine.query(req.question)

        latency = (time.perf_counter() - start) * 1000
        logger.info(f"Query answered in {latency:.0f}ms — session={req.session_id}")

        return QueryResponse(
            answer=str(response),
            sources=_extract_sources(response.source_nodes),
            latency_ms=round(latency, 1),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """
    Streaming version of /query.
    Yields answer tokens as Server-Sent Events, then a final JSON chunk
    with source citations.

    The Streamlit UI uses this endpoint so answers appear token-by-token.

    SSE format:
        data: <token>\n\n
        ...
        data: [SOURCES]{"sources": [...]}\n\n
        data: [DONE]\n\n
    """
    engine = get_query_engine()

    async def token_generator() -> AsyncGenerator[str, None]:
        start = time.perf_counter()
        try:
            streaming_response = engine.query(req.question)

            # Stream answer tokens
            for token in streaming_response.response_gen:
                yield f"data: {token}\n\n"

            # Send sources as a final structured event
            sources = _extract_sources(streaming_response.source_nodes)
            sources_payload = json.dumps({"sources": [s.model_dump() for s in sources]})
            yield f"data: [SOURCES]{sources_payload}\n\n"

            latency = (time.perf_counter() - start) * 1000
            logger.info(f"Streaming query answered in {latency:.0f}ms")

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


@app.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger document re-ingestion (runs in background so the API stays responsive).
    Wipes and rebuilds the index if force_rebuild=True.
    """
    def _run_ingestion():
        global _engine_cache
        try:
            if req.force_rebuild:
                import shutil
                if os.path.exists(settings.INDEX_PERSIST_DIR):
                    shutil.rmtree(settings.INDEX_PERSIST_DIR)
                logger.info("Deleted persisted index for forced rebuild")

            index = build_index(req.docs_dir)

            # Update the cached query engine
            from retrieval.query_engine import build_query_engine
            import retrieval.query_engine as qe_module
            qe_module._engine_cache = build_query_engine(index)

            logger.info("Re-ingestion complete — query engine updated")
        except Exception as e:
            logger.error(f"Re-ingestion failed: {e}")

    background_tasks.add_task(_run_ingestion)
    return {"status": "ingestion started", "force_rebuild": req.force_rebuild}


@app.get("/docs-list")
async def docs_list():
    """List all documents currently in the Qdrant collection."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )

    # Scroll through all points and collect unique source filenames
    try:
        results, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            with_payload=True,
            limit=1000,
        )
        sources = {}
        for point in results:
            meta = point.payload or {}
            src = meta.get("source", "unknown")
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "department": meta.get("department", "general"),
                    "file_type": meta.get("file_type", "unknown"),
                    "ingested_at": meta.get("ingested_at", "unknown"),
                }

        return {"count": len(sources), "documents": list(sources.values())}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection")
async def delete_collection():
    """Wipe the entire Qdrant collection. Use with caution — this is irreversible."""
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )
    try:
        client.delete_collection(settings.QDRANT_COLLECTION_NAME)
        logger.warning(f"Collection '{settings.QDRANT_COLLECTION_NAME}' deleted")
        return {"status": "deleted", "collection": settings.QDRANT_COLLECTION_NAME}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
