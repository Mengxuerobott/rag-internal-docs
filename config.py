"""
config.py
Central configuration — loaded once at startup, imported everywhere.
All values come from environment variables (set via .env or docker-compose).
"""

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── LLM / Embeddings ─────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # ── Document parsing ─────────────────────────────────────────────────────
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

    # ── Reranker ─────────────────────────────────────────────────────────────
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

    # ── Vector store ─────────────────────────────────────────────────────────
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "company_docs")

    # ── Observability ─────────────────────────────────────────────────────────
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "rag-internal-docs")

    # ── Retrieval tuning ──────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    TOP_N_RERANK: int = int(os.getenv("TOP_N_RERANK", "3"))
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Parsed as list of ints: "2048,512,128" → [2048, 512, 128]
    CHUNK_SIZES: list[int] = [
        int(x) for x in os.getenv("CHUNK_SIZES", "2048,512,128").split(",")
    ]

    # ── Multimodal processing ─────────────────────────────────────────────────
    # Set to "false" to skip table summarisation and image description entirely.
    ENABLE_MULTIMODAL: bool = os.getenv("ENABLE_MULTIMODAL", "true").lower() == "true"

    # Model for vision (image description) and table summarisation.
    # gpt-4o-mini is ~10x cheaper than gpt-4o and adequate for most use cases.
    VLM_MODEL: str = os.getenv("VLM_MODEL", "gpt-4o-mini")

    # Max chars of surrounding paragraph sent to the VLM as document context.
    MULTIMODAL_CONTEXT_WINDOW: int = int(os.getenv("MULTIMODAL_CONTEXT_WINDOW", "500"))

    # Max base64 image size in MB. Images larger than this are skipped.
    MULTIMODAL_MAX_IMAGE_MB: float = float(os.getenv("MULTIMODAL_MAX_IMAGE_MB", "4.0"))

    # ── Auth / JWT ────────────────────────────────────────────────────────────
    # IMPORTANT: set a strong random secret in production.
    # Generate one with:  python -c "import secrets; print(secrets.token_hex(32))"
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production-use-secrets-token-hex-32")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))  # 8 hours

    # ── Event-driven ingestion (Redis + ARQ) ──────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "4"))
    WORKER_MAX_RETRIES: int = int(os.getenv("WORKER_MAX_RETRIES", "3"))
    WORKER_RETRY_DELAY_S: int = int(os.getenv("WORKER_RETRY_DELAY_S", "30"))

    # ── Webhook HMAC secrets ──────────────────────────────────────────────────
    WEBHOOK_SECRET_CONFLUENCE: str = os.getenv("WEBHOOK_SECRET_CONFLUENCE", "")
    WEBHOOK_SECRET_SHAREPOINT: str = os.getenv("WEBHOOK_SECRET_SHAREPOINT", "")
    WEBHOOK_SECRET_GDRIVE: str = os.getenv("WEBHOOK_SECRET_GDRIVE", "")

    # ── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # ── Directories ───────────────────────────────────────────────────────────
    DOCS_DIR: str = os.getenv("DOCS_DIR", "data/sample_docs")
    INDEX_PERSIST_DIR: str = os.getenv("INDEX_PERSIST_DIR", "data/index_store")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


settings = get_settings()
