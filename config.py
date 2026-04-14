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
