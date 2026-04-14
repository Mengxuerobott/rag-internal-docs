"""
ingestion/embedder.py
─────────────────────
Builds (or rebuilds) the vector index in Qdrant.

What this module does:
  1. Initialises the Qdrant collection (creates it if it doesn't exist).
  2. Configures the LlamaIndex StorageContext to point at Qdrant.
  3. Embeds every leaf node with OpenAI text-embedding-3-small.
  4. Stores ALL nodes (parents + leaves) in Qdrant so the
     AutoMergingRetriever can walk up from leaf → parent at query time.
  5. Persists the docstore (node relationships) to disk so they survive restarts.

Why text-embedding-3-small?
  - 1536-dim, strong MTEB benchmarks, cheap ($0.02 / 1M tokens as of 2026).
  - Natively supported by LlamaIndex — no custom wrapper needed.
  - Swap to a local model (e.g. nomic-embed-text via Ollama) by changing
    EMBEDDING_MODEL in .env and replacing OpenAIEmbedding with OllamaEmbedding.

Run this module to (re)ingest all documents:
    python -m ingestion.embedder
    python -m ingestion.embedder --docs-dir path/to/docs  # custom directory
"""

import argparse

from llama_index.core import (
    Settings as LISettings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import settings
from ingestion.loader import load_documents
from ingestion.chunker import build_hierarchical_nodes


# ── LlamaIndex global settings ────────────────────────────────────────────────
# Set once here; all LlamaIndex objects created later inherit these.
def configure_llama_index() -> None:
    LISettings.llm = OpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )
    LISettings.embed_model = OpenAIEmbedding(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
        # Batch size: 100 is a safe default for the OpenAI API rate limits.
        embed_batch_size=100,
    )
    # Chunk size for the LLM context window (should match your largest chunk tier)
    LISettings.chunk_size = settings.CHUNK_SIZES[0]
    LISettings.chunk_overlap = 20


# ── Qdrant collection helpers ─────────────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    """Return a connected QdrantClient (local Docker or Qdrant Cloud)."""
    kwargs: dict = {"url": settings.QDRANT_URL}
    if settings.QDRANT_API_KEY:
        kwargs["api_key"] = settings.QDRANT_API_KEY
    return QdrantClient(**kwargs)


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Create the Qdrant collection if it doesn't already exist.
    text-embedding-3-small produces 1536-dim vectors.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        logger.info(f"Qdrant collection '{collection_name}' already exists — skipping creation")
        return

    # Embedding dimension depends on the model
    dim_map = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    dim = dim_map.get(settings.EMBEDDING_MODEL, 1536)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info(f"Created Qdrant collection '{collection_name}' ({dim}d cosine)")


# ── Main ingestion function ───────────────────────────────────────────────────
def build_index(docs_dir: str | None = None) -> VectorStoreIndex:
    """
    Full ingestion pipeline:
      load → chunk → embed → store in Qdrant.

    Returns the VectorStoreIndex (used by the query engine).
    """
    configure_llama_index()

    # 1. Load documents
    logger.info("Step 1/4 — Loading documents")
    documents = load_documents(docs_dir)

    # 2. Build hierarchical nodes
    logger.info("Step 2/4 — Chunking documents")
    all_nodes, leaf_nodes = build_hierarchical_nodes(documents)

    # 3. Set up Qdrant
    logger.info("Step 3/4 — Connecting to Qdrant")
    qdrant_client = get_qdrant_client()
    ensure_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        enable_hybrid=True,      # enables BM25 sparse vectors alongside dense
        batch_size=32,
    )

    # SimpleDocumentStore holds the full node tree (parent + child relationships)
    # so AutoMergingRetriever can walk up from a leaf to its parent at query time.
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    # 4. Embed leaf nodes and index them
    logger.info(f"Step 4/4 — Embedding {len(leaf_nodes)} leaf nodes into Qdrant")
    index = VectorStoreIndex(
        nodes=leaf_nodes,           # only embed the leaves (small, precise chunks)
        storage_context=storage_context,
        show_progress=True,
    )

    # Persist node relationships to disk (for fast restarts without re-embedding)
    import os
    os.makedirs(settings.INDEX_PERSIST_DIR, exist_ok=True)
    storage_context.persist(persist_dir=settings.INDEX_PERSIST_DIR)
    logger.info(f"Index persisted to '{settings.INDEX_PERSIST_DIR}'")

    return index


def load_index() -> VectorStoreIndex:
    """
    Load an already-built index from disk + Qdrant.
    Call this on API startup instead of rebuild_index() to avoid
    re-embedding on every container restart.
    """
    configure_llama_index()

    qdrant_client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        enable_hybrid=True,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=settings.INDEX_PERSIST_DIR,
    )

    index = load_index_from_storage(storage_context)
    logger.info("Index loaded from persisted storage")
    return index


def get_or_build_index(docs_dir: str | None = None) -> VectorStoreIndex:
    """
    Attempt to load a persisted index; fall back to full rebuild if not found.
    This is what the API calls at startup.
    """
    import os

    if os.path.exists(settings.INDEX_PERSIST_DIR) and os.listdir(settings.INDEX_PERSIST_DIR):
        logger.info("Persisted index found — loading from disk")
        try:
            return load_index()
        except Exception as e:
            logger.warning(f"Failed to load persisted index ({e}) — rebuilding")

    logger.info("No persisted index found — running full ingestion pipeline")
    return build_index(docs_dir)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument(
        "--docs-dir",
        default=None,
        help=f"Path to documents directory (default: {settings.DOCS_DIR})",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete existing index and rebuild from scratch",
    )
    args = parser.parse_args()

    if args.force_rebuild:
        import shutil, os
        if os.path.exists(settings.INDEX_PERSIST_DIR):
            shutil.rmtree(settings.INDEX_PERSIST_DIR)
            logger.info("Deleted persisted index — rebuilding from scratch")

    index = build_index(args.docs_dir)
    logger.info("Ingestion complete.")
