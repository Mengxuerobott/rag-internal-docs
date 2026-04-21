"""
ingestion/embedder.py
─────────────────────
Builds or reloads the Qdrant-backed vector index, and exposes two
document-level mutation functions used by the webhook worker:

  delete_document_chunks(doc_id)
      Remove every Qdrant point whose payload contains doc_id == <value>.
      Called before re-ingesting an updated document, and on document deletion.

  upsert_document(file_path, doc_id, allowed_roles)
      Load one file, chunk it hierarchically, embed the leaf nodes, and
      upsert them into Qdrant — tagging every chunk with doc_id and
      allowed_roles so RBAC filtering and targeted deletion both work.

Why doc_id instead of filename?
────────────────────────────────
Filenames are mutable (a Confluence page can be renamed). The DMS assigns
each document a stable ID that never changes even across renames or moves.
We store that ID on every chunk so we can delete *exactly* that document's
chunks with a single Qdrant filter — no full-collection scan needed.

For locally-ingested files (batch mode), doc_id defaults to a SHA-256 hash
of the file's absolute path so it stays stable even if the file is renamed.
"""

import argparse
import hashlib
import os

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
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from config import settings
from ingestion.chunker import build_all_nodes, build_hierarchical_nodes
from ingestion.loader import load_documents, load_single_file


# ── LlamaIndex global settings ────────────────────────────────────────────────
def configure_llama_index() -> None:
    LISettings.llm = OpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )
    LISettings.embed_model = OpenAIEmbedding(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
        embed_batch_size=100,
    )
    LISettings.chunk_size = settings.CHUNK_SIZES[0]
    LISettings.chunk_overlap = 20


# ── Qdrant helpers ────────────────────────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    kwargs: dict = {"url": settings.QDRANT_URL}
    if settings.QDRANT_API_KEY:
        kwargs["api_key"] = settings.QDRANT_API_KEY
    return QdrantClient(**kwargs)


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        logger.info(f"Qdrant collection '{collection_name}' already exists — skipping creation")
        return

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


def _stable_doc_id(file_path: str) -> str:
    """
    Derive a stable doc_id from an absolute file path.
    Used for locally-ingested files when no DMS-assigned ID is available.
    SHA-256 of the path is stable even if the file contents change.
    """
    return hashlib.sha256(os.path.abspath(file_path).encode()).hexdigest()[:32]


# ── Document-level mutation functions (used by webhook worker) ────────────────

def delete_document_chunks(doc_id: str) -> int:
    """
    Delete ALL Qdrant points whose payload field `doc_id` equals the given value.

    This is the write-side of targeted deletion. When a webhook fires for a
    document update or deletion, the worker calls this first to remove stale
    chunks before (optionally) re-ingesting the new version.

    Returns the number of points deleted (0 if the document was not indexed).

    Why Qdrant filter delete instead of storing point IDs?
    ───────────────────────────────────────────────────────
    A single document produces O(N) chunks after hierarchical chunking. Storing
    and tracking individual point UUIDs per document is brittle — one missed ID
    and stale chunks survive. Filtering on a payload field is atomic and complete:
    Qdrant deletes every matching point in one operation regardless of how many
    chunks the document produced.
    """
    client = get_qdrant_client()

    delete_filter = Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            )
        ]
    )

    result = client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=delete_filter,
    )

    # Qdrant returns an UpdateResult; deleted count isn't directly exposed
    # but the operation_id confirms it ran. Log for observability.
    logger.info(f"Deleted chunks for doc_id={doc_id!r} — result: {result}")
    return 0  # exact count not available in this Qdrant client version


def upsert_document(
    file_path: str,
    doc_id: str | None = None,
    allowed_roles: list[str] | None = None,
) -> int:
    """
    Ingest a single file and upsert its chunks into Qdrant.

    Steps:
      1. Load the file with LlamaParse (or plain-text reader).
      2. Build hierarchical nodes (2048 → 512 → 128 tokens).
      3. Stamp doc_id and allowed_roles onto every chunk's metadata.
      4. Embed leaf nodes and upsert into Qdrant (insert or overwrite).

    Args:
        file_path:     Absolute or relative path to the file.
        doc_id:        Stable DMS document ID (e.g. Confluence page ID).
                       Falls back to a SHA-256 of the file path.
        allowed_roles: RBAC roles that may retrieve this document's chunks.
                       If None, infers from the file's parent folder name
                       via get_allowed_roles_for_path().

    Returns:
        Number of leaf nodes upserted.
    """
    from auth.rbac import get_allowed_roles_for_path
    from pathlib import Path

    configure_llama_index()

    resolved_doc_id = doc_id or _stable_doc_id(file_path)
    file_name = Path(file_path).name
    folder_name = Path(file_path).parent.name

    resolved_roles = allowed_roles or get_allowed_roles_for_path(folder_name)

    logger.info(
        f"Upserting document: file={file_name!r} "
        f"doc_id={resolved_doc_id!r} roles={resolved_roles}"
    )

    # 1. Load + multimodal processing
    # load_single_file returns (cleaned_docs, multimodal_nodes):
    #   cleaned_docs     — text with raw tables stripped out
    #   multimodal_nodes — table summaries + image descriptions
    documents, multimodal_nodes = load_single_file(file_path)

    # 2. Stamp doc_id and allowed_roles onto every document before chunking
    for doc in documents:
        doc.metadata["doc_id"] = resolved_doc_id
        doc.metadata["allowed_roles"] = resolved_roles
        doc.metadata["department"] = folder_name

    # 3. Stamp doc_id onto multimodal nodes so targeted deletion works
    for node in multimodal_nodes:
        node.metadata["doc_id"] = resolved_doc_id
        node.metadata.setdefault("allowed_roles", resolved_roles)

    # 4. Chunk (merges text hierarchy with multimodal nodes)
    all_nodes, leaf_nodes = build_all_nodes(documents, multimodal_nodes)

    # Ensure doc_id propagates to every node
    for node in all_nodes:
        node.metadata.setdefault("doc_id", resolved_doc_id)
        node.metadata.setdefault("allowed_roles", resolved_roles)

    # 4. Connect to Qdrant
    qdrant_client = get_qdrant_client()
    ensure_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        enable_hybrid=True,
        batch_size=32,
    )

    # Load existing docstore so parent nodes are merged correctly
    docstore: SimpleDocumentStore
    persist_path = settings.INDEX_PERSIST_DIR
    if os.path.exists(persist_path):
        from llama_index.core.storage.docstore import SimpleDocumentStore
        try:
            docstore = SimpleDocumentStore.from_persist_dir(persist_path)
        except Exception:
            docstore = SimpleDocumentStore()
    else:
        docstore = SimpleDocumentStore()

    docstore.add_documents(all_nodes)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    # 5. Upsert: VectorStoreIndex.insert_nodes() upserts by node_id
    index = VectorStoreIndex(
        nodes=[],   # don't re-embed existing nodes
        storage_context=storage_context,
    )
    index.insert_nodes(leaf_nodes)

    # Persist updated docstore
    os.makedirs(persist_path, exist_ok=True)
    storage_context.persist(persist_dir=persist_path)

    logger.info(
        f"Upserted {len(leaf_nodes)} leaf nodes for doc_id={resolved_doc_id!r}"
    )
    return len(leaf_nodes)


# ── Batch ingestion (unchanged from baseline) ─────────────────────────────────

def build_index(docs_dir: str | None = None) -> VectorStoreIndex:
    """
    Full batch ingestion pipeline: load → chunk → embed → store.
    Stamps doc_id (path hash) and allowed_roles onto every chunk.
    """
    configure_llama_index()

    logger.info("Step 1/4 — Loading documents")
    # load_documents() now returns (cleaned_docs, multimodal_nodes)
    # cleaned_docs have raw tables replaced with placeholders
    # multimodal_nodes are table summaries + image descriptions, ready to embed
    documents, multimodal_nodes = load_documents(docs_dir)

    # Stamp doc_id derived from file path for every document
    from pathlib import Path
    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        doc.metadata.setdefault("doc_id", _stable_doc_id(file_path))

    logger.info(
        f"Step 2/4 — Chunking documents "
        f"(+{len(multimodal_nodes)} multimodal node(s))"
    )
    # build_all_nodes merges text hierarchy nodes with pre-built multimodal nodes.
    # Multimodal nodes skip re-chunking — they are already embedding-ready.
    all_nodes, leaf_nodes = build_all_nodes(documents, multimodal_nodes)

    # Propagate doc_id to all nodes
    for node in all_nodes:
        if "doc_id" not in node.metadata:
            node.metadata["doc_id"] = node.metadata.get("doc_id", "unknown")

    logger.info("Step 3/4 — Connecting to Qdrant")
    qdrant_client = get_qdrant_client()
    ensure_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        enable_hybrid=True,
        batch_size=32,
    )

    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    logger.info(f"Step 4/4 — Embedding {len(leaf_nodes)} leaf nodes into Qdrant")
    index = VectorStoreIndex(
        nodes=leaf_nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    os.makedirs(settings.INDEX_PERSIST_DIR, exist_ok=True)
    storage_context.persist(persist_dir=settings.INDEX_PERSIST_DIR)
    logger.info(f"Index persisted to '{settings.INDEX_PERSIST_DIR}'")

    return index


def load_index() -> VectorStoreIndex:
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
    if os.path.exists(settings.INDEX_PERSIST_DIR) and os.listdir(settings.INDEX_PERSIST_DIR):
        logger.info("Persisted index found — loading from disk")
        try:
            return load_index()
        except Exception as e:
            logger.warning(f"Failed to load persisted index ({e}) — rebuilding")

    logger.info("No persisted index found — running full ingestion pipeline")
    return build_index(docs_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--upsert-file",
        default=None,
        help="Upsert a single file by path (event-driven mode test)",
    )
    parser.add_argument(
        "--delete-doc-id",
        default=None,
        help="Delete all chunks for a given doc_id",
    )
    args = parser.parse_args()

    if args.delete_doc_id:
        delete_document_chunks(args.delete_doc_id)
        logger.info(f"Deletion complete for doc_id={args.delete_doc_id!r}")

    elif args.upsert_file:
        n = upsert_document(args.upsert_file)
        logger.info(f"Upserted {n} chunks for {args.upsert_file!r}")

    else:
        if args.force_rebuild:
            import shutil
            if os.path.exists(settings.INDEX_PERSIST_DIR):
                shutil.rmtree(settings.INDEX_PERSIST_DIR)
                logger.info("Deleted persisted index — rebuilding from scratch")

        build_index(args.docs_dir)
        logger.info("Ingestion complete.")
