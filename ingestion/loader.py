"""
ingestion/loader.py
───────────────────
Loads documents from disk using LlamaParse for rich PDF/DOCX extraction
and SimpleDirectoryReader for plain text / markdown files.

Multimodal upgrade (Feature 3)
────────────────────────────────
When ENABLE_MULTIMODAL=true, this module now:
  1. Configures LlamaParse with output_tables_as_HTML=False so tables are
     preserved as clean markdown (pipe-separated) for regex detection.
  2. Calls process_documents_multimodal() after loading to:
       - Extract + summarise every table via GPT-4o (or gpt-4o-mini).
       - Strip raw tables from the doc text to avoid double-embedding.
  3. Returns both the cleaned documents AND a list of pre-built multimodal
     TextNodes (table summaries + image descriptions).

The downstream pipeline in embedder.py merges these two outputs:
  cleaned docs  → hierarchical chunking → text leaf nodes
  extra nodes   → embedded directly (no re-chunking)

Both sets of nodes are upserted into Qdrant with the same RBAC metadata.

Run directly to test loading:
    python -m ingestion.loader
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import TextNode
from llama_parse import LlamaParse
from loguru import logger

from auth.rbac import get_allowed_roles_for_path
from config import settings


# ── Supported file types ───────────────────────────────────────────────────────

# Files we send through LlamaParse (handles complex layouts / tables / images)
LLAMAPARSE_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}

# Files we load directly (no cloud parser needed)
PLAIN_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".csv"}


def _build_llamaparse_extractor() -> Optional[LlamaParse]:
    """
    Return a configured LlamaParse instance, or None if no API key is set.

    Multimodal configuration notes:
      - result_type="markdown": tables come out as clean | col | col | syntax
        which our regex detector handles reliably.
      - output_tables_as_HTML=False: keeps tables as markdown not HTML —
        our summariser prompt is written for markdown.
      - skip_diagonal_text=True: ignores watermarks / stamps.
      - do_not_unroll_columns=False: preserves multi-column tables.
    """
    if not settings.LLAMA_CLOUD_API_KEY:
        logger.warning(
            "LLAMA_CLOUD_API_KEY not set — falling back to basic PDF loader. "
            "Complex tables and multi-column layouts may not parse correctly."
        )
        return None

    return LlamaParse(
        api_key=settings.LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=False,
        language="en",
        skip_diagonal_text=True,
        do_not_unroll_columns=False,
        # Preserve table structure as markdown — critical for our table extractor
        output_tables_as_HTML=False,
        # Request page-level structure so our approximate page_num tagging works
        page_separator="\f",
    )


def _enrich_metadata(doc: Document, source_dir_root: Path) -> None:
    """
    Stamp source, department, allowed_roles, and ingested_at onto a document.
    Mutates doc.metadata in place.
    """
    file_path = Path(doc.metadata.get("file_path", ""))

    doc.metadata["source"]    = file_path.name
    doc.metadata["file_type"] = file_path.suffix.lower().lstrip(".")

    # Department from folder structure:  .../hr/policy.pdf  →  "hr"
    parts = file_path.parts
    doc.metadata["department"] = parts[-2] if len(parts) >= 2 else "general"

    # RBAC: stamp allowed_roles from department → role mapping
    doc.metadata["allowed_roles"] = get_allowed_roles_for_path(
        doc.metadata["department"]
    )

    doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()


def load_documents(
    docs_dir: Optional[str] = None,
) -> tuple[list[Document], list[TextNode]]:
    """
    Load all documents from `docs_dir` and run multimodal processing.

    Returns:
        (cleaned_documents, multimodal_nodes) where:

        cleaned_documents — Document objects with raw tables stripped out
                            (replaced by placeholders). Pass these into
                            ingestion/chunker.py for hierarchical chunking.

        multimodal_nodes  — TextNodes for table summaries and image
                            descriptions, ready to embed directly.
                            Empty list if ENABLE_MULTIMODAL=false.
    """
    source_dir = Path(docs_dir or settings.DOCS_DIR)

    if not source_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {source_dir}")

    files = [
        f for f in source_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in (LLAMAPARSE_EXTENSIONS | PLAIN_TEXT_EXTENSIONS)
        and not f.name.startswith(".")
    ]

    if not files:
        raise ValueError(f"No supported documents found in {source_dir}")

    logger.info(f"Found {len(files)} document(s) in {source_dir}")

    parser    = _build_llamaparse_extractor()
    file_extractor = {ext: parser for ext in LLAMAPARSE_EXTENSIONS} if parser else {}

    reader = SimpleDirectoryReader(
        input_dir=str(source_dir),
        recursive=True,
        file_extractor=file_extractor,
        filename_as_id=True,
        required_exts=list(LLAMAPARSE_EXTENSIONS | PLAIN_TEXT_EXTENSIONS),
    )

    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} document chunk(s)")

    # Enrich metadata on all documents before multimodal processing so that
    # allowed_roles and doc_id are inherited by table/image nodes.
    for doc in documents:
        _enrich_metadata(doc, source_dir)

    # ── Multimodal processing ──────────────────────────────────────────────────
    if settings.ENABLE_MULTIMODAL:
        from ingestion.multimodal import process_documents_multimodal
        cleaned_docs, multimodal_nodes = process_documents_multimodal(documents)
        logger.info(
            f"Multimodal pipeline: {len(cleaned_docs)} cleaned doc(s), "
            f"{len(multimodal_nodes)} extra node(s)"
        )
        return cleaned_docs, multimodal_nodes

    return documents, []


def load_single_file(
    file_path: str,
) -> tuple[list[Document], list[TextNode]]:
    """
    Load a single file and run multimodal processing.
    Used by upsert_document() and the webhook worker.

    Returns the same (cleaned_documents, multimodal_nodes) tuple as load_documents().
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    parser = _build_llamaparse_extractor()

    reader = SimpleDirectoryReader(
        input_files=[str(path)],
        file_extractor={path.suffix.lower(): parser} if parser else {},
        filename_as_id=True,
    )

    docs = reader.load_data()
    logger.info(f"Loaded {len(docs)} chunk(s) from {path.name}")

    # Minimal metadata enrichment for single-file loads
    for doc in docs:
        _enrich_metadata(doc, path.parent)

    if settings.ENABLE_MULTIMODAL:
        from ingestion.multimodal import process_documents_multimodal
        return process_documents_multimodal(docs)

    return docs, []


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    docs_path = sys.argv[1] if len(sys.argv) > 1 else settings.DOCS_DIR

    docs, mm_nodes = load_documents(docs_path)

    from rich.console import Console
    from rich.table import Table as RichTable

    console = Console()

    # Document table
    tbl = RichTable(title=f"Loaded Documents from '{docs_path}'")
    tbl.add_column("Source", style="cyan")
    tbl.add_column("Department", style="green")
    tbl.add_column("Allowed Roles", style="magenta")
    tbl.add_column("Type", style="yellow")
    tbl.add_column("Chars", justify="right")

    for d in docs:
        tbl.add_row(
            d.metadata.get("source", "—"),
            d.metadata.get("department", "—"),
            str(d.metadata.get("allowed_roles", "—")),
            d.metadata.get("file_type", "—"),
            str(len(d.text)),
        )
    console.print(tbl)

    # Multimodal nodes table
    if mm_nodes:
        mm_tbl = RichTable(title="Multimodal Nodes")
        mm_tbl.add_column("Type", style="cyan")
        mm_tbl.add_column("Source", style="green")
        mm_tbl.add_column("Page", justify="right")
        mm_tbl.add_column("Summary (first 80 chars)", style="yellow")

        for n in mm_nodes:
            mm_tbl.add_row(
                n.metadata.get("content_type", "—"),
                n.metadata.get("source", "—"),
                str(n.metadata.get("page_num", "—")),
                n.text[:80],
            )
        console.print(mm_tbl)
    else:
        console.print("[dim]No multimodal nodes (ENABLE_MULTIMODAL=false or no tables/images found)[/dim]")
