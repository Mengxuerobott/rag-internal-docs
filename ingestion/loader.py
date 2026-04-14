"""
ingestion/loader.py
───────────────────
Loads documents from disk using LlamaParse for rich PDF/DOCX extraction
and SimpleDirectoryReader for plain text / markdown files.

LlamaParse advantages over naive PDF loaders:
  - Preserves table structure as markdown
  - Handles multi-column layouts
  - Extracts headers / section hierarchy
  - Returns clean markdown that chunkers can split on headings

Run this module directly to test loading:
    python -m ingestion.loader
"""

import os
from pathlib import Path
from typing import Optional

from llama_index.core import SimpleDirectoryReader, Document
from llama_parse import LlamaParse
from loguru import logger

from config import settings


# ── Supported file types ──────────────────────────────────────────────────────

# Files we send through LlamaParse (handles complex layouts / tables)
LLAMAPARSE_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}

# Files we load directly (plain text — no need for a cloud parser)
PLAIN_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".csv"}


def _build_llamaparse_extractor() -> Optional[LlamaParse]:
    """
    Return a configured LlamaParse instance, or None if no API key is set.
    Falls back to SimpleDirectoryReader's built-in PDF reader when unavailable.
    """
    if not settings.LLAMA_CLOUD_API_KEY:
        logger.warning(
            "LLAMA_CLOUD_API_KEY not set — falling back to basic PDF loader. "
            "Complex tables and multi-column layouts may not parse correctly."
        )
        return None

    return LlamaParse(
        api_key=settings.LLAMA_CLOUD_API_KEY,
        result_type="markdown",       # returns clean markdown with headings
        verbose=False,
        language="en",
        skip_diagonal_text=True,      # ignore watermarks / diagonal stamps
        do_not_unroll_columns=False,  # preserve table columns as markdown
    )


def load_documents(docs_dir: Optional[str] = None) -> list[Document]:
    """
    Load all documents from `docs_dir` (defaults to settings.DOCS_DIR).

    Returns a list of LlamaIndex Document objects, each with:
        doc.text        — full extracted text
        doc.metadata    — file_name, file_path, file_type, file_size, etc.
                          (plus any custom metadata injected below)
    """
    source_dir = Path(docs_dir or settings.DOCS_DIR)

    if not source_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {source_dir}")

    files = [
        f for f in source_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in (LLAMAPARSE_EXTENSIONS | PLAIN_TEXT_EXTENSIONS)
        and not f.name.startswith(".")   # skip hidden files
    ]

    if not files:
        raise ValueError(f"No supported documents found in {source_dir}")

    logger.info(f"Found {len(files)} document(s) in {source_dir}")

    parser = _build_llamaparse_extractor()

    # Build file_extractor dict: maps extension → parser instance
    # If LlamaParse is unavailable, fall through to SimpleDirectoryReader defaults
    file_extractor = (
        {ext: parser for ext in LLAMAPARSE_EXTENSIONS}
        if parser
        else {}
    )

    reader = SimpleDirectoryReader(
        input_dir=str(source_dir),
        recursive=True,
        file_extractor=file_extractor,
        filename_as_id=True,        # uses filename as stable doc ID for re-ingestion
        required_exts=list(LLAMAPARSE_EXTENSIONS | PLAIN_TEXT_EXTENSIONS),
    )

    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} document chunk(s)")

    # ── Enrich metadata ───────────────────────────────────────────────────────
    # Add extra fields that will be stored in Qdrant's payload and can be
    # used later for metadata filtering (e.g. filter by department or date).
    for doc in documents:
        file_path = Path(doc.metadata.get("file_path", ""))

        doc.metadata["source"] = file_path.name
        doc.metadata["file_type"] = file_path.suffix.lower().lstrip(".")

        # Extract "department" from folder structure:
        # data/sample_docs/hr/policy.pdf  →  department = "hr"
        parts = file_path.parts
        if len(parts) >= 2:
            doc.metadata["department"] = parts[-2]
        else:
            doc.metadata["department"] = "general"

        # Mark ingest timestamp (ISO 8601)
        from datetime import datetime, timezone
        doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()

    return documents


def load_single_file(file_path: str) -> list[Document]:
    """
    Convenience helper: load a single file by path.
    Useful for incremental ingestion (e.g. triggered by a file watcher).
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
    return docs


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    docs_path = sys.argv[1] if len(sys.argv) > 1 else settings.DOCS_DIR

    docs = load_documents(docs_path)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Loaded Documents from '{docs_path}'")
    table.add_column("Source", style="cyan")
    table.add_column("Department", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Characters", justify="right")

    for d in docs:
        table.add_row(
            d.metadata.get("source", "—"),
            d.metadata.get("department", "—"),
            d.metadata.get("file_type", "—"),
            str(len(d.text)),
        )

    console.print(table)
