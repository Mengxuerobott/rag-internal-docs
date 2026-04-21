"""
ingestion/multimodal.py
────────────────────────
Converts tables and images inside parsed documents into semantically rich,
embeddable TextNodes — rather than raw markdown or raw pixel data.

The problem this solves
───────────────────────
Standard text chunkers split on token boundaries.  A 40-row expense table
might get split halfway through row 14.  The LLM then sees an incomplete
table with no column headers and hallucinates the missing data.

Even if the table survives the split intact, embedding raw markdown like:
  "| Q1 | Q2 | Q3 | Q4 |\n|---|---|---|---|\n| 12 | 18 | 9 | 22 |"
produces a mediocre vector — the embedding model sees pipe characters and
numbers without the semantic context of "quarterly revenue figures".

The solution: summarise the table (or image) once during ingestion with a
capable LLM and embed the natural-language SUMMARY instead.  The summary
vector is rich and precise; the full table text is stored in the node for
the LLM to read during answer generation.

Pipeline overview
──────────────────
For every document coming out of the loader:

  Text document (markdown from LlamaParse)
       │
       ├─► extract_markdown_tables()
       │       │ table_1_markdown, context_before
       │       │ table_2_markdown, context_before
       │       └─► summarize_table_with_vlm()  → TextNode(content_type="table")
       │
       ├─► strip tables from original doc text (avoid double-embedding)
       │
       └─► [cleaned text doc] ──► normal hierarchical chunking
                                  (in ingestion/chunker.py)

For PDF/DOCX image blobs (extracted by LlamaParse JSON mode):

  image_blob (base64 PNG/JPEG)
       └─► describe_image_with_vlm()  → TextNode(content_type="image")

The TextNodes returned by this module carry:
  node.text           — the summary / description (what gets embedded)
  node.metadata       — content_type, source, page_num, original_table /
                        original_image_b64 (what the LLM reads at answer time)

Interview talking points
────────────────────────
"I separate WHAT GETS EMBEDDED from WHAT THE LLM READS. The summary is
 optimised for retrieval precision. The original table is stored in the
 metadata and injected into the LLM prompt verbatim so it can read the
 real numbers — not just the description."

"Table summarisation costs about $0.001 per table on gpt-4o-mini.
 For a 500-document corpus with ~3 tables per document, that's $1.50 —
 a one-time cost that dramatically improves retrieval quality."
"""

from __future__ import annotations

import base64
import re
import uuid
from typing import NamedTuple

from llama_index.core.schema import TextNode
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


# ── OpenAI client ─────────────────────────────────────────────────────────────
def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


# ── Data classes ──────────────────────────────────────────────────────────────

class ExtractedTable(NamedTuple):
    markdown: str      # raw markdown table text  "| col | col |\n|---|..."
    context:  str      # up to MULTIMODAL_CONTEXT_WINDOW chars before the table
    page_num: int      # 0-indexed page number (best effort)
    char_offset: int   # position in original doc text


class ExtractedImage(NamedTuple):
    b64_data:  str     # base64-encoded PNG/JPEG
    mime_type: str     # "image/png" | "image/jpeg"
    context:   str     # surrounding text caption or empty string
    page_num:  int


# ── Table extraction ──────────────────────────────────────────────────────────

# Matches a complete markdown table: header row, separator row, one or more data rows.
# Groups:
#   group 0 = full table string (header + separator + data rows)
_TABLE_RE = re.compile(
    r"(?m)"                          # multiline
    r"^\|.+\|\s*\n"                  # header row
    r"^\|[\s\-:|]+\|\s*\n"           # separator row  | --- | :---: |
    r"(?:^\|.+\|\s*\n)+",            # 1+ data rows
)


def extract_markdown_tables(
    text: str,
    context_window: int | None = None,
) -> list[ExtractedTable]:
    """
    Find all markdown tables in `text` and return them with surrounding context.

    Args:
        text:           Full document text (LlamaParse markdown output).
        context_window: How many characters before the table to include as
                        context for the VLM.  Defaults to settings.MULTIMODAL_CONTEXT_WINDOW.

    Returns:
        List of ExtractedTable objects.  Empty list if no tables found.
    """
    ctx_len = context_window or settings.MULTIMODAL_CONTEXT_WINDOW
    tables: list[ExtractedTable] = []

    for match in _TABLE_RE.finditer(text):
        start = match.start()
        table_md = match.group(0)

        # Grab the `ctx_len` characters immediately before the table as context.
        context_start = max(0, start - ctx_len)
        context = text[context_start:start].strip()

        # Approximate page number: count newlines before this table × 0.02
        # (very rough heuristic — good enough for metadata tagging).
        page_approx = text[:start].count("\f")  # form-feed = page break in some outputs

        tables.append(ExtractedTable(
            markdown=table_md.strip(),
            context=context,
            page_num=page_approx,
            char_offset=start,
        ))

    logger.debug(f"Found {len(tables)} markdown table(s)")
    return tables


def strip_tables_from_text(text: str) -> str:
    """
    Remove all markdown tables from `text`, replacing each with a
    one-line placeholder so surrounding paragraphs stay readable.

    The placeholder is kept so the hierarchical chunker sees that a table
    existed here, which helps the LLM answer questions like "what comes
    after the table on page 3".
    """
    def _replace(m: re.Match) -> str:
        row_count = m.group(0).count("\n") - 2  # subtract header + separator rows
        return f"[TABLE: {row_count} rows — see separate table summary node]\n"

    return _TABLE_RE.sub(_replace, text)


# ── Table summarisation ───────────────────────────────────────────────────────

_TABLE_SYSTEM_PROMPT = """\
You are a precise data analyst converting tables into dense semantic summaries
for a RAG retrieval system. Your summaries must be optimised for vector search —
include all key entities, numbers, dates, column names, and relationships.
Do NOT include formatting or markdown. Write in plain prose."""

_TABLE_USER_TMPL = """\
DOCUMENT CONTEXT (the paragraph before this table):
{context}

MARKDOWN TABLE:
{table_md}

Write a dense 2-5 sentence summary that captures:
1. What the table is about (subject / title)
2. What each column represents
3. Key figures, ranges, or patterns in the data
4. Any notable outliers or conclusions a reader would draw

Summary:"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def summarize_table_with_vlm(
    table: ExtractedTable,
    source_file: str = "",
    doc_metadata: dict | None = None,
) -> TextNode:
    """
    Send a markdown table to the VLM and return a TextNode containing
    the natural-language summary as its embeddable text.

    The original markdown table is stored in node.metadata["original_table"]
    so the LLM can read the actual data during answer synthesis — the summary
    is only used for retrieval.

    Args:
        table:        ExtractedTable from extract_markdown_tables().
        source_file:  Source filename for metadata (e.g. "expense_policy.pdf").
        doc_metadata: Additional metadata from the parent Document to propagate.

    Returns:
        TextNode ready to be embedded and upserted into Qdrant.
    """
    client = _get_openai_client()
    prompt = _TABLE_USER_TMPL.format(
        context=table.context[:settings.MULTIMODAL_CONTEXT_WINDOW] if table.context else "(no context)",
        table_md=table.markdown,
    )

    logger.debug(f"Summarising table (rows≈{table.markdown.count(chr(10))}) from {source_file!r}")

    response = client.chat.completions.create(
        model=settings.VLM_MODEL,
        messages=[
            {"role": "system", "content": _TABLE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=300,
        temperature=0.1,
    )

    summary = response.choices[0].message.content.strip()

    # Build metadata — propagate parent doc fields so RBAC and doc_id filters work
    metadata = {
        "content_type": "table",
        "source": source_file,
        "page_num": table.page_num,
        "original_table": table.markdown,   # stored for LLM answer generation
        "table_context": table.context[:200],
    }
    if doc_metadata:
        # Propagate critical fields: doc_id, allowed_roles, department, ingested_at
        for key in ("doc_id", "allowed_roles", "department", "ingested_at", "source"):
            if key in doc_metadata:
                metadata.setdefault(key, doc_metadata[key])

    node = TextNode(
        id_=str(uuid.uuid4()),
        text=summary,
        metadata=metadata,
    )

    logger.debug(f"Table summary ({len(summary)} chars): {summary[:80]}…")
    return node


# ── Image description ─────────────────────────────────────────────────────────

_IMAGE_SYSTEM_PROMPT = """\
You are a precise technical analyst describing charts, diagrams, and figures
from business documents for a RAG retrieval system. Your descriptions must be
optimised for vector search — include all visible text, axis labels, legend
entries, key data points, trends, and conclusions. Write in plain prose."""

_IMAGE_USER_TMPL = """\
DOCUMENT CONTEXT (caption or surrounding text):
{context}

Describe this image/chart in detail (3-6 sentences). Include:
- Chart/image type (bar chart, pie chart, org chart, photo, diagram, etc.)
- All visible text labels, axis names, legend entries, and title
- Key data points, values, or ranges shown
- The main trend, pattern, or conclusion a reader would draw
- Any notable outliers or annotations

Description:"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def describe_image_with_vlm(
    image: ExtractedImage,
    source_file: str = "",
    doc_metadata: dict | None = None,
) -> TextNode | None:
    """
    Send a base64-encoded image to GPT-4o vision and return a TextNode
    containing the dense natural-language description as embeddable text.

    The original base64 image data is stored in node.metadata["image_b64"]
    so the answer-generation step can optionally pass it back to the VLM
    for precise visual question answering.

    Args:
        image:        ExtractedImage from the loader.
        source_file:  Source filename for metadata.
        doc_metadata: Additional metadata to propagate from parent Document.

    Returns:
        TextNode, or None if the image is too large or the API call fails.
    """
    # Size gate: skip images > MULTIMODAL_MAX_IMAGE_MB to avoid huge API bills
    size_mb = len(image.b64_data) * 3 / 4 / (1024 * 1024)  # approx decoded size
    if size_mb > settings.MULTIMODAL_MAX_IMAGE_MB:
        logger.warning(
            f"Skipping image from {source_file!r} — "
            f"size {size_mb:.1f}MB > limit {settings.MULTIMODAL_MAX_IMAGE_MB}MB"
        )
        return None

    client = _get_openai_client()

    logger.debug(f"Describing image ({size_mb:.2f}MB) from {source_file!r}")

    try:
        response = client.chat.completions.create(
            model=settings.VLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": _IMAGE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _IMAGE_USER_TMPL.format(
                                context=image.context[:200] if image.context else "(no caption)"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image.mime_type};base64,{image.b64_data}",
                                "detail": "high",   # use high-detail for charts with small text
                            },
                        },
                    ],
                },
            ],
            max_tokens=400,
            temperature=0.1,
        )
    except Exception as e:
        logger.error(f"VLM image description failed for {source_file!r}: {e}")
        return None

    description = response.choices[0].message.content.strip()

    metadata = {
        "content_type": "image",
        "source": source_file,
        "page_num": image.page_num,
        "mime_type": image.mime_type,
        "image_b64": image.b64_data,    # stored for re-use in answer synthesis
        "image_context": image.context[:200],
    }
    if doc_metadata:
        for key in ("doc_id", "allowed_roles", "department", "ingested_at", "source"):
            if key in doc_metadata:
                metadata.setdefault(key, doc_metadata[key])

    node = TextNode(
        id_=str(uuid.uuid4()),
        text=description,
        metadata=metadata,
    )

    logger.debug(f"Image description ({len(description)} chars): {description[:80]}…")
    return node


# ── Document-level orchestrator ───────────────────────────────────────────────

def process_document_multimodal(
    doc,                               # llama_index.core.Document
    images: list[ExtractedImage] | None = None,
) -> tuple:
    """
    Process one Document through the multimodal pipeline.

    Steps:
      1. Extract all markdown tables from doc.text.
      2. For each table: generate a VLM summary → TextNode.
      3. Strip the raw tables from doc.text to avoid double-embedding.
      4. For each image blob: generate a VLM description → TextNode.
      5. Return the cleaned document + all extra nodes.

    Args:
        doc:    A LlamaIndex Document (output of load_documents / load_single_file).
        images: Optional list of ExtractedImage objects from the same document.
                Pass these when using LlamaParse JSON mode or a separate image
                extractor.  If None, only table processing runs.

    Returns:
        (cleaned_doc, extra_nodes) where:
          cleaned_doc  — Document with raw tables replaced by placeholders.
                         This goes through normal hierarchical text chunking.
          extra_nodes  — List of TextNodes (table summaries + image descriptions).
                         These are embedded as-is — no further chunking.
    """
    if not settings.ENABLE_MULTIMODAL:
        return doc, []

    source = doc.metadata.get("source", "unknown")
    doc_meta = doc.metadata
    extra_nodes: list[TextNode] = []

    # ── Tables ────────────────────────────────────────────────────────────────
    tables = extract_markdown_tables(doc.text)

    if tables:
        logger.info(f"Processing {len(tables)} table(s) from {source!r}")
        for table in tables:
            try:
                node = summarize_table_with_vlm(
                    table=table,
                    source_file=source,
                    doc_metadata=doc_meta,
                )
                extra_nodes.append(node)
            except Exception as e:
                logger.error(f"Table summarisation failed ({source!r}): {e}")

        # Strip raw tables from the document text so the hierarchical chunker
        # doesn't embed the same information twice (once as summary, once as
        # raw markdown fragments that may be split mid-row).
        doc.text = strip_tables_from_text(doc.text)
        logger.debug(f"Stripped {len(tables)} table(s) from {source!r} text")

    # ── Images ────────────────────────────────────────────────────────────────
    if images:
        logger.info(f"Processing {len(images)} image(s) from {source!r}")
        for img in images:
            try:
                node = describe_image_with_vlm(
                    image=img,
                    source_file=source,
                    doc_metadata=doc_meta,
                )
                if node:
                    extra_nodes.append(node)
            except Exception as e:
                logger.error(f"Image description failed ({source!r}): {e}")

    n_tables = sum(1 for n in extra_nodes if n.metadata.get("content_type") == "table")
    n_images = sum(1 for n in extra_nodes if n.metadata.get("content_type") == "image")
    logger.info(
        f"Multimodal processing complete for {source!r}: "
        f"{n_tables} table node(s), {n_images} image node(s)"
    )

    return doc, extra_nodes


def process_documents_multimodal(
    documents: list,
    images_by_doc_id: dict[str, list[ExtractedImage]] | None = None,
) -> tuple[list, list[TextNode]]:
    """
    Run multimodal processing across a list of Documents.

    Args:
        documents:        Output of load_documents() or load_single_file().
        images_by_doc_id: Optional mapping of doc_id → list[ExtractedImage].
                          When using LlamaParse JSON mode, pass extracted images here.

    Returns:
        (cleaned_documents, all_multimodal_nodes)
    """
    if not settings.ENABLE_MULTIMODAL:
        logger.info("ENABLE_MULTIMODAL=false — skipping multimodal processing")
        return documents, []

    cleaned_docs = []
    all_extra_nodes: list[TextNode] = []

    for doc in documents:
        doc_id = doc.metadata.get("doc_id", doc.doc_id)
        images = (images_by_doc_id or {}).get(doc_id, [])

        cleaned_doc, extra_nodes = process_document_multimodal(doc, images)
        cleaned_docs.append(cleaned_doc)
        all_extra_nodes.extend(extra_nodes)

    logger.info(
        f"Multimodal pipeline complete — "
        f"{len(cleaned_docs)} document(s), "
        f"{len(all_extra_nodes)} extra node(s) generated"
    )

    return cleaned_docs, all_extra_nodes


# ── LlamaParse JSON image extractor ──────────────────────────────────────────

def extract_images_from_llamaparse_json(json_result: list) -> dict[str, list[ExtractedImage]]:
    """
    Extract image blobs from LlamaParse JSON mode output and group them
    by document ID.

    LlamaParse JSON mode returns a list of Document objects where each
    Document's text field is a JSON string with this structure:
    {
      "pages": [
        {
          "page": 1,
          "md": "...",
          "images": [
            {"name": "img_0.png", "data": "<base64>", "width": 400, "height": 300}
          ]
        }
      ]
    }

    Args:
        json_result: List of Document objects returned by LlamaParse in JSON mode.

    Returns:
        Dict mapping doc_id → list[ExtractedImage].
    """
    import json

    result: dict[str, list[ExtractedImage]] = {}

    for doc in json_result:
        doc_id = doc.metadata.get("doc_id", doc.doc_id)
        images: list[ExtractedImage] = []

        try:
            pages_data = json.loads(doc.text)
            pages = pages_data.get("pages", [])

            for page in pages:
                page_num = page.get("page", 0) - 1  # convert to 0-indexed
                page_md  = page.get("md", "")

                for img_obj in page.get("images", []):
                    b64 = img_obj.get("data", "")
                    if not b64:
                        continue

                    img_name = img_obj.get("name", "")
                    mime = "image/png" if img_name.endswith(".png") else "image/jpeg"

                    # Use surrounding markdown text as context for the VLM
                    context = page_md[:200].strip()

                    images.append(ExtractedImage(
                        b64_data=b64,
                        mime_type=mime,
                        context=context,
                        page_num=page_num,
                    ))

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Could not parse LlamaParse JSON for doc {doc_id!r}: {e}")

        if images:
            result[doc_id] = images
            logger.debug(f"Extracted {len(images)} image(s) from doc {doc_id!r}")

    return result
