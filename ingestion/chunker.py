"""
ingestion/chunker.py
────────────────────
Converts raw Documents into a hierarchy of IndexNodes (parent → child → leaf),
and merges those nodes with any pre-built multimodal nodes (table summaries,
image descriptions) so all content flows into a single embedding pass.

Multimodal integration (Feature 3)
────────────────────────────────────
Multimodal nodes (content_type in ["table", "image"]) are self-contained:
  - The VLM already produced a semantically dense, embedding-ready text.
  - They must NOT be re-chunked — splitting a table summary mid-sentence
    destroys its meaning.

build_all_nodes() therefore:
  1. Runs build_hierarchical_nodes() on the cleaned text documents.
  2. Treats multimodal nodes as leaf-level nodes and appends them directly
     to both all_nodes and leaf_nodes.
  3. Returns the merged set.

The downstream embedder embeds all leaf_nodes in a single pass, so Qdrant
receives one collection containing text chunks, table summaries, and image
descriptions — all retrievable via the same hybrid search.

Chunk size ladder (configurable via .env CHUNK_SIZES):
  Level 0 (root):  2048 tokens  — entire section / large topic
  Level 1 (mid):    512 tokens  — paragraph / sub-topic
  Level 2 (leaf):   128 tokens  — sentence group (what gets embedded)
"""

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.schema import BaseNode, TextNode
from loguru import logger

from config import settings


def build_hierarchical_nodes(
    documents: list[Document],
    chunk_sizes: list[int] | None = None,
) -> tuple[list[BaseNode], list[BaseNode]]:
    """
    Parse documents into a hierarchy of nodes.

    Args:
        documents:   List of Document objects from the loader (tables already stripped).
        chunk_sizes: [parent_size, child_size, leaf_size].
                     Defaults to settings.CHUNK_SIZES.

    Returns:
        all_nodes:   Every node in the hierarchy (store ALL in Qdrant so
                     AutoMergingRetriever can walk up from leaf → parent).
        leaf_nodes:  Only the smallest (leaf) nodes — what gets embedded.
    """
    sizes = chunk_sizes or settings.CHUNK_SIZES

    logger.info(
        f"Building hierarchical nodes — chunk sizes: {sizes} "
        f"from {len(documents)} document(s)"
    )

    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=sizes,
        chunk_overlap=20,
        include_metadata=True,
        include_prev_next_rel=True,
    )

    all_nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    leaf_nodes = get_leaf_nodes(all_nodes)
    root_nodes = get_root_nodes(all_nodes)

    logger.info(
        f"Created {len(all_nodes)} total nodes "
        f"({len(root_nodes)} root, {len(leaf_nodes)} leaf)"
    )

    return all_nodes, leaf_nodes


def build_all_nodes(
    documents: list[Document],
    multimodal_nodes: list[TextNode] | None = None,
    chunk_sizes: list[int] | None = None,
) -> tuple[list[BaseNode], list[BaseNode]]:
    """
    Build hierarchical text nodes and merge with pre-built multimodal nodes.

    This is the main entry point called by embedder.py. It replaces direct
    calls to build_hierarchical_nodes() when multimodal processing is enabled.

    Args:
        documents:        Cleaned Document objects (tables stripped by loader).
        multimodal_nodes: Pre-built TextNodes from ingestion/multimodal.py.
                          These are already embedding-ready — not re-chunked.
                          Includes table summary nodes and image description nodes.
        chunk_sizes:      Passed through to build_hierarchical_nodes().

    Returns:
        all_nodes:  Text hierarchy nodes + multimodal nodes (for docstore).
        leaf_nodes: Text leaf nodes + multimodal nodes (for embedding).

    Why multimodal nodes count as leaves:
        A table summary is a complete, self-contained semantic unit.
        An image description is likewise atomic.
        There is no meaningful "parent" for either — they ARE the retrievable unit.
        The AutoMerging Retriever ignores them (no parent relationship) which
        is exactly the desired behaviour: they always go to the LLM as-is.
    """
    # Step 1: hierarchical chunking on text documents
    all_text_nodes, leaf_text_nodes = build_hierarchical_nodes(documents, chunk_sizes)

    if not multimodal_nodes:
        return all_text_nodes, leaf_text_nodes

    # Step 2: merge multimodal nodes as leaves
    n_tables = sum(1 for n in multimodal_nodes if n.metadata.get("content_type") == "table")
    n_images = sum(1 for n in multimodal_nodes if n.metadata.get("content_type") == "image")
    logger.info(
        f"Merging {len(multimodal_nodes)} multimodal node(s) "
        f"({n_tables} table, {n_images} image) into node set"
    )

    all_nodes  = all_text_nodes  + list(multimodal_nodes)
    leaf_nodes = leaf_text_nodes + list(multimodal_nodes)

    logger.info(
        f"Final node counts: {len(all_nodes)} total, {len(leaf_nodes)} leaf "
        f"({len(leaf_text_nodes)} text + {len(multimodal_nodes)} multimodal)"
    )

    return all_nodes, leaf_nodes


def build_sentence_nodes(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[BaseNode]:
    """
    Flat (non-hierarchical) chunking — used as a RAGAS evaluation baseline.
    Useful for demonstrating that hierarchical chunking outperforms flat chunking.
    """
    logger.info(f"Building flat sentence nodes — chunk_size={chunk_size}, overlap={chunk_overlap}")

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
    )

    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    logger.info(f"Created {len(nodes)} flat nodes")
    return nodes


def print_node_stats(
    all_nodes: list[BaseNode],
    leaf_nodes: list[BaseNode],
) -> None:
    """Pretty-print node statistics including multimodal breakdown."""
    from rich.console import Console
    from rich.table import Table

    leaf_ids = {n.node_id for n in leaf_nodes}

    # Count by type
    counts: dict[str, int] = {"parent": 0, "leaf_text": 0, "table": 0, "image": 0}
    for node in all_nodes:
        ctype = node.metadata.get("content_type", "")
        if node.node_id not in leaf_ids:
            counts["parent"] += 1
        elif ctype == "table":
            counts["table"] += 1
        elif ctype == "image":
            counts["image"] += 1
        else:
            counts["leaf_text"] += 1

    text_leaves = [n for n in leaf_nodes if not n.metadata.get("content_type")]
    avg_chars = (
        sum(len(n.text) for n in text_leaves) / max(len(text_leaves), 1)
    )

    console = Console()
    tbl = Table(title="Node Statistics")
    tbl.add_column("Type")
    tbl.add_column("Count", justify="right")
    tbl.add_column("Avg chars", justify="right")

    tbl.add_row("parent (text)",    str(counts["parent"]),     "—")
    tbl.add_row("leaf (text)",      str(counts["leaf_text"]),  f"{avg_chars:.0f}")
    tbl.add_row("leaf (table)",     str(counts["table"]),      "—")
    tbl.add_row("leaf (image)",     str(counts["image"]),      "—")

    console.print(tbl)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingestion.loader import load_documents

    docs, mm_nodes = load_documents()
    all_nodes, leaf_nodes = build_all_nodes(docs, mm_nodes)
    print_node_stats(all_nodes, leaf_nodes)
