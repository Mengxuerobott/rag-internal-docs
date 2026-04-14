"""
ingestion/chunker.py
────────────────────
Converts raw Documents into a hierarchy of IndexNodes (parent → child → leaf).

Why hierarchical chunking?
──────────────────────────
Small chunks (128 tokens) are great for RETRIEVAL precision — the vector
similarity search finds exactly the right passage without drowning in noise.

But small chunks are terrible for GENERATION — the LLM loses context when it
only sees one sentence. The AutoMerging Retriever solves this by:
  1. Retrieving the best leaf chunks (precise match).
  2. Checking whether enough sibling leaves were retrieved from the same parent.
  3. If yes → swap the leaves out and pass the full parent chunk to the LLM.

Result: needle-sharp retrieval + rich, coherent context for the LLM.
This is LlamaIndex's AutoMerging pattern and it's a strong interview talking point.

Chunk size ladder (configurable via .env CHUNK_SIZES):
  Level 0 (root):  2048 tokens  — entire section / large topic
  Level 1 (mid):    512 tokens  — paragraph / sub-topic
  Level 2 (leaf):   128 tokens  — sentence group (what gets embedded)
"""

from typing import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.schema import BaseNode
from loguru import logger

from config import settings


def build_hierarchical_nodes(
    documents: list[Document],
    chunk_sizes: list[int] | None = None,
) -> tuple[list[BaseNode], list[BaseNode]]:
    """
    Parse documents into a hierarchy of nodes.

    Args:
        documents:   List of Document objects from the loader.
        chunk_sizes: [parent_size, child_size, leaf_size].
                     Defaults to settings.CHUNK_SIZES.

    Returns:
        all_nodes:   Every node in the hierarchy (store ALL of these in Qdrant
                     so the AutoMergingRetriever can walk up from leaf → parent).
        leaf_nodes:  Only the smallest (leaf) nodes — these are the ones we
                     embed and index for similarity search.
    """
    sizes = chunk_sizes or settings.CHUNK_SIZES

    logger.info(
        f"Building hierarchical nodes — chunk sizes: {sizes} "
        f"from {len(documents)} document(s)"
    )

    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=sizes,
        # chunk_overlap gives each chunk a small overlap with its neighbors so
        # relevant context at chunk boundaries is never completely lost.
        chunk_overlap=20,
        # include_metadata propagates document metadata down to every child node
        # so every leaf knows which file / department it came from.
        include_metadata=True,
        include_prev_next_rel=True,  # links sibling nodes for context windows
    )

    all_nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    leaf_nodes = get_leaf_nodes(all_nodes)
    root_nodes = get_root_nodes(all_nodes)

    logger.info(
        f"Created {len(all_nodes)} total nodes "
        f"({len(root_nodes)} root, {len(leaf_nodes)} leaf)"
    )

    return all_nodes, leaf_nodes


def build_sentence_nodes(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[BaseNode]:
    """
    Flat (non-hierarchical) chunking — simpler alternative when you want a
    single embedding tier.  Useful as a baseline to compare against the
    hierarchical approach in your RAGAS evaluation.

    Args:
        documents:     Source documents.
        chunk_size:    Max tokens per chunk.
        chunk_overlap: Token overlap between consecutive chunks.

    Returns:
        List of SentenceNode objects ready for embedding.
    """
    logger.info(
        f"Building flat sentence nodes — chunk_size={chunk_size}, "
        f"overlap={chunk_overlap}"
    )

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
    )

    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    logger.info(f"Created {len(nodes)} flat nodes")
    return nodes


def print_node_stats(all_nodes: list[BaseNode], leaf_nodes: list[BaseNode]) -> None:
    """Pretty-print a summary of node sizes — useful during development."""
    from rich.console import Console
    from rich.table import Table

    leaf_ids = {n.node_id for n in leaf_nodes}

    level_counts: dict[str, int] = {}
    for node in all_nodes:
        level = "leaf" if node.node_id in leaf_ids else "parent"
        level_counts[level] = level_counts.get(level, 0) + 1

    leaf_lengths = [len(n.text) for n in leaf_nodes]
    avg_leaf = sum(leaf_lengths) / max(len(leaf_lengths), 1)

    console = Console()
    table = Table(title="Node Statistics")
    table.add_column("Level")
    table.add_column("Count", justify="right")
    table.add_column("Avg chars", justify="right")

    table.add_row("parent", str(level_counts.get("parent", 0)), "—")
    table.add_row("leaf", str(level_counts.get("leaf", 0)), f"{avg_leaf:.0f}")

    console.print(table)


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingestion.loader import load_documents

    docs = load_documents()
    all_nodes, leaf_nodes = build_hierarchical_nodes(docs)
    print_node_stats(all_nodes, leaf_nodes)
