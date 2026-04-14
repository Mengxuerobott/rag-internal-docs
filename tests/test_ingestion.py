"""
tests/test_ingestion.py
───────────────────────
Unit tests for the ingestion pipeline (loader + chunker).
No external services required — tests run against the sample docs.

Run:
    pytest tests/test_ingestion.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_docs_dir(tmp_path_factory) -> Path:
    """Create a small temp directory of markdown files for testing."""
    base = tmp_path_factory.mktemp("docs")

    # HR subdirectory
    hr = base / "hr"
    hr.mkdir()
    (hr / "leave_policy.md").write_text(
        "# Leave Policy\n\nEmployees get 15 days vacation per year.\n"
        "Parental leave is 16 weeks for primary caregivers.\n"
        "Sick leave is 10 days per year.\n",
        encoding="utf-8",
    )
    (hr / "handbook.md").write_text(
        "# Employee Handbook\n\nProbationary period is 90 days.\n"
        "Performance reviews happen annually in December.\n",
        encoding="utf-8",
    )

    # Engineering subdirectory
    eng = base / "engineering"
    eng.mkdir()
    (eng / "onboarding.md").write_text(
        "# Onboarding Guide\n\nDay 1: Set up laptop and install VPN.\n"
        "Day 2: Clone the main repo and run tests.\n"
        "Week 1: Complete security training in Workday.\n",
        encoding="utf-8",
    )

    return base


@pytest.fixture(scope="module")
def loaded_documents(sample_docs_dir):
    """Load documents from the temp directory."""
    from ingestion.loader import load_documents
    return load_documents(str(sample_docs_dir))


@pytest.fixture(scope="module")
def hierarchical_nodes(loaded_documents):
    """Build hierarchical nodes from loaded documents."""
    from ingestion.chunker import build_hierarchical_nodes
    return build_hierarchical_nodes(loaded_documents, chunk_sizes=[512, 128, 64])


# ── Loader tests ──────────────────────────────────────────────────────────────
class TestLoader:
    def test_loads_expected_number_of_documents(self, loaded_documents):
        """Should load one Document per file (3 files)."""
        assert len(loaded_documents) == 3

    def test_documents_have_text(self, loaded_documents):
        """Every Document must have non-empty text."""
        for doc in loaded_documents:
            assert doc.text.strip(), f"Empty document: {doc.metadata.get('source')}"

    def test_documents_have_source_metadata(self, loaded_documents):
        """Every Document must have 'source' metadata set."""
        for doc in loaded_documents:
            assert "source" in doc.metadata, f"Missing 'source' in {doc.metadata}"
            assert doc.metadata["source"]

    def test_documents_have_department_metadata(self, loaded_documents):
        """Department should be inferred from the subdirectory name."""
        departments = {doc.metadata["department"] for doc in loaded_documents}
        assert "hr" in departments
        assert "engineering" in departments

    def test_documents_have_ingested_at_metadata(self, loaded_documents):
        """Every Document should have an ISO 8601 ingested_at timestamp."""
        for doc in loaded_documents:
            assert "ingested_at" in doc.metadata
            ts = doc.metadata["ingested_at"]
            assert "T" in ts, f"Expected ISO timestamp, got: {ts}"

    def test_raises_on_missing_directory(self):
        """Should raise FileNotFoundError for non-existent directory."""
        from ingestion.loader import load_documents
        with pytest.raises(FileNotFoundError):
            load_documents("/this/path/does/not/exist")

    def test_raises_on_empty_directory(self, tmp_path):
        """Should raise ValueError if no supported files are found."""
        from ingestion.loader import load_documents
        with pytest.raises(ValueError):
            load_documents(str(tmp_path))

    def test_single_file_loader(self, sample_docs_dir):
        """load_single_file should return at least one Document."""
        from ingestion.loader import load_single_file
        md_file = str(list(sample_docs_dir.rglob("*.md"))[0])
        docs = load_single_file(md_file)
        assert len(docs) >= 1
        assert docs[0].text.strip()


# ── Chunker tests ─────────────────────────────────────────────────────────────
class TestChunker:
    def test_returns_two_lists(self, hierarchical_nodes):
        """build_hierarchical_nodes should return (all_nodes, leaf_nodes)."""
        all_nodes, leaf_nodes = hierarchical_nodes
        assert isinstance(all_nodes, list)
        assert isinstance(leaf_nodes, list)

    def test_leaf_nodes_are_subset_of_all_nodes(self, hierarchical_nodes):
        """Every leaf node should exist in all_nodes."""
        all_nodes, leaf_nodes = hierarchical_nodes
        all_ids = {n.node_id for n in all_nodes}
        for leaf in leaf_nodes:
            assert leaf.node_id in all_ids

    def test_more_nodes_than_documents(self, loaded_documents, hierarchical_nodes):
        """Chunking should always produce more nodes than source documents."""
        all_nodes, _ = hierarchical_nodes
        assert len(all_nodes) >= len(loaded_documents)

    def test_leaf_nodes_smaller_than_parents(self, hierarchical_nodes):
        """Leaf nodes should generally be shorter than parent nodes."""
        all_nodes, leaf_nodes = hierarchical_nodes
        leaf_ids = {n.node_id for n in leaf_nodes}
        parents = [n for n in all_nodes if n.node_id not in leaf_ids]

        if parents and leaf_nodes:
            avg_leaf = sum(len(n.text) for n in leaf_nodes) / len(leaf_nodes)
            avg_parent = sum(len(n.text) for n in parents) / len(parents)
            assert avg_leaf <= avg_parent, (
                f"Expected leaf nodes (avg {avg_leaf:.0f} chars) to be shorter "
                f"than parents (avg {avg_parent:.0f} chars)"
            )

    def test_metadata_propagated_to_leaves(self, hierarchical_nodes):
        """Leaf nodes should inherit source metadata from their parent documents."""
        _, leaf_nodes = hierarchical_nodes
        for leaf in leaf_nodes:
            assert "source" in leaf.metadata, (
                f"Leaf node {leaf.node_id[:8]} missing 'source' metadata"
            )

    def test_flat_chunker(self, loaded_documents):
        """build_sentence_nodes should return a flat list of nodes."""
        from ingestion.chunker import build_sentence_nodes
        nodes = build_sentence_nodes(loaded_documents, chunk_size=256, chunk_overlap=20)
        assert len(nodes) > 0
        for node in nodes:
            assert node.text.strip()
