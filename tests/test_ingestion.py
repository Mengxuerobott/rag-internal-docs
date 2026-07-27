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
from unittest.mock import MagicMock
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
    """Load documents from the temp directory.

    load_documents returns (documents, multimodal_nodes). Only the documents
    are of interest here; multimodal extraction is covered by test_multimodal.
    Returning the raw tuple made every consumer below see a 2-element tuple
    instead of a document list.
    """
    from ingestion.loader import load_documents
    documents, _multimodal_nodes = load_documents(str(sample_docs_dir))
    return documents


@pytest.fixture(scope="module")
def hierarchical_nodes(loaded_documents):
    """Build hierarchical nodes from loaded documents."""
    from ingestion.chunker import build_hierarchical_nodes
    # Smaller than the [2048, 512, 128] production default so the test corpus
    # still produces a multi-level hierarchy, but the leaf size must stay above
    # the per-node metadata length (~76 chars here) or LlamaIndex raises
    # "Metadata length is longer than chunk size". 64 was below it.
    return build_hierarchical_nodes(loaded_documents, chunk_sizes=[512, 256, 128])


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
        # Returns the same (documents, multimodal_nodes) tuple as load_documents.
        docs, _multimodal_nodes = load_single_file(md_file)
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


# ── CHUNK_SIZES validation ────────────────────────────────────────────────────
class TestChunkSizesValidation:
    """
    CHUNK_SIZES is user-editable in .env and feeds the hierarchical parser.
    A bad value used to surface as an opaque parser error partway through an
    ingestion run; it is now rejected at import with an actionable message.
    """

    def test_default_parses(self):
        from config import _parse_chunk_sizes
        assert _parse_chunk_sizes("2048,512,128") == [2048, 512, 128]

    def test_whitespace_tolerated(self):
        from config import _parse_chunk_sizes
        assert _parse_chunk_sizes(" 1024 , 256 ") == [1024, 256]

    def test_non_integer_rejected(self):
        from config import _parse_chunk_sizes
        with pytest.raises(ValueError, match="comma-separated integers"):
            _parse_chunk_sizes("2048,abc")

    def test_single_value_rejected(self):
        from config import _parse_chunk_sizes
        with pytest.raises(ValueError, match="at least two values"):
            _parse_chunk_sizes("2048")

    def test_ascending_order_rejected(self):
        from config import _parse_chunk_sizes
        with pytest.raises(ValueError, match="strictly descending"):
            _parse_chunk_sizes("128,512,2048")

    def test_duplicate_sizes_rejected(self):
        from config import _parse_chunk_sizes
        with pytest.raises(ValueError, match="must not repeat"):
            _parse_chunk_sizes("512,512,128")

    def test_non_positive_rejected(self):
        from config import _parse_chunk_sizes
        with pytest.raises(ValueError, match="must all be positive"):
            _parse_chunk_sizes("2048,-5")

    def test_leaf_below_metadata_length_rejected(self):
        """The failure this validation exists to prevent."""
        from config import _parse_chunk_sizes, MIN_LEAF_CHUNK_SIZE
        with pytest.raises(ValueError, match="below the minimum"):
            _parse_chunk_sizes(f"2048,512,{MIN_LEAF_CHUNK_SIZE - 1}")


# ── Durable index storage ─────────────────────────────────────────────────────
class TestIndexSync:
    """
    The docstore holds the parent nodes AutoMerging walks to; Qdrant holds only
    the embedded leaves. On Fargate the persist directory dies with the task, so
    without this the whole corpus is re-embedded on every deploy purely to
    rebuild it.
    """

    def _patch_settings(self, monkeypatch, bucket="", prefix="index_store"):
        from config import settings
        monkeypatch.setattr(settings, "INDEX_S3_BUCKET", bucket)
        monkeypatch.setattr(settings, "INDEX_S3_PREFIX", prefix)

    # ── Disabled by default ───────────────────────────────────────────────
    def test_noop_when_bucket_unset(self, monkeypatch):
        from ingestion import index_sync
        self._patch_settings(monkeypatch, bucket="")
        assert index_sync.s3_configured() is False
        assert index_sync.download_index_store() is False
        assert index_sync.upload_index_store() is False

    def test_no_s3_client_created_when_unset(self, monkeypatch):
        """Must not touch boto3 at all on the default path."""
        from ingestion import index_sync
        self._patch_settings(monkeypatch, bucket="")
        called = []
        monkeypatch.setattr(index_sync, "_client", lambda: called.append(1))
        index_sync.download_index_store()
        index_sync.upload_index_store()
        assert called == []

    # ── Download ──────────────────────────────────────────────────────────
    def test_download_restores_files_preserving_layout(self, monkeypatch, tmp_path):
        from ingestion import index_sync
        self._patch_settings(monkeypatch, bucket="my-bucket")

        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [
            {"Contents": [
                {"Key": "index_store/docstore.json"},
                {"Key": "index_store/nested/graph_store.json"},
                {"Key": "index_store/"},          # directory marker, must be skipped
            ]}
        ]
        monkeypatch.setattr(index_sync, "_client", lambda: client)

        assert index_sync.download_index_store(str(tmp_path)) is True

        downloaded = [c.args[2] for c in client.download_file.call_args_list]
        assert len(downloaded) == 2, "directory marker should have been skipped"
        assert any(p.endswith("docstore.json") for p in downloaded)
        assert any("nested" in p for p in downloaded)

    def test_download_returns_false_when_bucket_empty(self, monkeypatch, tmp_path):
        """An empty bucket means build-and-upload, not an error."""
        from ingestion import index_sync
        self._patch_settings(monkeypatch, bucket="my-bucket")
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [{}]
        monkeypatch.setattr(index_sync, "_client", lambda: client)
        assert index_sync.download_index_store(str(tmp_path)) is False

    def test_download_failure_degrades_to_rebuild(self, monkeypatch, tmp_path):
        """S3 being unreachable must not take the API down."""
        from ingestion import index_sync
        self._patch_settings(monkeypatch, bucket="my-bucket")
        client = MagicMock()
        client.get_paginator.side_effect = RuntimeError("network is down")
        monkeypatch.setattr(index_sync, "_client", lambda: client)
        assert index_sync.download_index_store(str(tmp_path)) is False

    # ── Upload ────────────────────────────────────────────────────────────
    def test_upload_mirrors_every_file(self, monkeypatch, tmp_path):
        from ingestion import index_sync
        self._patch_settings(monkeypatch, bucket="my-bucket")

        (tmp_path / "docstore.json").write_text("{}", encoding="utf-8")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "index_store.json").write_text("{}", encoding="utf-8")

        client = MagicMock()
        monkeypatch.setattr(index_sync, "_client", lambda: client)

        assert index_sync.upload_index_store(str(tmp_path)) is True

        keys = sorted(c.args[2] for c in client.upload_file.call_args_list)
        assert keys == ["index_store/docstore.json", "index_store/nested/index_store.json"]

    def test_upload_failure_is_swallowed(self, monkeypatch, tmp_path):
        """The index is still valid in memory; only the next restart pays."""
        from ingestion import index_sync
        self._patch_settings(monkeypatch, bucket="my-bucket")
        (tmp_path / "docstore.json").write_text("{}", encoding="utf-8")
        client = MagicMock()
        client.upload_file.side_effect = RuntimeError("access denied")
        monkeypatch.setattr(index_sync, "_client", lambda: client)
        assert index_sync.upload_index_store(str(tmp_path)) is False

    def test_upload_missing_directory_is_not_an_error(self, monkeypatch, tmp_path):
        from ingestion import index_sync
        self._patch_settings(monkeypatch, bucket="my-bucket")
        assert index_sync.upload_index_store(str(tmp_path / "nope")) is False

    # ── Local presence check ──────────────────────────────────────────────
    def test_local_present_false_for_empty_dir(self, tmp_path):
        from ingestion.index_sync import local_index_store_present
        assert local_index_store_present(str(tmp_path)) is False

    def test_local_present_true_when_populated(self, tmp_path):
        from ingestion.index_sync import local_index_store_present
        (tmp_path / "docstore.json").write_text("{}", encoding="utf-8")
        assert local_index_store_present(str(tmp_path)) is True
