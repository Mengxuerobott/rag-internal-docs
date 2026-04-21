"""
tests/test_multimodal.py
─────────────────────────
Tests for the multimodal ingestion pipeline:
  - Markdown table detection and stripping
  - Table summarisation (VLM call mocked)
  - Image description (VLM call mocked)
  - Full document processing orchestrator
  - Loader integration (returns tuple, passes multimodal nodes through)
  - Chunker merge (multimodal nodes land in leaf set, not re-chunked)

No real OpenAI API calls are made — all VLM calls are mocked.

Run:
    pytest tests/test_multimodal.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ───────────────────────────────────────────────────────────────────

SIMPLE_TABLE_MD = """\
| Department | Budget | Actual | Variance |
|-----------|--------|--------|----------|
| Engineering | $500K | $480K | -$20K |
| HR | $200K | $210K | +$10K |
| Marketing | $300K | $290K | -$10K |
"""

TABLE_IN_TEXT = f"""
# Q1 Financial Report

The following table summarises departmental spending for Q1 2026.

{SIMPLE_TABLE_MD}

Overall the company came in under budget by $20K.
"""

MULTI_TABLE_TEXT = f"""
# Report

## Section 1
{SIMPLE_TABLE_MD}

## Section 2
| Name | Role | Start Date |
|------|------|-----------|
| Alice | Engineer | 2024-01-15 |
| Bob | Manager | 2023-06-01 |
"""


def _make_document(text: str, source: str = "test.pdf", dept: str = "finance") -> Document:
    doc = Document(text=text)
    doc.metadata = {
        "source": source,
        "department": dept,
        "allowed_roles": ["finance", "management", "admin"],
        "doc_id": "doc-test-001",
        "ingested_at": "2026-01-01T00:00:00+00:00",
    }
    return doc


def _mock_openai_response(text: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Table extraction tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractMarkdownTables:
    def test_finds_single_table(self):
        from ingestion.multimodal import extract_markdown_tables
        tables = extract_markdown_tables(TABLE_IN_TEXT)
        assert len(tables) == 1

    def test_finds_multiple_tables(self):
        from ingestion.multimodal import extract_markdown_tables
        tables = extract_markdown_tables(MULTI_TABLE_TEXT)
        assert len(tables) == 2

    def test_returns_empty_for_no_tables(self):
        from ingestion.multimodal import extract_markdown_tables
        tables = extract_markdown_tables("No tables here. Just plain text.")
        assert tables == []

    def test_table_markdown_contains_pipe_chars(self):
        from ingestion.multimodal import extract_markdown_tables
        tables = extract_markdown_tables(TABLE_IN_TEXT)
        assert "|" in tables[0].markdown

    def test_context_contains_preceding_paragraph(self):
        from ingestion.multimodal import extract_markdown_tables
        tables = extract_markdown_tables(TABLE_IN_TEXT, context_window=500)
        assert "departmental spending" in tables[0].context

    def test_table_without_data_rows_not_matched(self):
        """A header + separator with no data rows is not a valid table."""
        from ingestion.multimodal import extract_markdown_tables
        text = "| A | B |\n|---|---|\n"
        # This has only header + separator but zero data rows — should not match
        tables = extract_markdown_tables(text)
        assert len(tables) == 0


class TestStripTablesFromText:
    def test_removes_table_from_text(self):
        from ingestion.multimodal import strip_tables_from_text
        result = strip_tables_from_text(TABLE_IN_TEXT)
        assert "| Department |" not in result

    def test_inserts_placeholder(self):
        from ingestion.multimodal import strip_tables_from_text
        result = strip_tables_from_text(TABLE_IN_TEXT)
        assert "[TABLE:" in result

    def test_surrounding_text_preserved(self):
        from ingestion.multimodal import strip_tables_from_text
        result = strip_tables_from_text(TABLE_IN_TEXT)
        assert "Q1 Financial Report" in result
        assert "came in under budget" in result

    def test_multiple_tables_all_stripped(self):
        from ingestion.multimodal import strip_tables_from_text
        result = strip_tables_from_text(MULTI_TABLE_TEXT)
        assert result.count("[TABLE:") == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Table summarisation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummarizeTableWithVlm:
    def test_returns_text_node(self):
        from ingestion.multimodal import ExtractedTable, summarize_table_with_vlm

        table = ExtractedTable(
            markdown=SIMPLE_TABLE_MD,
            context="Q1 departmental budget report",
            page_num=0,
            char_offset=100,
        )

        mock_resp = _mock_openai_response(
            "The table shows Q1 departmental budgets. Engineering spent $480K "
            "against a $500K budget. HR slightly overspent at $210K vs $200K."
        )

        with patch("ingestion.multimodal._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            node = summarize_table_with_vlm(table, source_file="budget.pdf")

        assert isinstance(node, TextNode)
        assert len(node.text) > 0

    def test_node_content_type_is_table(self):
        from ingestion.multimodal import ExtractedTable, summarize_table_with_vlm

        table = ExtractedTable(SIMPLE_TABLE_MD, "context", 0, 0)
        mock_resp = _mock_openai_response("Department budget summary.")

        with patch("ingestion.multimodal._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            node = summarize_table_with_vlm(table)

        assert node.metadata["content_type"] == "table"

    def test_original_table_stored_in_metadata(self):
        from ingestion.multimodal import ExtractedTable, summarize_table_with_vlm

        table = ExtractedTable(SIMPLE_TABLE_MD, "", 0, 0)
        mock_resp = _mock_openai_response("Summary text.")

        with patch("ingestion.multimodal._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            node = summarize_table_with_vlm(table)

        assert node.metadata["original_table"] == SIMPLE_TABLE_MD

    def test_doc_metadata_propagated_to_node(self):
        from ingestion.multimodal import ExtractedTable, summarize_table_with_vlm

        table = ExtractedTable(SIMPLE_TABLE_MD, "", 0, 0)
        doc_meta = {
            "doc_id": "doc-xyz",
            "allowed_roles": ["finance", "admin"],
            "department": "finance",
        }
        mock_resp = _mock_openai_response("Summary.")

        with patch("ingestion.multimodal._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            node = summarize_table_with_vlm(table, doc_metadata=doc_meta)

        assert node.metadata["doc_id"] == "doc-xyz"
        assert node.metadata["allowed_roles"] == ["finance", "admin"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Image description tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDescribeImageWithVlm:
    def _make_image(self, size_bytes: int = 1000) -> "ExtractedImage":
        from ingestion.multimodal import ExtractedImage
        import base64
        fake_b64 = base64.b64encode(b"x" * size_bytes).decode()
        return ExtractedImage(
            b64_data=fake_b64,
            mime_type="image/png",
            context="Figure 1: Q1 Revenue by Region",
            page_num=2,
        )

    def test_returns_text_node_for_valid_image(self):
        from ingestion.multimodal import describe_image_with_vlm

        img = self._make_image(1000)
        mock_resp = _mock_openai_response(
            "A bar chart showing Q1 revenue by region. North America leads at $1.2M."
        )

        with patch("ingestion.multimodal._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            node = describe_image_with_vlm(img, source_file="report.pdf")

        assert isinstance(node, TextNode)
        assert node.metadata["content_type"] == "image"

    def test_oversized_image_returns_none(self):
        from ingestion.multimodal import describe_image_with_vlm, ExtractedImage
        import base64

        # 10MB image — above the 4MB default limit
        big_b64 = base64.b64encode(b"x" * (10 * 1024 * 1024)).decode()
        img = ExtractedImage(big_b64, "image/png", "", 0)

        with patch("config.settings.MULTIMODAL_MAX_IMAGE_MB", 4.0):
            result = describe_image_with_vlm(img)

        assert result is None

    def test_image_b64_stored_in_metadata(self):
        from ingestion.multimodal import describe_image_with_vlm

        img = self._make_image(500)
        mock_resp = _mock_openai_response("A pie chart.")

        with patch("ingestion.multimodal._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            node = describe_image_with_vlm(img)

        assert "image_b64" in node.metadata
        assert node.metadata["mime_type"] == "image/png"

    def test_failed_api_call_returns_none(self):
        from ingestion.multimodal import describe_image_with_vlm

        img = self._make_image(500)

        with patch("ingestion.multimodal._get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = RuntimeError("API down")
            result = describe_image_with_vlm(img)

        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Document orchestrator tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcessDocumentMultimodal:
    def test_returns_tuple(self):
        from ingestion.multimodal import process_document_multimodal

        doc = _make_document(TABLE_IN_TEXT)
        mock_resp = _mock_openai_response("Budget summary.")

        with patch("ingestion.multimodal._get_openai_client") as mock_client, \
             patch("config.settings.ENABLE_MULTIMODAL", True):
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            result = process_document_multimodal(doc)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cleaned_doc_has_no_raw_table(self):
        from ingestion.multimodal import process_document_multimodal

        doc = _make_document(TABLE_IN_TEXT)
        mock_resp = _mock_openai_response("Summary.")

        with patch("ingestion.multimodal._get_openai_client") as mock_client, \
             patch("config.settings.ENABLE_MULTIMODAL", True):
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            cleaned_doc, _ = process_document_multimodal(doc)

        assert "| Department |" not in cleaned_doc.text

    def test_extra_nodes_contain_table_node(self):
        from ingestion.multimodal import process_document_multimodal

        doc = _make_document(TABLE_IN_TEXT)
        mock_resp = _mock_openai_response("Table about Q1 departmental budgets.")

        with patch("ingestion.multimodal._get_openai_client") as mock_client, \
             patch("config.settings.ENABLE_MULTIMODAL", True):
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            _, extra_nodes = process_document_multimodal(doc)

        assert len(extra_nodes) == 1
        assert extra_nodes[0].metadata["content_type"] == "table"

    def test_two_tables_produce_two_nodes(self):
        from ingestion.multimodal import process_document_multimodal

        doc = _make_document(MULTI_TABLE_TEXT)
        mock_resp = _mock_openai_response("Summary.")

        with patch("ingestion.multimodal._get_openai_client") as mock_client, \
             patch("config.settings.ENABLE_MULTIMODAL", True):
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            _, extra_nodes = process_document_multimodal(doc)

        table_nodes = [n for n in extra_nodes if n.metadata["content_type"] == "table"]
        assert len(table_nodes) == 2

    def test_disabled_multimodal_returns_unchanged_doc(self):
        from ingestion.multimodal import process_document_multimodal

        doc = _make_document(TABLE_IN_TEXT)
        original_text = doc.text

        with patch("config.settings.ENABLE_MULTIMODAL", False):
            cleaned_doc, extra_nodes = process_document_multimodal(doc)

        # Doc text unchanged (table NOT stripped)
        assert cleaned_doc.text == original_text
        # No extra nodes generated
        assert extra_nodes == []


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Chunker integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildAllNodes:
    def _make_text_doc(self) -> Document:
        doc = Document(text="This is a plain text document with multiple sentences. " * 20)
        doc.metadata = {"source": "doc.txt", "department": "general",
                        "allowed_roles": ["employee"]}
        return doc

    def _make_mm_node(self, ctype: str = "table") -> TextNode:
        node = TextNode(text=f"A {ctype} description node.")
        node.metadata = {"content_type": ctype, "source": "doc.pdf",
                         "doc_id": "doc-123", "allowed_roles": ["finance", "admin"]}
        return node

    def test_without_multimodal_nodes_same_as_hierarchical(self):
        from ingestion.chunker import build_all_nodes, build_hierarchical_nodes

        doc = self._make_text_doc()
        all1, leaf1 = build_hierarchical_nodes([doc])
        all2, leaf2 = build_all_nodes([doc], multimodal_nodes=None)

        assert len(all1) == len(all2)
        assert len(leaf1) == len(leaf2)

    def test_multimodal_nodes_appear_in_leaf_set(self):
        from ingestion.chunker import build_all_nodes

        doc = self._make_text_doc()
        mm_nodes = [self._make_mm_node("table"), self._make_mm_node("image")]
        _, leaf_nodes = build_all_nodes([doc], multimodal_nodes=mm_nodes)

        leaf_ids = {n.node_id for n in leaf_nodes}
        for mm in mm_nodes:
            assert mm.node_id in leaf_ids, "Multimodal node not found in leaf set"

    def test_multimodal_nodes_appear_in_all_nodes(self):
        from ingestion.chunker import build_all_nodes

        doc = self._make_text_doc()
        mm_nodes = [self._make_mm_node()]
        all_nodes, _ = build_all_nodes([doc], multimodal_nodes=mm_nodes)

        all_ids = {n.node_id for n in all_nodes}
        assert mm_nodes[0].node_id in all_ids

    def test_total_leaf_count_equals_text_leaves_plus_multimodal(self):
        from ingestion.chunker import build_all_nodes, build_hierarchical_nodes

        doc = self._make_text_doc()
        mm_nodes = [self._make_mm_node("table"), self._make_mm_node("image")]

        _, text_leaf_nodes = build_hierarchical_nodes([doc])
        _, all_leaf_nodes = build_all_nodes([doc], multimodal_nodes=mm_nodes)

        assert len(all_leaf_nodes) == len(text_leaf_nodes) + len(mm_nodes)

    def test_multimodal_node_text_unchanged(self):
        """Multimodal nodes must NOT be re-chunked or modified."""
        from ingestion.chunker import build_all_nodes

        doc = self._make_text_doc()
        mm = self._make_mm_node("table")
        original_text = mm.text

        _, leaf_nodes = build_all_nodes([doc], multimodal_nodes=[mm])

        # Find the multimodal node by ID in the leaf set
        found = next((n for n in leaf_nodes if n.node_id == mm.node_id), None)
        assert found is not None
        assert found.text == original_text, "Multimodal node text was modified!"
