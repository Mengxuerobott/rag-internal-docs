"""
tests/test_api.py
─────────────────
Integration tests for the FastAPI endpoints.
Uses TestClient (no real server needed) and mocks the query engine
so these tests don't require Qdrant or OpenAI to be running.

Run:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Shared mock objects ───────────────────────────────────────────────────────
def _make_mock_node(text: str = "Sample context text.", source: str = "handbook.md"):
    node = MagicMock()
    node.node.text = text
    node.node.metadata = {
        "source": source,
        "department": "hr",
        "file_type": "md",
        "ingested_at": "2026-01-01T00:00:00+00:00",
    }
    node.score = 0.87
    return node


def _make_mock_response(answer: str = "Employees get 15 vacation days per year."):
    response = MagicMock()
    response.__str__ = lambda self: answer
    response.source_nodes = [
        _make_mock_node("Employees accrue 15 days of paid vacation per year.", "leave_policy.md"),
        _make_mock_node("Vacation requests must be submitted 14 days in advance.", "leave_policy.md"),
    ]
    return response


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """
    Build a TestClient with the query engine and index mocked out.
    This avoids needing Qdrant + OpenAI during CI.
    """
    mock_engine = MagicMock()
    mock_engine.query.return_value = _make_mock_response()

    with patch("api.main.get_or_build_index", return_value=MagicMock()), \
         patch("api.main.get_query_engine", return_value=mock_engine), \
         patch("retrieval.query_engine.get_query_engine", return_value=mock_engine):

        from api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Health tests ──────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_has_index_ready_field(self, client):
        r = client.get("/health")
        assert "index_ready" in r.json()


# ── Query tests ───────────────────────────────────────────────────────────────
class TestQuery:
    def test_query_returns_200(self, client):
        r = client.post("/query", json={"question": "What is the vacation policy?"})
        assert r.status_code == 200

    def test_query_response_has_required_fields(self, client):
        r = client.post("/query", json={"question": "What is the vacation policy?"})
        data = r.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data

    def test_query_answer_is_non_empty_string(self, client):
        r = client.post("/query", json={"question": "How do I take parental leave?"})
        assert isinstance(r.json()["answer"], str)
        assert len(r.json()["answer"]) > 0

    def test_query_sources_have_required_fields(self, client):
        r = client.post("/query", json={"question": "What is the sick leave policy?"})
        sources = r.json()["sources"]
        assert isinstance(sources, list)
        if sources:
            src = sources[0]
            assert "source" in src
            assert "department" in src
            assert "score" in src
            assert "text_snippet" in src

    def test_query_latency_is_positive(self, client):
        r = client.post("/query", json={"question": "Expense report process?"})
        assert r.json()["latency_ms"] > 0

    def test_query_rejects_empty_question(self, client):
        r = client.post("/query", json={"question": ""})
        assert r.status_code == 422   # Pydantic validation error

    def test_query_rejects_too_short_question(self, client):
        r = client.post("/query", json={"question": "Hi"})
        assert r.status_code == 422

    def test_query_accepts_optional_filters(self, client):
        r = client.post(
            "/query",
            json={
                "question": "What is our leave policy?",
                "filters": {"department": "hr"},
            },
        )
        assert r.status_code == 200

    def test_query_accepts_session_id(self, client):
        r = client.post(
            "/query",
            json={"question": "Vacation days?", "session_id": "test-session-123"},
        )
        assert r.status_code == 200


# ── Ingest tests ──────────────────────────────────────────────────────────────
class TestIngest:
    def test_ingest_returns_202_style_response(self, client):
        r = client.post("/ingest", json={"force_rebuild": False})
        assert r.status_code == 200
        assert "status" in r.json()

    def test_ingest_with_force_rebuild(self, client):
        r = client.post("/ingest", json={"force_rebuild": True})
        assert r.status_code == 200


# ── Input validation tests ────────────────────────────────────────────────────
class TestValidation:
    def test_question_too_long_rejected(self, client):
        r = client.post("/query", json={"question": "A" * 2001})
        assert r.status_code == 422

    def test_missing_question_rejected(self, client):
        r = client.post("/query", json={})
        assert r.status_code == 422
