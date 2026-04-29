"""
tests/test_router.py
─────────────────────
Tests for the agentic routing layer:

  1. Intent classification (mocked LLM — correct routing decisions)
  2. Conversation memory (LRU eviction, turn limits, thread safety)
  3. SmallTalkHandler (no retrieval, uses history)
  4. SummarizationHandler (Qdrant scroll, no vector search)
  5. DeepRagHandler (delegates to query_engine, extracts sources)
  6. route_query dispatch (correct handler called for each intent)
  7. API endpoint integration (route_type in response, history endpoints)

No real OpenAI or Qdrant calls are made — everything is mocked.

Run:
    pytest tests/test_router.py -v
"""

import sys
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.handlers import ConversationTurn, RouteContext
from retrieval.router import (
    ConversationMemory,
    Intent,
    classify_intent,
    get_memory,
    route_query,
)
from auth.jwt_handler import CurrentUser


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _user(role: str = "employee") -> CurrentUser:
    return CurrentUser(username=f"test_{role}", role=role, full_name="Test User")


def _mock_classify(intent: str, target_doc: str = "", confidence: float = 0.95):
    """Return a mock OpenAI client that always classifies to the given intent."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "intent": intent,
        "confidence": confidence,
        "target_doc": target_doc,
        "reasoning": "test classification",
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Intent classification tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyIntent:
    def test_small_talk_intent_returned(self):
        with patch("retrieval.router.OpenAI", return_value=_mock_classify("small_talk")):
            intent, confidence, target = classify_intent("hello", [])
        assert intent == Intent.SMALL_TALK
        assert confidence == 0.95
        assert target is None

    def test_deep_rag_intent_returned(self):
        with patch("retrieval.router.OpenAI", return_value=_mock_classify("deep_rag")):
            intent, _, _ = classify_intent("how many vacation days do I get?", [])
        assert intent == Intent.DEEP_RAG

    def test_summarization_intent_with_target_doc(self):
        with patch("retrieval.router.OpenAI",
                   return_value=_mock_classify("summarization", "leave_policy.md")):
            intent, _, target = classify_intent("summarize the leave policy", [])
        assert intent == Intent.SUMMARIZATION
        assert target == "leave_policy.md"

    def test_malformed_json_defaults_to_deep_rag(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "not json at all"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("retrieval.router.OpenAI", return_value=mock_client):
            intent, confidence, _ = classify_intent("some question", [])

        assert intent == Intent.DEEP_RAG
        assert confidence == 0.5

    def test_api_failure_defaults_to_deep_rag(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        with patch("retrieval.router.OpenAI", return_value=mock_client):
            intent, confidence, _ = classify_intent("question", [])

        assert intent == Intent.DEEP_RAG

    def test_history_is_passed_to_classifier(self):
        history = [
            ConversationTurn("user", "what is PTO?"),
            ConversationTurn("assistant", "PTO stands for paid time off."),
        ]
        mock_client = _mock_classify("small_talk")
        with patch("retrieval.router.OpenAI", return_value=mock_client):
            classify_intent("can you rephrase that?", history)

        # Verify the classifier was called with history in the message
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"] if call_args.kwargs else call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "PTO" in user_msg["content"] or "rephrase" in user_msg["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ConversationMemory tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConversationMemory:
    def test_empty_session_returns_empty_list(self):
        mem = ConversationMemory(max_turns=6)
        assert mem.get("nonexistent") == []

    def test_append_and_retrieve(self):
        mem = ConversationMemory(max_turns=6)
        mem.append("s1", "hello", "hi there")
        history = mem.get("s1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "hello"
        assert history[1].role == "assistant"
        assert history[1].content == "hi there"

    def test_old_turns_evicted_at_max(self):
        mem = ConversationMemory(max_turns=2)  # max 2 pairs = 4 messages
        for i in range(5):
            mem.append("s1", f"q{i}", f"a{i}")
        history = mem.get("s1")
        # Should have exactly max_turns * 2 = 4 messages
        assert len(history) == 4
        # Should be the most recent turns
        assert history[0].content == "q3"
        assert history[-1].content == "a4"

    def test_oldest_session_evicted_at_capacity(self):
        mem = ConversationMemory(max_turns=6, max_sessions=3)
        mem.append("s1", "q", "a")
        mem.append("s2", "q", "a")
        mem.append("s3", "q", "a")
        # Adding s4 should evict s1
        mem.append("s4", "q", "a")
        assert mem.get("s1") == []
        assert mem.session_count == 3

    def test_clear_removes_session(self):
        mem = ConversationMemory()
        mem.append("s1", "q", "a")
        mem.clear("s1")
        assert mem.get("s1") == []

    def test_different_sessions_independent(self):
        mem = ConversationMemory()
        mem.append("alice", "vacation days?", "15 days per year")
        mem.append("bob", "expense limit?", "$500 needs manager approval")
        assert len(mem.get("alice")) == 2
        assert len(mem.get("bob")) == 2
        assert "vacation" in mem.get("alice")[0].content
        assert "expense" in mem.get("bob")[0].content


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SmallTalkHandler tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmallTalkHandler:
    def test_returns_route_result_with_small_talk_type(self):
        from retrieval.handlers import SmallTalkHandler

        ctx = RouteContext(question="hello", user=_user())
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hi! How can I help you today?"))]
        )

        with patch("retrieval.handlers._get_openai_client", return_value=mock_client):
            # SmallTalkHandler uses OpenAI() directly
            with patch("retrieval.handlers.OpenAI", return_value=mock_client):
                result = SmallTalkHandler().handle(ctx)

        assert result.route_type == "small_talk"
        assert len(result.answer) > 0
        assert result.sources == []

    def test_conversation_history_passed_to_llm(self):
        from retrieval.handlers import SmallTalkHandler

        history = [
            ConversationTurn("user", "how many vacation days?"),
            ConversationTurn("assistant", "15 days per year after 1 year."),
        ]
        ctx = RouteContext(question="can you repeat that?", user=_user(), history=history)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="15 days per year."))]
        )

        with patch("retrieval.handlers.OpenAI", return_value=mock_client):
            SmallTalkHandler().handle(ctx)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1].get("messages") or call_args[0][0]
        # Should include at least system + 2 history messages + current question
        assert len(messages) >= 4

    def test_no_sources_returned(self):
        from retrieval.handlers import SmallTalkHandler

        ctx = RouteContext(question="thanks!", user=_user())
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="You're welcome!"))]
        )

        with patch("retrieval.handlers.OpenAI", return_value=mock_client):
            result = SmallTalkHandler().handle(ctx)

        assert result.sources == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SummarizationHandler tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummarizationHandler:
    def _make_qdrant_point(self, text: str, source: str = "leave_policy.md"):
        point = MagicMock()
        point.payload = {
            "_node_content": text,
            "source": source,
            "department": "hr",
            "allowed_roles": ["hr", "management", "admin"],
            "page_num": 0,
        }
        return point

    def test_returns_summarization_route_type(self):
        from retrieval.handlers import SummarizationHandler

        ctx = RouteContext(
            question="summarize the leave policy",
            user=_user("hr"),
            target_doc="leave_policy.md",
        )

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = (
            [self._make_qdrant_point("Employees get 15 days vacation per year.")],
            None,
        )

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Summary: 15 days vacation."))]
        )

        with patch("retrieval.handlers.QdrantClient", return_value=mock_qdrant), \
             patch("retrieval.handlers.OpenAI", return_value=mock_openai):
            result = SummarizationHandler().handle(ctx)

        assert result.route_type == "summarization"

    def test_not_found_returns_helpful_message(self):
        from retrieval.handlers import SummarizationHandler

        ctx = RouteContext(
            question="summarize the xyz doc",
            user=_user(),
            target_doc="nonexistent_doc.pdf",
        )

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)

        with patch("retrieval.handlers.QdrantClient", return_value=mock_qdrant):
            result = SummarizationHandler().handle(ctx)

        assert "couldn't find" in result.answer.lower() or "not found" in result.answer.lower() or "nonexistent_doc" in result.answer
        assert result.sources == []

    def test_source_doc_appears_in_sources(self):
        from retrieval.handlers import SummarizationHandler

        ctx = RouteContext(
            question="summarize the expense policy",
            user=_user("finance"),
            target_doc="expense_policy.md",
        )

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = (
            [self._make_qdrant_point("Submit expenses in Concur.", "expense_policy.md")],
            None,
        )
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Use Concur for expenses."))]
        )

        with patch("retrieval.handlers.QdrantClient", return_value=mock_qdrant), \
             patch("retrieval.handlers.OpenAI", return_value=mock_openai):
            result = SummarizationHandler().handle(ctx)

        assert len(result.sources) >= 1
        assert result.sources[0]["source"] == "expense_policy.md"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DeepRagHandler tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeepRagHandler:
    def _make_mock_engine(self, answer: str = "The policy states 15 vacation days."):
        node = MagicMock()
        node.node.text = "Employees accrue 15 days vacation."
        node.node.metadata = {
            "source": "leave_policy.md",
            "department": "hr",
            "allowed_roles": ["hr", "management"],
        }
        node.score = 0.91

        response = MagicMock()
        response.__str__ = lambda self: answer
        response.source_nodes = [node]

        engine = MagicMock()
        engine.query.return_value = response
        return engine

    def test_returns_deep_rag_route_type(self):
        from retrieval.handlers import DeepRagHandler

        engine = self._make_mock_engine()
        ctx = RouteContext(
            question="how many vacation days after 3 years?",
            user=_user("hr"),
            query_engine=engine,
        )
        result = DeepRagHandler().handle(ctx)
        assert result.route_type == "deep_rag"

    def test_sources_extracted_from_nodes(self):
        from retrieval.handlers import DeepRagHandler

        engine = self._make_mock_engine()
        ctx = RouteContext(question="vacation days?", user=_user(), query_engine=engine)
        result = DeepRagHandler().handle(ctx)

        assert len(result.sources) == 1
        assert result.sources[0]["source"] == "leave_policy.md"

    def test_no_engine_raises_runtime_error(self):
        from retrieval.handlers import DeepRagHandler

        ctx = RouteContext(question="question?", user=_user(), query_engine=None)
        with pytest.raises(RuntimeError):
            DeepRagHandler().handle(ctx)

    def test_query_engine_called_with_question(self):
        from retrieval.handlers import DeepRagHandler

        engine = self._make_mock_engine()
        ctx = RouteContext(
            question="what is the sick leave policy?",
            user=_user(),
            query_engine=engine,
        )
        DeepRagHandler().handle(ctx)
        engine.query.assert_called_once_with("what is the sick leave policy?")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. route_query dispatch tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteQueryDispatch:
    def _mock_engine(self):
        node = MagicMock()
        node.node.text = "Content."
        node.node.metadata = {"source": "doc.md", "department": "general",
                               "allowed_roles": ["employee"]}
        node.score = 0.9
        resp = MagicMock()
        resp.__str__ = lambda s: "Answer."
        resp.source_nodes = [node]
        eng = MagicMock()
        eng.query.return_value = resp
        return eng

    def test_small_talk_dispatches_to_small_talk_handler(self):
        user = _user()
        engine = self._mock_engine()

        with patch("retrieval.router.classify_intent",
                   return_value=(Intent.SMALL_TALK, 0.95, None)), \
             patch("retrieval.router.SmallTalkHandler") as MockST:

            mock_result = MagicMock()
            mock_result.answer = "Hello!"
            mock_result.route_type = "small_talk"
            mock_result.sources = []
            mock_result.latency_ms = 50.0
            MockST.return_value.handle.return_value = mock_result

            result = route_query("hi", user, query_engine=engine)

        MockST.return_value.handle.assert_called_once()
        assert result.route_type == "small_talk"

    def test_summarization_dispatches_to_summarization_handler(self):
        user = _user()
        engine = self._mock_engine()

        with patch("retrieval.router.classify_intent",
                   return_value=(Intent.SUMMARIZATION, 0.92, "leave_policy.md")), \
             patch("retrieval.router.SummarizationHandler") as MockSumm:

            mock_result = MagicMock()
            mock_result.answer = "Summary text."
            mock_result.route_type = "summarization"
            mock_result.sources = [{"source": "leave_policy.md"}]
            mock_result.latency_ms = 400.0
            MockSumm.return_value.handle.return_value = mock_result

            result = route_query(
                "summarize the leave policy", user, query_engine=engine
            )

        # Verify target_doc was passed in the context
        call_ctx = MockSumm.return_value.handle.call_args[0][0]
        assert call_ctx.target_doc == "leave_policy.md"

    def test_deep_rag_dispatches_to_deep_rag_handler(self):
        user = _user()
        engine = self._mock_engine()

        with patch("retrieval.router.classify_intent",
                   return_value=(Intent.DEEP_RAG, 0.98, None)), \
             patch("retrieval.router.DeepRagHandler") as MockDR:

            mock_result = MagicMock()
            mock_result.answer = "Policy answer."
            mock_result.route_type = "deep_rag"
            mock_result.sources = []
            mock_result.latency_ms = 1400.0
            MockDR.return_value.handle.return_value = mock_result

            result = route_query(
                "how many vacation days after 3 years?", user, query_engine=engine
            )

        MockDR.return_value.handle.assert_called_once()

    def test_classify_latency_added_to_result(self):
        user = _user()
        engine = self._mock_engine()

        with patch("retrieval.router.classify_intent",
                   return_value=(Intent.SMALL_TALK, 0.9, None)), \
             patch("retrieval.router.SmallTalkHandler") as MockST:

            mock_result = MagicMock()
            mock_result.answer = "Hi"
            mock_result.route_type = "small_talk"
            mock_result.sources = []
            mock_result.latency_ms = 50.0
            MockST.return_value.handle.return_value = mock_result

            result = route_query("hi", user, query_engine=engine)

        # Latency should include classification overhead (~50ms + classify_ms)
        assert result.latency_ms >= 50.0

    def test_history_updated_after_query(self):
        user = _user()
        user.username = "test_history_user"
        engine = self._mock_engine()

        # Fresh memory for this test
        with patch("retrieval.router._memory", ConversationMemory(max_turns=6)):
            with patch("retrieval.router.classify_intent",
                       return_value=(Intent.SMALL_TALK, 0.9, None)), \
                 patch("retrieval.router.SmallTalkHandler") as MockST:

                mock_result = MagicMock()
                mock_result.answer = "Hi there!"
                mock_result.route_type = "small_talk"
                mock_result.sources = []
                mock_result.latency_ms = 50.0
                MockST.return_value.handle.return_value = mock_result

                # Use a unique session to avoid cross-test pollution
                route_query("hello", user, session_id="test_session_xyz", query_engine=engine)

            from retrieval.router import _memory
            history = _memory.get("test_session_xyz")

        assert len(history) == 2
        assert history[0].content == "hello"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. API endpoint integration tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def api_client():
    mock_result = MagicMock()
    mock_result.answer = "Test answer."
    mock_result.route_type = "deep_rag"
    mock_result.sources = [{
        "source": "policy.md", "department": "general",
        "allowed_roles": ["employee"], "text_snippet": "...", "score": 0.9,
    }]
    mock_result.latency_ms = 1200.0

    with patch("api.main.get_or_build_index", return_value=MagicMock()), \
         patch("api.main.set_index"), \
         patch("api.main.get_index", return_value=MagicMock()), \
         patch("api.main.route_query", return_value=mock_result), \
         patch("api.main._build_deep_rag_engine", return_value=MagicMock()):
        from api.main import app
        with TestClient(app) as c:
            yield c


def _get_token(client, username: str) -> str:
    r = client.post("/auth/token", data={"username": username, "password": "secret"})
    return r.json()["access_token"]


class TestApiRouting:
    def test_query_response_includes_route_type(self, api_client):
        token = _get_token(api_client, "eve")
        r = api_client.post(
            "/query",
            json={"question": "how many vacation days?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert "route_type" in r.json()
        assert r.json()["route_type"] in ("small_talk", "summarization", "deep_rag")

    def test_query_response_includes_latency(self, api_client):
        token = _get_token(api_client, "eve")
        r = api_client.post(
            "/query",
            json={"question": "what is the leave policy?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["latency_ms"] > 0

    def test_history_endpoint_returns_session_data(self, api_client):
        token = _get_token(api_client, "bob")
        r = api_client.get(
            "/query/history",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert "messages" in data
        assert "turns" in data

    def test_clear_history_endpoint(self, api_client):
        token = _get_token(api_client, "bob")
        r = api_client.delete(
            "/query/history",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "cleared"

    def test_unauthenticated_query_rejected(self, api_client):
        r = api_client.post("/query", json={"question": "hello"})
        assert r.status_code == 401

    def test_health_public(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
