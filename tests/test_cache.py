"""
tests/test_cache.py
────────────────────
Tests for the semantic caching layer.

Covers:
  1. Namespace construction (role + dept_filter keying)
  2. Cache miss (empty namespace, below threshold)
  3. Cache hit (above threshold, correct entry returned)
  4. RBAC namespace isolation (different roles never share entries)
  5. Department filter namespace isolation
  6. TTL expiry behaviour (mocked Redis)
  7. LRU eviction when max_entries is exceeded
  8. Cache write (set) — verifies Redis data layout
  9. Small-talk never written to cache
  10. invalidate_all() flushes all namespaces
  11. invalidate_namespace() scoped flush
  12. stats() returns correct counts
  13. Router integration — cache check before classify_intent
  14. Router integration — result written to cache after deep_rag
  15. API endpoint — /cache/stats returns expected shape
  16. API endpoint — DELETE /cache requires admin
  17. API endpoint — cache_hit field in QueryResponse
  18. Graceful degradation — Redis unavailable at init

Run:
    pytest tests/test_cache.py -v
"""

import json
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.semantic_cache import (
    CacheEntry,
    SemanticCache,
    _cosine_similarity,
    _namespace,
    get_semantic_cache,
    init_semantic_cache,
)
from retrieval.handlers import ConversationTurn, RouteResult


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_redis() -> MagicMock:
    """Return a MagicMock that mimics the minimal Redis interface we use."""
    r = MagicMock()
    r.ping.return_value = True
    # zrange returns empty list by default (cache miss)
    r.zrange.return_value = []
    r.pipeline.return_value.__enter__ = MagicMock(return_value=MagicMock())
    r.pipeline.return_value.__exit__ = MagicMock(return_value=False)
    pipe = MagicMock()
    pipe.execute.return_value = [True, True, True, True, True]
    r.pipeline.return_value = pipe
    return r


def _fake_vec(seed: int = 42, dim: int = 8) -> list[float]:
    """Produce a unit-normalised random vector (small dim for test speed)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _similar_vec(base: list[float], noise: float = 0.02) -> list[float]:
    """Return a vector very similar to `base` (cosine sim ≈ 0.998)."""
    v = np.array(base, dtype=np.float32)
    rng = np.random.default_rng(99)
    v = v + noise * rng.standard_normal(len(base)).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _different_vec(dim: int = 8) -> list[float]:
    """Return a vector very different from _fake_vec (opposite direction)."""
    v = np.array(_fake_vec(42, dim), dtype=np.float32) * -1
    return (v / np.linalg.norm(v)).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Namespace construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestNamespace:
    def test_role_only(self):
        assert _namespace("hr", None) == "hr:"

    def test_role_and_dept(self):
        assert _namespace("hr", "engineering") == "hr:engineering"

    def test_role_and_empty_dept(self):
        assert _namespace("admin", "") == "admin:"

    def test_different_roles_different_namespaces(self):
        assert _namespace("hr", None) != _namespace("engineering", None)

    def test_same_role_different_dept(self):
        assert _namespace("hr", "finance") != _namespace("hr", "engineering")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cosine similarity helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def test_identical_vectors_returns_one(self):
        v = _fake_vec(1)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-5

    def test_similar_vectors_above_threshold(self):
        v  = _fake_vec(1)
        v2 = _similar_vec(v)
        sim = _cosine_similarity(v, v2)
        assert sim > 0.90

    def test_opposite_vectors_returns_negative(self):
        v  = _fake_vec(1)
        v2 = [-x for x in v]
        sim = _cosine_similarity(v, v2)
        assert sim < -0.99

    def test_orthogonal_vectors_near_zero(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = _cosine_similarity(v1, v2)
        assert abs(sim) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SemanticCache.get() — cache miss cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheMiss:
    def test_empty_namespace_returns_none(self):
        redis = _mock_redis()
        redis.zrange.return_value = []
        cache = SemanticCache(redis)

        with patch("cache.semantic_cache._embed", return_value=_fake_vec()), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True):
            result = cache.get("what are the holidays?", "employee")

        assert result is None

    def test_below_threshold_returns_none(self):
        redis = _mock_redis()
        entry_id = "entry-001"
        redis.zrange.return_value = [entry_id.encode()]

        # Very different vector — low cosine similarity
        pipe = MagicMock()
        pipe.execute.return_value = [json.dumps(_different_vec()).encode()]
        redis.pipeline.return_value = pipe

        cache = SemanticCache(redis)

        with patch("cache.semantic_cache._embed", return_value=_fake_vec()), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True), \
             patch("config.settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.92):
            result = cache.get("what are the holidays?", "employee")

        assert result is None

    def test_disabled_cache_always_returns_none(self):
        cache = SemanticCache(_mock_redis())
        with patch("config.settings.SEMANTIC_CACHE_ENABLED", False):
            result = cache.get("any question", "admin")
        assert result is None

    def test_redis_unavailable_returns_none(self):
        from redis.exceptions import RedisError
        redis = _mock_redis()
        redis.zrange.side_effect = RedisError("Connection refused")
        cache = SemanticCache(redis)

        with patch("config.settings.SEMANTIC_CACHE_ENABLED", True):
            result = cache.get("question", "hr")

        assert result is None

    def test_embedding_failure_returns_none(self):
        redis = _mock_redis()
        redis.zrange.return_value = ["entry-001".encode()]
        cache = SemanticCache(redis)

        with patch("cache.semantic_cache._embed", side_effect=Exception("API down")), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True):
            result = cache.get("question", "hr")

        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SemanticCache.get() — cache hit
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheHit:
    def _setup_hit(self, redis: MagicMock, query_vec: list[float],
                   cached_vec: list[float]) -> None:
        """Wire a Redis mock to return a cache hit for the given vectors."""
        entry_id = "entry-abc"
        redis.zrange.return_value = [entry_id.encode()]

        # Batch vector read
        pipe = MagicMock()
        pipe.execute.return_value = [json.dumps(cached_vec).encode()]
        redis.pipeline.return_value = pipe

        # Full entry read
        redis.hgetall.return_value = {
            b"question":   b"what are the company holidays?",
            b"answer":     b"The company observes 11 federal holidays.",
            b"sources":    json.dumps([{"source": "it_security_policy.md"}]).encode(),
            b"route_type": b"deep_rag",
            b"created_at": b"1700000000.0",
        }

    def test_returns_cache_entry_on_hit(self):
        redis = _mock_redis()
        query_vec  = _fake_vec(1)
        cached_vec = _similar_vec(query_vec)
        self._setup_hit(redis, query_vec, cached_vec)
        cache = SemanticCache(redis)

        with patch("cache.semantic_cache._embed", return_value=query_vec), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True), \
             patch("config.settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.92):
            result = cache.get("what are the holidays?", "employee")

        assert result is not None
        assert isinstance(result, CacheEntry)

    def test_hit_returns_correct_answer(self):
        redis = _mock_redis()
        query_vec  = _fake_vec(1)
        cached_vec = _similar_vec(query_vec)
        self._setup_hit(redis, query_vec, cached_vec)
        cache = SemanticCache(redis)

        with patch("cache.semantic_cache._embed", return_value=query_vec), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True), \
             patch("config.settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.92):
            result = cache.get("company holidays?", "employee")

        assert result.answer == "The company observes 11 federal holidays."

    def test_hit_returns_route_type(self):
        redis = _mock_redis()
        query_vec  = _fake_vec(2)
        cached_vec = _similar_vec(query_vec)
        self._setup_hit(redis, query_vec, cached_vec)
        cache = SemanticCache(redis)

        with patch("cache.semantic_cache._embed", return_value=query_vec), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True), \
             patch("config.settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.92):
            result = cache.get("holidays?", "employee")

        assert result.route_type == "deep_rag"

    def test_hit_increments_hit_counter(self):
        redis = _mock_redis()
        query_vec  = _fake_vec(3)
        cached_vec = _similar_vec(query_vec)
        self._setup_hit(redis, query_vec, cached_vec)
        cache = SemanticCache(redis)

        with patch("cache.semantic_cache._embed", return_value=query_vec), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True), \
             patch("config.settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.92):
            cache.get("holidays?", "employee")
            cache.get("holidays?", "employee")

        assert cache._hits == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RBAC namespace isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestRBACIsolation:
    def test_different_roles_query_different_redis_keys(self):
        redis = _mock_redis()
        redis.zrange.return_value = []
        cache = SemanticCache(redis)

        with patch("cache.semantic_cache._embed", return_value=_fake_vec()), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True):
            cache.get("salary bands?", "hr")
            cache.get("salary bands?", "employee")

        # zrange should have been called with two different index keys
        calls = [str(c) for c in redis.zrange.call_args_list]
        assert any("hr:" in s for s in calls)
        assert any("employee:" in s for s in calls)

    def test_same_question_different_role_is_cache_miss(self):
        """An HR cache entry must not be served to an employee."""
        redis = _mock_redis()
        entry_id = "entry-hr"

        def zrange_side_effect(key, *args, **kwargs):
            # Only the hr: namespace has entries
            if "hr:" in str(key):
                return [entry_id.encode()]
            return []

        redis.zrange.side_effect = zrange_side_effect

        pipe = MagicMock()
        pipe.execute.return_value = [json.dumps(_similar_vec(_fake_vec(1))).encode()]
        redis.pipeline.return_value = pipe

        cache = SemanticCache(redis)

        with patch("cache.semantic_cache._embed", return_value=_fake_vec(1)), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True), \
             patch("config.settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.92):
            # employee asks same question — should miss because hr: namespace used for HR only
            result = cache.get("salary bands?", "employee")

        # employee namespace is empty → cache miss
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SemanticCache.set()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheSet:
    def test_set_calls_redis_pipeline(self):
        redis = _mock_redis()
        pipe = MagicMock()
        pipe.execute.return_value = [True] * 6
        redis.pipeline.return_value = pipe
        redis.zcard.return_value = 1

        cache = SemanticCache(redis)
        with patch("cache.semantic_cache._embed", return_value=_fake_vec()), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True), \
             patch("config.settings.SEMANTIC_CACHE_TTL_SECONDS", 3600):
            result = cache.set(
                question="what are the holidays?",
                answer="11 federal holidays.",
                sources=[],
                route_type="deep_rag",
                role="employee",
            )

        assert result is True
        pipe.hset.assert_called_once()  # entry hash written
        pipe.set.assert_called_once()   # vector written

    def test_small_talk_never_written(self):
        redis = _mock_redis()
        cache = SemanticCache(redis)

        with patch("config.settings.SEMANTIC_CACHE_ENABLED", True):
            result = cache.set(
                question="hi there",
                answer="Hello!",
                sources=[],
                route_type="small_talk",
                role="employee",
            )

        assert result is False
        redis.pipeline.assert_not_called()

    def test_disabled_cache_set_returns_false(self):
        cache = SemanticCache(_mock_redis())
        with patch("config.settings.SEMANTIC_CACHE_ENABLED", False):
            result = cache.set("q", "a", [], "deep_rag", "hr")
        assert result is False

    def test_lru_eviction_triggered_when_over_limit(self):
        redis = _mock_redis()
        pipe = MagicMock()
        pipe.execute.return_value = [True] * 6
        redis.pipeline.return_value = pipe
        redis.zcard.return_value = 10001    # over max_entries
        redis.zrange.return_value = [b"old-entry"]

        cache = SemanticCache(redis)
        with patch("cache.semantic_cache._embed", return_value=_fake_vec()), \
             patch("config.settings.SEMANTIC_CACHE_ENABLED", True), \
             patch("config.settings.SEMANTIC_CACHE_MAX_ENTRIES", 10000):
            cache.set("new question", "answer", [], "deep_rag", "employee")

        # delete should have been called for the old entry
        redis.delete.assert_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Invalidation
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvalidation:
    def test_invalidate_all_scans_and_deletes(self):
        redis = _mock_redis()
        redis.scan_iter.return_value = [
            b"semantic_cache:hr::index",
            b"semantic_cache:hr::entry:abc",
            b"semantic_cache:hr::vec:abc",
        ]
        cache = SemanticCache(redis)
        n = cache.invalidate_all()

        assert n == 3
        redis.delete.assert_called_once()

    def test_invalidate_namespace_targets_correct_pattern(self):
        redis = _mock_redis()
        redis.scan_iter.return_value = [b"semantic_cache:hr::entry:abc"]
        cache = SemanticCache(redis)

        cache.invalidate_namespace("hr", None)

        # Verify scan was called with the hr namespace pattern
        scan_call = redis.scan_iter.call_args
        assert "hr:" in str(scan_call)

    def test_redis_error_in_invalidate_returns_zero(self):
        from redis.exceptions import RedisError
        redis = _mock_redis()
        redis.scan_iter.side_effect = RedisError("Connection lost")
        cache = SemanticCache(redis)
        n = cache.invalidate_all()
        assert n == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestStats:
    def test_stats_returns_correct_hit_rate(self):
        redis = _mock_redis()
        redis.scan_iter.return_value = []
        cache = SemanticCache(redis)
        cache._hits   = 7
        cache._misses = 3

        with patch("config.settings.SEMANTIC_CACHE_ENABLED", True):
            stats = cache.stats()

        assert stats.hit_rate == 0.7
        assert stats.total_hits == 7
        assert stats.total_misses == 3

    def test_stats_zero_queries_hit_rate_is_zero(self):
        redis = _mock_redis()
        redis.scan_iter.return_value = []
        cache = SemanticCache(redis)
        stats = cache.stats()
        assert stats.hit_rate == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Router integration
# ═══════════════════════════════════════════════════════════════════════════════

from auth.jwt_handler import CurrentUser

def _user(role: str = "employee") -> CurrentUser:
    return CurrentUser(username=f"test_{role}", role=role, full_name="Test User")


class TestRouterCacheIntegration:
    def test_cache_hit_bypasses_classify_intent(self):
        from retrieval.router import route_query

        cached_entry = CacheEntry(
            entry_id="e1",
            question="what are the holidays?",
            answer="11 federal holidays.",
            sources=[],
            route_type="deep_rag",
            created_at=1700000000.0,
            similarity=0.95,
            lookup_ms=85.0,
        )

        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_entry

        with patch("retrieval.router.get_semantic_cache", return_value=mock_cache), \
             patch("retrieval.router.classify_intent") as mock_classify:

            result = route_query(
                question="what are the 2026 holidays?",
                user=_user(),
                query_engine=MagicMock(),
            )

        # classify_intent must NOT have been called — cache hit bypasses it
        mock_classify.assert_not_called()
        assert result.route_type == "cache_hit"
        assert result.answer == "11 federal holidays."

    def test_cache_miss_calls_classify_intent(self):
        from retrieval.router import route_query

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # cache miss

        mock_engine = MagicMock()
        resp = MagicMock()
        resp.__str__ = lambda s: "Policy answer."
        resp.source_nodes = []
        mock_engine.query.return_value = resp

        with patch("retrieval.router.get_semantic_cache", return_value=mock_cache), \
             patch("retrieval.router.classify_intent",
                   return_value=("deep_rag", 0.95, None)) as mock_classify, \
             patch("retrieval.router.DeepRagHandler") as MockDR:

            mock_dr_result = MagicMock()
            mock_dr_result.answer = "Policy answer."
            mock_dr_result.route_type = "deep_rag"
            mock_dr_result.sources = []
            mock_dr_result.latency_ms = 1200.0
            MockDR.return_value.handle.return_value = mock_dr_result

            result = route_query(
                question="how many vacation days?",
                user=_user(),
                query_engine=mock_engine,
            )

        mock_classify.assert_called_once()

    def test_deep_rag_result_written_to_cache(self):
        from retrieval.router import route_query

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch("retrieval.router.get_semantic_cache", return_value=mock_cache), \
             patch("retrieval.router.classify_intent",
                   return_value=("deep_rag", 0.95, None)), \
             patch("retrieval.router.DeepRagHandler") as MockDR:

            mock_result = MagicMock()
            mock_result.answer = "15 vacation days per year."
            mock_result.route_type = "deep_rag"
            mock_result.sources = []
            mock_result.latency_ms = 1100.0
            MockDR.return_value.handle.return_value = mock_result

            route_query("vacation days?", _user(), query_engine=MagicMock())

        mock_cache.set.assert_called_once()
        set_kwargs = mock_cache.set.call_args.kwargs
        assert set_kwargs["route_type"] == "deep_rag"

    def test_small_talk_result_not_written_to_cache(self):
        from retrieval.router import route_query

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch("retrieval.router.get_semantic_cache", return_value=mock_cache), \
             patch("retrieval.router.classify_intent",
                   return_value=("small_talk", 0.95, None)), \
             patch("retrieval.router.SmallTalkHandler") as MockST:

            mock_result = MagicMock()
            mock_result.answer = "Hello!"
            mock_result.route_type = "small_talk"
            mock_result.sources = []
            mock_result.latency_ms = 80.0
            MockST.return_value.handle.return_value = mock_result

            route_query("hi there", _user(), query_engine=MagicMock())

        mock_cache.set.assert_not_called()

    def test_department_filter_passed_to_cache(self):
        from retrieval.router import route_query

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch("retrieval.router.get_semantic_cache", return_value=mock_cache), \
             patch("retrieval.router.classify_intent",
                   return_value=("deep_rag", 0.95, None)), \
             patch("retrieval.router.DeepRagHandler") as MockDR:

            mock_result = MagicMock()
            mock_result.answer = "Answer."
            mock_result.route_type = "deep_rag"
            mock_result.sources = []
            mock_result.latency_ms = 500.0
            MockDR.return_value.handle.return_value = mock_result

            route_query(
                "expense policy?",
                _user(),
                query_engine=MagicMock(),
                department_filter="finance",
            )

        get_call = mock_cache.get.call_args
        assert get_call.kwargs.get("dept_filter") == "finance" or \
               (len(get_call.args) >= 3 and get_call.args[2] == "finance")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. API endpoint tests
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def api_client():
    mock_cache = MagicMock()
    mock_cache.stats.return_value = MagicMock(
        enabled=True, total_hits=42, total_misses=58,
        hit_rate=0.42, namespace_sizes={"employee:": 150},
        threshold=0.92, ttl_seconds=86400,
    )
    mock_cache.invalidate_all.return_value = 300

    mock_route_result = MagicMock()
    mock_route_result.answer = "Test answer."
    mock_route_result.route_type = "deep_rag"
    mock_route_result.sources = []
    mock_route_result.latency_ms = 1200.0

    with patch("api.main.get_or_build_index", return_value=MagicMock()), \
         patch("api.main.set_index"), \
         patch("api.main.get_index", return_value=MagicMock()), \
         patch("api.main.init_semantic_cache", return_value=mock_cache), \
         patch("api.main.get_semantic_cache", return_value=mock_cache), \
         patch("api.main.route_query", return_value=mock_route_result), \
         patch("api.main._build_deep_rag_engine", return_value=MagicMock()):
        from api.main import app
        with TestClient(app) as c:
            yield c


def _get_token(client: TestClient, username: str) -> str:
    r = client.post("/auth/token", data={"username": username, "password": "secret"})
    return r.json()["access_token"]


class TestCacheApiEndpoints:
    def test_cache_stats_returns_200(self, api_client):
        token = _get_token(api_client, "eve")
        r = api_client.get(
            "/cache/stats",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200

    def test_cache_stats_has_expected_fields(self, api_client):
        token = _get_token(api_client, "eve")
        r = api_client.get("/cache/stats", headers={"Authorization": f"Bearer {token}"})
        data = r.json()
        assert "enabled" in data
        assert "total_hits" in data
        assert "hit_rate" in data

    def test_cache_flush_requires_admin(self, api_client):
        token = _get_token(api_client, "eve")  # employee role
        r = api_client.delete("/cache", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403

    def test_cache_flush_allowed_for_admin(self, api_client):
        token = _get_token(api_client, "admin")
        r = api_client.delete("/cache", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_query_response_has_cache_hit_field(self, api_client):
        token = _get_token(api_client, "eve")
        r = api_client.post(
            "/query",
            json={"question": "what are the holidays?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert "cache_hit" in r.json()

    def test_deep_rag_result_cache_hit_is_false(self, api_client):
        token = _get_token(api_client, "eve")
        r = api_client.post(
            "/query",
            json={"question": "vacation policy?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        # route_type is "deep_rag" so cache_hit should be False
        assert r.json()["cache_hit"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Graceful degradation
# ═══════════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    def test_init_returns_none_when_redis_unavailable(self):
        with patch("cache.semantic_cache.Redis") as MockRedis:
            MockRedis.return_value.ping.side_effect = Exception("Connection refused")
            with patch("cache.semantic_cache._cache_instance", None):
                result = init_semantic_cache()
        assert result is None

    def test_none_cache_does_not_break_router(self):
        """When cache is None, route_query should work exactly as before."""
        from retrieval.router import route_query

        with patch("retrieval.router.get_semantic_cache", return_value=None), \
             patch("retrieval.router.classify_intent",
                   return_value=("small_talk", 0.95, None)), \
             patch("retrieval.router.SmallTalkHandler") as MockST:

            mock_result = MagicMock()
            mock_result.answer = "Hello!"
            mock_result.route_type = "small_talk"
            mock_result.sources = []
            mock_result.latency_ms = 80.0
            MockST.return_value.handle.return_value = mock_result

            result = route_query("hi", _user(), query_engine=MagicMock())

        assert result.answer == "Hello!"
        assert result.route_type == "small_talk"
