"""
retrieval/router.py
────────────────────
The agentic router — sits at the very front of the pipeline and decides
which handler processes each incoming query.

Architecture (updated for Feature 5 — semantic caching)
────────────────────────────────────────────────────────

  User query
      │
      ▼
  SemanticCache.get()        ← embed + cosine similarity vs Redis, ~100ms
      │
      ├──► CACHE HIT   ──► return cached answer immediately           (~100ms)
      │
      └──► CACHE MISS
               │
               ▼
           classify_intent()  ← gpt-4o-mini, structured JSON          (~150ms)
               │
               ├──► SMALL_TALK   ──► SmallTalkHandler                (~50ms)
               │
               ├──► SUMMARIZATION ──► SummarizationHandler           (~500ms)
               │                         │
               └──► DEEP_RAG      ──► DeepRagHandler                 (~1200ms)
                                         │
                                   (RBAC pre-filter)
                                   Hybrid search
                                   AutoMerging
                                   Cohere Rerank
                                   GPT-4o
                                         │
                                         ▼
                                  SemanticCache.set()
                                  (write result to Redis for future hits)

Cache scoping
──────────────
Cache is partitioned by (user_role, dept_filter).  Two users with different
roles never share cache entries — this is the RBAC guarantee of the cache.

Small-talk responses are NEVER written to cache (they are conversational,
session-specific, and cheap enough that caching them provides no value).

Conversation memory
────────────────────
The router maintains an in-process LRU conversation store keyed by session_id.
Each session holds the last CONVERSATION_MEMORY_TURNS (user, assistant) pairs.

  - SmallTalk and Summarization receive full history (they are conversational).
  - DeepRAG receives NO history — retrieval is stateless by design.

Why a single classifier LLM call instead of keyword heuristics?
────────────────────────────────────────────────────────────────
Keyword rules break quickly:
  "summarize how the expense policy works" → should be DEEP_RAG
  "hi, what is the expense policy?" → SMALL_TALK despite mentioning a policy

A cheap LLM classifier (gpt-4o-mini, ~50ms, ~$0.00002/call) handles these
correctly and can use conversation history to resolve pronouns.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from enum import Enum
from typing import Optional

from loguru import logger
from openai import OpenAI

from auth.jwt_handler import CurrentUser
from auth.rbac import expand_roles
from cache.semantic_cache import CacheEntry, SemanticCache, get_semantic_cache
from config import settings
from observability import traced, update_current_observation, update_current_trace
from retrieval.handlers import (
    ConversationTurn,
    DeepRagHandler,
    RouteContext,
    RouteResult,
    SmallTalkHandler,
    SummarizationHandler,
)


# ── Intent enum ───────────────────────────────────────────────────────────────

class Intent(str, Enum):
    SMALL_TALK     = "small_talk"
    SUMMARIZATION  = "summarization"
    DEEP_RAG       = "deep_rag"


# ── Classification prompt ─────────────────────────────────────────────────────

_CLASSIFIER_SYSTEM = """\
You are a query intent classifier for a company internal document RAG system.
Classify each user query into exactly one of three intents.

INTENT DEFINITIONS:

  small_talk
    - Greetings, farewells, thank-yous, compliments
    - Questions about what the system can do ("what can you help me with?")
    - Conversational follow-ups that don't require document lookup
    - Questions the assistant can answer from general knowledge without documents
    Examples: "hi", "thanks", "what are your capabilities?", "who made you?",
              "that's helpful, can you rephrase it?", "what do you mean by PTO?"

  summarization
    - Explicit requests to summarize, overview, or give a rundown of a SPECIFIC document
    - "summarize the X policy", "give me an overview of Y", "what's in the Z guide?"
    - Key signal: user names a specific document and wants ALL of it, not a fact from it
    - If the user says "summarize [document name]" → this intent, extract target_doc
    Examples: "summarize the leave policy", "can you give me an overview of the
              onboarding guide?", "what does the expense policy document cover?"

  deep_rag
    - Any substantive question requiring precise fact retrieval from documents
    - Questions starting with "how", "what", "when", "who", "can I", "do I need to"
      when they ask for a SPECIFIC FACT (not a full document summary)
    - Policy questions, procedural questions, rule lookups
    Examples: "how many vacation days do I get after 3 years?",
              "what's the approval threshold for expenses over $500?",
              "do I need a doctor's note for sick leave?",
              "when does the probationary period end?"

RESPONSE FORMAT — respond ONLY with valid JSON, no markdown, no explanation:
{
  "intent": "small_talk" | "summarization" | "deep_rag",
  "confidence": 0.0-1.0,
  "target_doc": "<filename or empty string>",
  "reasoning": "<one sentence>"
}

For summarization intent, set target_doc to the most likely filename:
  "leave policy" → "leave_policy.md"
  "expense report" → "expense_policy.md"
  "onboarding" → "onboarding_guide.md"
  "security policy" / "IT security" → "it_security_policy.md"
  "data retention" → "data_retention_policy.md"
  "handbook" → "employee_handbook.md"
  "incident response" → "incident_response.md"
If unsure of the filename, use the user's exact words as target_doc.
"""

_CLASSIFIER_USER_TMPL = """\
CONVERSATION HISTORY (last {n} turns):
{history}

CURRENT QUERY: {query}

Classify the intent of CURRENT QUERY only."""


# ── Intent classification ─────────────────────────────────────────────────────

@traced(name="classify_intent", as_type="generation")
def classify_intent(
    question: str,
    history: list[ConversationTurn],
) -> tuple[Intent, float, Optional[str]]:
    """
    Classify the intent of `question` using a single gpt-4o-mini call.

    Traced manually: this goes through the raw openai SDK, so the LlamaIndex
    global handler never sees it.

    Returns:
        (intent, confidence, target_doc)
        target_doc is non-empty only for SUMMARIZATION intent.
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    history_text = "\n".join(
        f"  {t.role.upper()}: {t.content[:200]}"
        for t in history[-settings.CONVERSATION_MEMORY_TURNS:]
    ) or "(no history)"

    user_message = _CLASSIFIER_USER_TMPL.format(
        n=len(history),
        history=history_text,
        query=question,
    )

    try:
        response = client.chat.completions.create(
            model=settings.ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=150,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw  = response.choices[0].message.content.strip()
        data = json.loads(raw)

        intent_str = data.get("intent", "deep_rag")
        confidence = float(data.get("confidence", 0.9))
        target_doc = data.get("target_doc", "") or ""
        reasoning  = data.get("reasoning", "")

        intent = Intent(intent_str)

        # Record model, token usage and the classification result on the span
        # so the routing decision is visible without re-reading logs.
        usage = getattr(response, "usage", None)
        update_current_observation(
            model=settings.ROUTER_MODEL,
            usage={
                "input":  getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
                "total":  getattr(usage, "total_tokens", None),
            } if usage else None,
            intent=intent.value,
            confidence=confidence,
            target_doc=target_doc or None,
            reasoning=reasoning,
            history_turns=len(history),
        )

        logger.debug(
            f"Intent classified: {intent.value!r} "
            f"(confidence={confidence:.2f}, target_doc={target_doc!r}) "
            f"— {reasoning}"
        )

        return intent, confidence, target_doc if target_doc else None

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(
            f"Intent classifier returned malformed JSON ({e}) — "
            f"defaulting to DEEP_RAG"
        )
        # Flag the fallback on the span — a silent default to DEEP_RAG is
        # exactly the kind of thing that hides a broken classifier.
        update_current_observation(
            level="WARNING",
            status_message=f"malformed classifier JSON: {e}",
            intent=Intent.DEEP_RAG.value,
            fallback=True,
        )
        return Intent.DEEP_RAG, 0.5, None

    except Exception as e:
        logger.error(f"Intent classification failed ({e}) — defaulting to DEEP_RAG")
        update_current_observation(
            level="ERROR",
            status_message=f"classifier call failed: {e}",
            intent=Intent.DEEP_RAG.value,
            fallback=True,
        )
        return Intent.DEEP_RAG, 0.5, None


# ── Conversation memory ───────────────────────────────────────────────────────

class ConversationMemory:
    """
    Thread-safe LRU store of conversation histories keyed by session_id.

    Each session holds a list of ConversationTurn objects (alternating
    user/assistant). We keep the last CONVERSATION_MEMORY_TURNS pairs.

    In production this should be backed by Redis so history survives
    worker restarts and scales across multiple API replicas. For a
    portfolio project, in-process memory is sufficient.
    """

    def __init__(
        self,
        max_turns: int | None = None,
        max_sessions: int | None = None,
    ) -> None:
        self._max_turns    = max_turns    or settings.CONVERSATION_MEMORY_TURNS
        self._max_sessions = max_sessions or settings.CONVERSATION_MAX_SESSIONS
        self._store: OrderedDict[str, list[ConversationTurn]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[ConversationTurn]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def append(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        with self._lock:
            if session_id not in self._store:
                if len(self._store) >= self._max_sessions:
                    self._store.popitem(last=False)
                self._store[session_id] = []
            else:
                self._store.move_to_end(session_id)

            turns = self._store[session_id]
            turns.append(ConversationTurn(role="user",      content=user_message))
            turns.append(ConversationTurn(role="assistant", content=assistant_message))

            if len(turns) > self._max_turns * 2:
                self._store[session_id] = turns[-(self._max_turns * 2):]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._store)


_memory = ConversationMemory()


def get_memory() -> ConversationMemory:
    return _memory


# ── Cache helper ──────────────────────────────────────────────────────────────

def _result_from_cache(entry: CacheEntry, question: str) -> RouteResult:
    """Wrap a CacheEntry in a RouteResult for the router to return."""
    return RouteResult(
        answer=entry.answer,
        route_type="cache_hit",
        sources=entry.sources,
        latency_ms=entry.lookup_ms,
    )


# ── Main dispatch function ────────────────────────────────────────────────────

@traced(name="route_query")
def route_query(
    question:        str,
    user:            CurrentUser,
    session_id:      Optional[str] = None,
    query_engine=None,
    department_filter: Optional[str] = None,
) -> RouteResult:
    """
    Semantic-cache check → intent classify → dispatch to handler.

    Pipeline (in order):
      1. SemanticCache.get() — return cached answer if cosine sim ≥ threshold
      2. classify_intent()  — gpt-4o-mini call
      3. Handler dispatch   — SmallTalk / Summarization / DeepRAG
      4. SemanticCache.set() — write non-small-talk results to cache

    Args:
        question:          Raw user query string.
        user:              Authenticated CurrentUser from JWT.
        session_id:        Conversation session key (defaults to username).
        query_engine:      Pre-built DeepRAG engine from api/main.py.
        department_filter: Department restriction (used as cache namespace key).

    Returns:
        RouteResult with answer, sources, route_type, and latency_ms.
    """
    effective_session = session_id or user.username

    # Request-level context on the root trace. RBAC gets no span of its own —
    # it builds a filter object with no I/O — so the expanded role list is
    # recorded here instead, making "what was this user allowed to see?"
    # answerable straight from the trace.
    update_current_trace(
        user_id=user.username,
        session_id=effective_session,
        input=question,
        user_role=user.role,
        accessible_roles=expand_roles(user.role),
        department_filter=department_filter,
        streaming=False,
    )

    # ── Step 1: Semantic cache check ──────────────────────────────────────────
    # This runs BEFORE intent classification, saving ~150ms on a cache hit.
    cache: Optional[SemanticCache] = get_semantic_cache()
    if cache:
        cached_entry = cache.get(question, user.role, department_filter)
        if cached_entry:
            result = _result_from_cache(cached_entry, question)
            _memory.append(
                session_id=effective_session,
                user_message=question,
                assistant_message=result.answer[:500],
            )
            update_current_trace(
                output=result.answer,
                route_type="cache_hit",
                cache_hit=True,
                cache_similarity=cached_entry.similarity,
                tags=["cache_hit"],
            )
            logger.info(
                f"Cache HIT returned — ns={user.role}:{department_filter or ''!r} "
                f"sim={cached_entry.similarity} user={user.username!r}"
            )
            return result

    # ── Step 2: Intent classification ─────────────────────────────────────────
    history = _memory.get(effective_session)

    t_classify_start = time.perf_counter()
    intent, confidence, target_doc = classify_intent(question, history)
    classify_ms = (time.perf_counter() - t_classify_start) * 1000

    logger.info(
        f"Router: intent={intent.value!r} confidence={confidence:.2f} "
        f"user={user.username!r} session={effective_session!r} "
        f"classify_ms={classify_ms:.0f}"
    )

    # ── Step 3: Build context and dispatch ────────────────────────────────────
    ctx = RouteContext(
        question=question,
        user=user,
        history=history,
        query_engine=query_engine,
        target_doc=target_doc,
    )

    if intent == Intent.SMALL_TALK:
        result = SmallTalkHandler().handle(ctx)

    elif intent == Intent.SUMMARIZATION:
        result = SummarizationHandler().handle(ctx)

    else:
        if query_engine is None:
            logger.error("DeepRAG route selected but no query_engine provided — fallback to SmallTalk")
            result = SmallTalkHandler().handle(ctx)
        else:
            result = DeepRagHandler().handle(ctx)

    # ── Step 4: Update conversation memory ────────────────────────────────────
    answer_for_memory = result.answer
    if result.route_type == "deep_rag" and result.sources:
        answer_for_memory = result.answer[:500] + ("…" if len(result.answer) > 500 else "")

    _memory.append(
        session_id=effective_session,
        user_message=question,
        assistant_message=answer_for_memory,
    )

    # ── Step 5: Write to cache (non-small-talk results only) ──────────────────
    if cache and result.route_type not in ("small_talk",):
        cache.set(
            question=question,
            answer=result.answer,
            sources=result.sources,
            route_type=result.route_type,
            role=user.role,
            dept_filter=department_filter,
        )

    result.latency_ms = round(result.latency_ms + classify_ms, 1)

    update_current_trace(
        output=result.answer,
        route_type=result.route_type,
        cache_hit=False,
        intent=intent.value,
        confidence=confidence,
        classify_ms=round(classify_ms, 1),
        total_ms=result.latency_ms,
        n_sources=len(result.sources),
        tags=[result.route_type],
    )

    logger.info(
        f"Router complete: route={result.route_type!r} "
        f"total_ms={result.latency_ms:.0f} "
        f"(classify={classify_ms:.0f} + handle={result.latency_ms - classify_ms:.0f})"
    )

    return result


# ── Streaming variant ─────────────────────────────────────────────────────────

def _sse_answer_only(items: list) -> str:
    """
    Rebuild the assistant's answer from the raw SSE frames a stream yielded.

    Without this, Langfuse would record the span's output as every yielded
    string concatenated — i.e. the wire format, control events and all
    (`data: [ROUTE]deep_rag`, `data: [DONE]`). We want the answer text.
    """
    answer = []
    for item in items:
        if not isinstance(item, str) or not item.startswith("data: "):
            continue
        payload = item[len("data: "):].rstrip("\n")
        # Skip control frames: [ROUTE], [CACHE], [SOURCES], [DONE]
        if payload.startswith("[") and "]" in payload[:12]:
            continue
        answer.append(payload)
    return "".join(answer)


@traced(name="route_query_stream", transform_to_string=_sse_answer_only)
async def route_query_stream(
    question:          str,
    user:              CurrentUser,
    session_id:        Optional[str] = None,
    query_engine=None,
    department_filter: Optional[str] = None,
):
    """
    Async generator version of route_query for the SSE streaming endpoint.

    Cache-hit SSE sequence:
        data: [ROUTE]cache_hit\\n\\n
        data: [CACHE]<similarity_score>\\n\\n
        data: <word1> \\n\\n
        data: <word2> \\n\\n
        ...
        data: [SOURCES]{...}\\n\\n
        data: [DONE]\\n\\n

    Cache-miss SSE sequence (unchanged from Feature 4):
        data: [ROUTE]<intent>\\n\\n
        data: <token>...\\n\\n
        data: [SOURCES]{...}\\n\\n
        data: [DONE]\\n\\n
    """
    import json as _json
    effective_session = session_id or user.username

    # Same request-level context as the non-streaming path. The span this
    # attaches to stays open until the generator is exhausted, so it covers
    # the whole stream rather than just its construction.
    update_current_trace(
        user_id=user.username,
        session_id=effective_session,
        input=question,
        user_role=user.role,
        accessible_roles=expand_roles(user.role),
        department_filter=department_filter,
        streaming=True,
    )

    # ── Cache check ───────────────────────────────────────────────────────────
    cache: Optional[SemanticCache] = get_semantic_cache()
    if cache:
        cached_entry = cache.get(question, user.role, department_filter)
        if cached_entry:
            update_current_trace(
                route_type="cache_hit",
                cache_hit=True,
                cache_similarity=cached_entry.similarity,
                tags=["cache_hit", "streaming"],
            )
            yield f"data: [ROUTE]cache_hit\n\n"
            # Send a cache metadata event so the UI can show the similarity score
            yield f"data: [CACHE]{cached_entry.similarity:.4f}\n\n"

            # Stream the cached answer word-by-word for natural UX
            # (feels like a fast stream rather than a wall of text appearing)
            words = cached_entry.answer.split(" ")
            for i, word in enumerate(words):
                # Add space back except after last word
                token = word + (" " if i < len(words) - 1 else "")
                yield f"data: {token}\n\n"

            _memory.append(
                session_id=effective_session,
                user_message=question,
                assistant_message=cached_entry.answer[:500],
            )
            yield f"data: [SOURCES]{_json.dumps({'sources': cached_entry.sources})}\n\n"
            yield "data: [DONE]\n\n"
            return

    # ── Cache miss: run full pipeline ─────────────────────────────────────────
    history = _memory.get(effective_session)

    t_classify_start = time.perf_counter()
    intent, confidence, target_doc = classify_intent(question, history)

    yield f"data: [ROUTE]{intent.value}\n\n"

    ctx = RouteContext(
        question=question,
        user=user,
        history=history,
        query_engine=query_engine,
        target_doc=target_doc,
    )

    answer  = ""
    sources: list[dict] = []

    if intent == Intent.SMALL_TALK:
        from openai import OpenAI as _OAI
        from retrieval.handlers import _SMALL_TALK_SYSTEM, _history_to_messages

        client = _OAI(api_key=settings.OPENAI_API_KEY)
        messages = [{"role": "system", "content": _SMALL_TALK_SYSTEM}]
        messages += _history_to_messages(history)
        messages.append({"role": "user", "content": question})

        stream = client.chat.completions.create(
            model=settings.ROUTER_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                answer += delta
                yield f"data: {delta}\n\n"

    elif intent == Intent.SUMMARIZATION:
        result = SummarizationHandler().handle(ctx)
        answer  = result.answer
        sources = result.sources
        yield f"data: {answer}\n\n"

    else:
        if query_engine is None:
            answer = "I'm unable to retrieve documents right now."
            yield f"data: {answer}\n\n"
        else:
            streaming_response = query_engine.query(question)
            for token in streaming_response.response_gen:
                answer += token
                yield f"data: {token}\n\n"

            seen: set[str] = set()
            for node in getattr(streaming_response, "source_nodes", []):
                meta = node.node.metadata
                src  = meta.get("source", "unknown")
                if src not in seen:
                    seen.add(src)
                    sources.append({
                        "source":       src,
                        "department":   meta.get("department", "general"),
                        "allowed_roles": meta.get("allowed_roles", []),
                        "text_snippet": node.node.text[:300].replace("\n", " "),
                        "score":        round(node.score or 0.0, 4),
                    })

    # Update conversation memory
    _memory.append(
        session_id=effective_session,
        user_message=question,
        assistant_message=answer[:500],
    )

    # Write to cache (never small_talk)
    if cache and intent != Intent.SMALL_TALK and answer:
        cache.set(
            question=question,
            answer=answer,
            sources=sources,
            route_type=intent.value,
            role=user.role,
            dept_filter=department_filter,
        )

    update_current_trace(
        route_type=intent.value,
        cache_hit=False,
        intent=intent.value,
        n_sources=len(sources),
        answer_chars=len(answer),
        tags=[intent.value, "streaming"],
    )

    yield f"data: [SOURCES]{_json.dumps({'sources': sources})}\n\n"
    yield "data: [DONE]\n\n"
