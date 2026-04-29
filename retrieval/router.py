"""
retrieval/router.py
────────────────────
The agentic router — sits at the very front of the pipeline and decides
which handler processes each incoming query.

Architecture
────────────
Every request flows through route_query():

  User query
      │
      ▼
  classify_intent()          ← one gpt-4o-mini call, structured JSON output
      │
      ├──► SMALL_TALK   ──► SmallTalkHandler     (no retrieval, ~50ms)
      │
      ├──► SUMMARIZATION ──► SummarizationHandler (Qdrant scroll + LLM, ~500ms)
      │
      └──► DEEP_RAG      ──► DeepRagHandler       (full pipeline, ~1500ms)
                                    │
                              (RBAC pre-filter)
                              Hybrid search
                              AutoMerging
                              Cohere Rerank
                              GPT-4o

Conversation memory
────────────────────
The router maintains an in-process LRU conversation store keyed by session_id.
Each session holds the last CONVERSATION_MEMORY_TURNS (user, assistant) pairs.

  - SmallTalk and Summarization receive full history (they are conversational).
  - DeepRAG receives NO history — retrieval is stateless by design, and injecting
    previous turns into the vector query would degrade retrieval precision.

The session_id comes from the JWT (username) as a fallback if the API client
doesn't supply one explicitly, so each user gets a stable per-user history.

Why a single classifier LLM call instead of keyword heuristics?
────────────────────────────────────────────────────────────────
Keyword rules break quickly:
  "summarize how the expense policy works" → should be DEEP_RAG (asking how,
  not asking for a full doc summary)
  "hi, what is the expense policy?" → SMALL_TALK despite mentioning a policy name

A cheap LLM classifier (gpt-4o-mini, ~50ms, ~$0.00002/call) handles these
ambiguities correctly and can use conversation history to resolve pronouns.
The structured JSON output with a `target_doc` field means the summarisation
handler gets the resolved filename without a second LLM call.

Interview talking point
────────────────────────
"The router adds ~100ms of latency but saves 1-2 seconds on 30-40% of queries.
In user-facing chat applications, response time under 200ms feels instant;
1500ms feels slow. For small-talk queries the full RAG pipeline was pure waste —
the vector search would return irrelevant chunks and the LLM would ignore them
anyway. The structured intent classification also extracts the target document
name in the same call, so summarisation doesn't need a second round-trip."
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
from config import settings
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

def classify_intent(
    question: str,
    history: list[ConversationTurn],
) -> tuple[Intent, float, Optional[str]]:
    """
    Classify the intent of `question` using a single gpt-4o-mini call.

    Returns:
        (intent, confidence, target_doc)
        target_doc is non-empty only for SUMMARIZATION intent.
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # Format history for the classifier
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
            temperature=0.0,    # deterministic classification
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        intent_str = data.get("intent", "deep_rag")
        confidence = float(data.get("confidence", 0.9))
        target_doc = data.get("target_doc", "") or ""
        reasoning  = data.get("reasoning", "")

        intent = Intent(intent_str)

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
        return Intent.DEEP_RAG, 0.5, None

    except Exception as e:
        logger.error(f"Intent classification failed ({e}) — defaulting to DEEP_RAG")
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
        """Return the conversation history for a session (empty list if new)."""
        with self._lock:
            return list(self._store.get(session_id, []))

    def append(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Append a (user, assistant) turn to the session.
        Evicts the oldest session if the store is at capacity.
        Trims old turns beyond max_turns per session.
        """
        with self._lock:
            if session_id not in self._store:
                # Evict oldest session if at capacity
                if len(self._store) >= self._max_sessions:
                    self._store.popitem(last=False)
                self._store[session_id] = []
            else:
                # Move to end (LRU — most recently used last)
                self._store.move_to_end(session_id)

            turns = self._store[session_id]
            turns.append(ConversationTurn(role="user",      content=user_message))
            turns.append(ConversationTurn(role="assistant", content=assistant_message))

            # Keep only the last max_turns × 2 messages (each turn = user + assistant)
            if len(turns) > self._max_turns * 2:
                self._store[session_id] = turns[-(self._max_turns * 2):]

    def clear(self, session_id: str) -> None:
        """Clear all history for a session."""
        with self._lock:
            self._store.pop(session_id, None)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton — shared across all requests in this process
_memory = ConversationMemory()


def get_memory() -> ConversationMemory:
    """Return the module-level conversation memory singleton."""
    return _memory


# ── Main dispatch function ────────────────────────────────────────────────────

def route_query(
    question: str,
    user: CurrentUser,
    session_id: Optional[str] = None,
    query_engine=None,   # RetrieverQueryEngine from api/main.py
) -> RouteResult:
    """
    Classify the query intent and dispatch to the correct handler.

    This is the single entry point called by api/main.py for every query.
    It replaces the old `_build_engine_for(user, ...)` + `engine.query(question)`.

    Args:
        question:     The user's raw query string.
        user:         Authenticated CurrentUser from JWT.
        session_id:   Conversation session key (defaults to username).
        query_engine: Pre-built DeepRAG engine (built in api/main.py with RBAC
                      filter already applied). Required for DEEP_RAG route.

    Returns:
        RouteResult with answer, sources, route_type, and latency_ms.
    """
    effective_session = session_id or user.username
    history = _memory.get(effective_session)

    t_classify_start = time.perf_counter()
    intent, confidence, target_doc = classify_intent(question, history)
    classify_ms = (time.perf_counter() - t_classify_start) * 1000

    logger.info(
        f"Router: intent={intent.value!r} confidence={confidence:.2f} "
        f"user={user.username!r} session={effective_session!r} "
        f"classify_ms={classify_ms:.0f}"
    )

    # ── Build RouteContext ────────────────────────────────────────────────────
    ctx = RouteContext(
        question=question,
        user=user,
        history=history,
        query_engine=query_engine,
        target_doc=target_doc,
    )

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if intent == Intent.SMALL_TALK:
        result = SmallTalkHandler().handle(ctx)

    elif intent == Intent.SUMMARIZATION:
        result = SummarizationHandler().handle(ctx)

    else:  # DEEP_RAG (default / fallback)
        if query_engine is None:
            logger.error("DeepRAG route selected but no query_engine provided — fallback to SmallTalk")
            result = SmallTalkHandler().handle(ctx)
        else:
            result = DeepRagHandler().handle(ctx)

    # ── Update conversation memory ────────────────────────────────────────────
    # We store every turn so follow-up questions have context.
    # Note: DEEP_RAG answers include source citations — strip them for cleaner history.
    answer_for_memory = result.answer
    if result.route_type == "deep_rag" and result.sources:
        # Truncate long RAG answers in memory to save space
        answer_for_memory = result.answer[:500] + ("…" if len(result.answer) > 500 else "")

    _memory.append(
        session_id=effective_session,
        user_message=question,
        assistant_message=answer_for_memory,
    )

    logger.info(
        f"Router complete: route={result.route_type!r} "
        f"total_ms={result.latency_ms + classify_ms:.0f} "
        f"(classify={classify_ms:.0f} + handle={result.latency_ms:.0f})"
    )

    # Attach classify overhead to total for observability
    result.latency_ms = round(result.latency_ms + classify_ms, 1)

    return result


# ── Streaming variant ─────────────────────────────────────────────────────────

async def route_query_stream(
    question: str,
    user: CurrentUser,
    session_id: Optional[str] = None,
    query_engine=None,
):
    """
    Async generator version of route_query for the SSE streaming endpoint.

    Yields string tokens for small_talk and deep_rag, then a final
    [ROUTE_TYPE] event and a [SOURCES] event.

    For summarization, the LLM doesn't support token-level streaming cleanly
    over the concatenated document, so we yield the full answer in one chunk.
    """
    effective_session = session_id or user.username
    history = _memory.get(effective_session)

    t_classify_start = time.perf_counter()
    intent, confidence, target_doc = classify_intent(question, history)
    classify_ms = (time.perf_counter() - t_classify_start) * 1000

    # Immediately yield the route type so the UI can show a routing indicator
    import json as _json
    yield f"data: [ROUTE]{intent.value}\n\n"

    ctx = RouteContext(
        question=question,
        user=user,
        history=history,
        query_engine=query_engine,
        target_doc=target_doc,
    )

    answer = ""
    sources: list[dict] = []

    if intent == Intent.SMALL_TALK:
        # Stream token-by-token from OpenAI
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
        # Non-streaming for summarization (fetching chunks is synchronous)
        result = SummarizationHandler().handle(ctx)
        answer = result.answer
        sources = result.sources
        yield f"data: {answer}\n\n"

    else:
        # DeepRAG — LlamaIndex streaming
        if query_engine is None:
            answer = "I'm unable to retrieve documents right now."
            yield f"data: {answer}\n\n"
        else:
            streaming_response = query_engine.query(question)
            for token in streaming_response.response_gen:
                answer += token
                yield f"data: {token}\n\n"

            # Extract sources
            seen: set[str] = set()
            for node in getattr(streaming_response, "source_nodes", []):
                meta = node.node.metadata
                src = meta.get("source", "unknown")
                if src not in seen:
                    seen.add(src)
                    sources.append({
                        "source": src,
                        "department": meta.get("department", "general"),
                        "allowed_roles": meta.get("allowed_roles", []),
                        "text_snippet": node.node.text[:300].replace("\n", " "),
                        "score": round(node.score or 0.0, 4),
                    })

    # Update memory
    _memory.append(
        session_id=effective_session,
        user_message=question,
        assistant_message=answer[:500],
    )

    # Final SSE events
    yield f"data: [SOURCES]{_json.dumps({'sources': sources})}\n\n"
    yield "data: [DONE]\n\n"
