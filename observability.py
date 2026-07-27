"""
observability.py
────────────────
LLM tracing for the RAG pipeline, backed by Langfuse.

Why Langfuse and not LangSmith
───────────────────────────────
LangSmith's automatic tracing hooks LangChain runnables. This codebase is
LlamaIndex-only, so LANGCHAIN_TRACING_V2 captures nothing — the LANGCHAIN_*
settings in config.py have always been dead configuration.

Langfuse plugs into LlamaIndex's CallbackManager through
`set_global_handler("langfuse")`, which llama-index-core 0.10.x supports
natively. One call at startup instruments the entire retrieval pipeline:

    retrieve          → hybrid search (dense + SPLADE sparse, RRF-fused in
                        Qdrant), with the RBAC pre-filter that was applied
    (sub-span)        → AutoMergingRetriever leaf→parent swaps
    node_postprocess  → SimilarityPostprocessor + CohereRerank, including the
                        node set and scores before/after reranking
    synthesize        → the assembled prompt
    llm               → model, full prompt/completion, and token usage

Crucially, this covers *both* engine assembly paths — the one in
retrieval/query_engine.py and the duplicated block in api/main.py — because
instrumentation attaches to LlamaIndex itself, not to our call sites.

What it does NOT cover
───────────────────────
Anything that bypasses LlamaIndex. Those need the @traced decorator below:

  - retrieval/router.py::classify_intent      (raw openai SDK)
  - retrieval/handlers.py::SmallTalkHandler   (raw openai SDK)
  - retrieval/handlers.py::SummarizationHandler
  - retrieval/handlers.py::_fetch_document_chunks (raw Qdrant scroll)

Failure policy
───────────────
Observability must never take down the API. Every entry point here swallows
its exceptions and degrades to a no-op, so a bad key or an unreachable
Langfuse host costs you traces, never requests.

Disabled by default
────────────────────
With LANGFUSE_ENABLED unset or false (or either key blank), init_tracing() is
a no-op and @traced returns the undecorated function, so the request path is
exactly what it was before tracing existed. Set LANGFUSE_ENABLED=true in .env
once you have keys.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

from config import settings

F = TypeVar("F", bound=Callable[..., Any])


# ── Module state ──────────────────────────────────────────────────────────────
# _active is the single source of truth for "is tracing live right now".
# It only flips to True after set_global_handler() has actually succeeded, so
# a configured-but-broken Langfuse never makes @traced try to emit spans.
_active: bool = False


def is_tracing_active() -> bool:
    """True only if init_tracing() ran and wired up the handler successfully."""
    return _active


def _tracing_configured() -> bool:
    """Config-level check: is tracing switched on and fully credentialed?"""
    if not settings.LANGFUSE_ENABLED:
        return False
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        logger.warning(
            "LANGFUSE_ENABLED=true but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            "is blank — tracing stays disabled."
        )
        return False
    return True


# ── Lifecycle ─────────────────────────────────────────────────────────────────
def init_tracing() -> bool:
    """
    Install the Langfuse global handler on LlamaIndex.

    Call once during FastAPI lifespan startup, before any query is served.
    Idempotent, and safe to call when tracing is disabled or misconfigured.

    Returns True if tracing is now live, False otherwise. Never raises.
    """
    global _active

    if _active:
        return True

    if not _tracing_configured():
        logger.info("LLM tracing disabled (set LANGFUSE_ENABLED=true to enable).")
        return False

    try:
        # Langfuse reads credentials from the environment, so mirror the
        # config values across before installing the handler. We set them
        # rather than requiring the user to duplicate them in .env.
        import os

        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST

        from llama_index.core import set_global_handler

        set_global_handler("langfuse")

        _active = True
        logger.info(f"LLM tracing enabled — Langfuse at {settings.LANGFUSE_HOST}")
        return True

    except Exception as e:
        # Wrong key, unreachable host, version drift — none of it is worth
        # failing startup over. Log loudly and serve requests untraced.
        _active = False
        logger.warning(f"Failed to initialise Langfuse tracing ({e}) — continuing untraced.")
        return False


def shutdown_tracing() -> None:
    """
    Flush buffered spans before the process exits.

    Langfuse batches spans in a background thread. Without an explicit flush,
    the tail of a run — often the most interesting part when debugging a
    crash — is lost. Especially relevant for SSE requests, whose spans close
    late. Call from FastAPI lifespan shutdown. Never raises.
    """
    if not _active:
        return

    try:
        from langfuse import Langfuse

        Langfuse().flush()
        logger.info("Langfuse spans flushed.")
    except Exception as e:
        logger.warning(f"Langfuse flush failed ({e}) — some spans may be lost.")


# ── Manual span decorator ─────────────────────────────────────────────────────
def traced(
    name: Optional[str] = None,
    as_type: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Wrap a function in a Langfuse span.

    For the parts of the pipeline LlamaIndex doesn't own — the raw `openai`
    SDK calls in the router and small-talk handler, and the raw Qdrant scroll
    in the summarization handler.

    The decision to trace is made once, at import time: when tracing is off,
    this returns the original function unchanged, so there is zero call-time
    overhead and zero behaviour change on the disabled path.

    Args:
        name:    Span name shown in the Langfuse UI. Defaults to __name__.
        as_type: Pass "generation" for spans that wrap an LLM call, so Langfuse
                 renders them with model/token/cost detail instead of as a
                 plain span.

    Usage:
        @traced(name="classify_intent", as_type="generation")
        def classify_intent(...): ...
    """

    def decorator(fn: F) -> F:
        if not _tracing_configured():
            return fn

        try:
            from langfuse.decorators import observe

            return observe(name=name or fn.__name__, as_type=as_type)(fn)
        except Exception as e:
            logger.warning(
                f"Could not attach tracing to {fn.__qualname__} ({e}) — leaving it untraced."
            )
            return fn

    return decorator


# Keyword arguments the Langfuse context API accepts natively. Anything else
# a caller passes is folded into `metadata` instead of being dropped — without
# this, a stray kwarg makes the whole update call raise and the attribute is
# silently lost.
_TRACE_KWARGS = frozenset({
    "name", "input", "output", "user_id", "session_id",
    "version", "release", "metadata", "tags", "public",
})

_OBSERVATION_KWARGS = frozenset({
    "input", "output", "name", "version", "metadata", "start_time", "end_time",
    "release", "tags", "user_id", "session_id", "level", "status_message",
    "completion_start_time", "model", "model_parameters", "usage",
    "usage_details", "cost_details", "prompt", "public",
})


def _split_kwargs(attributes: dict, allowed: frozenset) -> dict:
    """Route unrecognised keys into `metadata`, merging with any passed explicitly."""
    native = {k: v for k, v in attributes.items() if k in allowed}
    extra = {k: v for k, v in attributes.items() if k not in allowed}

    if extra:
        native["metadata"] = {**(native.get("metadata") or {}), **extra}

    return native


def update_current_trace(**attributes: Any) -> None:
    """
    Attach key/value metadata to the trace currently in scope.

    Used to hang request-level context — user_role, route_type, cache_hit, the
    RBAC-expanded role list — off the root span so a trace can be read (and
    filtered) without digging into individual child spans.

    `user_id`, `session_id` and `tags` map onto Langfuse's first-class trace
    fields; any other keyword lands in the trace's metadata.

    Silently does nothing when tracing is off or no trace is in scope.
    """
    if not _active:
        return

    try:
        from langfuse.decorators import langfuse_context

        langfuse_context.update_current_trace(**_split_kwargs(attributes, _TRACE_KWARGS))
    except Exception as e:
        logger.debug(f"update_current_trace failed ({e}) — ignored.")


def update_current_observation(**attributes: Any) -> None:
    """
    Attach metadata to the current span (as opposed to the whole trace).

    Use for step-local detail: the classified intent and its confidence, the
    number of chunks a Qdrant scroll returned, and so on. Pass `usage` and
    `model` on generation spans so Langfuse can render token counts and cost.

    Unrecognised keywords land in the span's metadata.

    Silently does nothing when tracing is off or no span is in scope.
    """
    if not _active:
        return

    try:
        from langfuse.decorators import langfuse_context

        langfuse_context.update_current_observation(
            **_split_kwargs(attributes, _OBSERVATION_KWARGS)
        )
    except Exception as e:
        logger.debug(f"update_current_observation failed ({e}) — ignored.")
