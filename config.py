"""
config.py
Central configuration — loaded once at startup, imported everywhere.
All values come from environment variables (set via .env or docker-compose).
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# Smallest leaf chunk we allow. LlamaIndex refuses to build a node whose
# metadata is longer than its chunk size, and this project stamps source,
# department, allowed_roles and ingestion timestamp onto every node — around
# 76 characters in practice. 128 leaves comfortable headroom; below roughly
# 80 the parser raises "Metadata length is longer than chunk size" partway
# through ingestion, which is a confusing place to discover a config typo.
MIN_LEAF_CHUNK_SIZE = 128

# The placeholder JWT signing key shipped in .env.example. It is committed to a
# public repository, so anything running with it will accept tokens forged by
# anyone who has read the repo — including tokens claiming the "admin" role,
# which bypasses RBAC entirely. Startup refuses it unless explicitly allowed.
DEFAULT_JWT_SECRET = "change-me-in-production-use-secrets-token-hex-32"

# A signing key shorter than this is brute-forceable. `secrets.token_hex(32)`,
# the command this project documents, produces 64 characters.
MIN_JWT_SECRET_LENGTH = 32

# Substrings that mark a value as an unfilled template rather than a real
# secret. Checking only for DEFAULT_JWT_SECRET was not enough: the deployed
# task definition carried "REPLACE_JWT_SECRET", which is not the .env.example
# default and so passed silently while being trivially guessable. Placeholders
# come from many templates, so match the shape, not one literal.
_PLACEHOLDER_MARKERS = (
    "replace",
    "change-me",
    "changeme",
    "placeholder",
    "your-secret",
    "your_secret",
    "example",
    "todo",
    "xxx",
)

# The demo accounts in auth/jwt_handler.py all share one password, which used
# to be hardcoded as "secret". A strong JWT_SECRET stops forged tokens but does
# nothing about someone simply signing in as `admin` on a reachable deployment.
DEFAULT_DEMO_USER_PASSWORD = "secret"

# Lower bar than a signing key: this is a password a person types, not a key.
MIN_DEMO_PASSWORD_LENGTH = 12

# Exact values too commonly guessed to accept even if they clear the length bar.
_WEAK_PASSWORDS = frozenset({
    "secret", "password", "admin", "letmein", "welcome",
    "123456", "12345678", "qwerty", "changeme", "default",
})


def _parse_chunk_sizes(raw: str) -> list[int]:
    """
    Parse and validate the CHUNK_SIZES setting.

    Expects a comma-separated, strictly descending list of at least two
    positive integers, e.g. "2048,512,128" → [2048, 512, 128], representing
    parent → child → leaf sizes for the hierarchical node parser.

    Raises ValueError with an actionable message rather than letting a bad
    value fail deep inside ingestion.
    """
    try:
        sizes = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(
            f"CHUNK_SIZES must be comma-separated integers (got {raw!r}): {e}"
        ) from e

    if len(sizes) < 2:
        raise ValueError(
            f"CHUNK_SIZES needs at least two values (parent and leaf), got {sizes!r}. "
            f"The default is 2048,512,128."
        )

    if any(s <= 0 for s in sizes):
        raise ValueError(f"CHUNK_SIZES values must all be positive, got {sizes!r}.")

    if sizes != sorted(sizes, reverse=True):
        raise ValueError(
            f"CHUNK_SIZES must be strictly descending (parent → child → leaf), "
            f"got {sizes!r}. The default is 2048,512,128."
        )

    if len(set(sizes)) != len(sizes):
        raise ValueError(
            f"CHUNK_SIZES must not repeat a size, got {sizes!r} — each level of "
            f"the hierarchy needs to be smaller than the one above it."
        )

    if sizes[-1] < MIN_LEAF_CHUNK_SIZE:
        raise ValueError(
            f"CHUNK_SIZES leaf size {sizes[-1]} is below the minimum "
            f"{MIN_LEAF_CHUNK_SIZE}. Node metadata (source, department, "
            f"allowed_roles, timestamp) is longer than that, and the parser "
            f"rejects a chunk smaller than its own metadata."
        )

    return sizes


class Settings:
    # ── LLM / Embeddings ─────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # ── Document parsing ─────────────────────────────────────────────────────
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

    # ── Reranker ─────────────────────────────────────────────────────────────
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

    # ── Vector store ─────────────────────────────────────────────────────────
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "company_docs")

    # ── Observability ─────────────────────────────────────────────────────────
    # Langfuse traces the full LlamaIndex pipeline automatically via
    # set_global_handler("langfuse") — see observability.py.
    #
    # Master switch. When false (the default) tracing is a no-op and the
    # request path is byte-for-byte what it was before tracing was added.
    # Tracing also stays off if either key below is blank.
    LANGFUSE_ENABLED: bool = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"

    # Credentials from your Langfuse project settings. Keep these in .env only.
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")

    # Langfuse Cloud EU (default), US (https://us.cloud.langfuse.com),
    # or your own self-hosted URL.
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # DEPRECATED — LangSmith is not wired into this project and never was.
    # Its automatic tracing instruments LangChain runnables; this codebase is
    # LlamaIndex-only, so LANGCHAIN_TRACING_V2 captures nothing. Kept so that
    # existing .env files don't break. Use the LANGFUSE_* settings instead.
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "rag-internal-docs")

    # ── Retrieval tuning ──────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    TOP_N_RERANK: int = int(os.getenv("TOP_N_RERANK", "3"))
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Parsed as list of ints: "2048,512,128" → [2048, 512, 128]
    # Validated at import so a bad value fails at startup with a clear message,
    # rather than surfacing much later as an opaque parser error partway
    # through an ingestion run.
    CHUNK_SIZES: list[int] = _parse_chunk_sizes(
        os.getenv("CHUNK_SIZES", "2048,512,128")
    )

    # ── Multimodal processing ─────────────────────────────────────────────────
    # Set to "false" to skip table summarisation and image description entirely.
    ENABLE_MULTIMODAL: bool = os.getenv("ENABLE_MULTIMODAL", "true").lower() == "true"

    # Model for vision (image description) and table summarisation.
    # gpt-4o-mini is ~10x cheaper than gpt-4o and adequate for most use cases.
    VLM_MODEL: str = os.getenv("VLM_MODEL", "gpt-4o-mini")

    # Max chars of surrounding paragraph sent to the VLM as document context.
    MULTIMODAL_CONTEXT_WINDOW: int = int(os.getenv("MULTIMODAL_CONTEXT_WINDOW", "500"))

    # Max base64 image size in MB. Images larger than this are skipped.
    MULTIMODAL_MAX_IMAGE_MB: float = float(os.getenv("MULTIMODAL_MAX_IMAGE_MB", "4.0"))

    # ── Auth / JWT ────────────────────────────────────────────────────────────
    # IMPORTANT: set a strong random secret in production.
    # Generate one with:  python -c "import secrets; print(secrets.token_hex(32))"
    JWT_SECRET: str = os.getenv("JWT_SECRET", DEFAULT_JWT_SECRET)

    # Shared password for the demo accounts in auth/jwt_handler.py. Must be set
    # to something real for any deployment that is reachable, or `admin` can
    # simply be logged into.
    DEMO_USER_PASSWORD: str = os.getenv(
        "DEMO_USER_PASSWORD", DEFAULT_DEMO_USER_PASSWORD
    )

    # Escape hatch for local development and CI, where placeholder credentials
    # are harmless. Covers the signing key and the demo password together —
    # they answer the same question ("is this a real environment?"), so they
    # share one switch. ALLOW_INSECURE_JWT_SECRET is honoured as a deprecated
    # alias so existing environments keep working.
    ALLOW_INSECURE_AUTH: bool = (
        os.getenv(
            "ALLOW_INSECURE_AUTH",
            os.getenv("ALLOW_INSECURE_JWT_SECRET", "false"),
        ).lower() == "true"
    )
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))  # 8 hours

    # ── Event-driven ingestion (Redis + ARQ) ──────────────────────────────────
    # Redis is the job queue broker used by ARQ workers.
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Maximum number of concurrent ingestion jobs a single worker processes.
    WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "4"))

    # Job retry settings
    WORKER_MAX_RETRIES: int = int(os.getenv("WORKER_MAX_RETRIES", "3"))
    WORKER_RETRY_DELAY_S: int = int(os.getenv("WORKER_RETRY_DELAY_S", "30"))

    # ── Webhook HMAC secrets ──────────────────────────────────────────────────
    # Each DMS provider signs its webhook payloads with one of these secrets.
    # Set them to the values configured in each provider's webhook settings UI.
    # Leave blank to SKIP signature verification (only for local dev/testing).
    WEBHOOK_SECRET_CONFLUENCE: str = os.getenv("WEBHOOK_SECRET_CONFLUENCE", "")
    WEBHOOK_SECRET_SHAREPOINT: str = os.getenv("WEBHOOK_SECRET_SHAREPOINT", "")
    WEBHOOK_SECRET_GDRIVE: str = os.getenv("WEBHOOK_SECRET_GDRIVE", "")

    # ── Agentic router ───────────────────────────────────────────────────────
    # Model used for the intent classification call (the router).
    # This must be fast and cheap — gpt-4o-mini is ideal.
    # The router call adds ~100-200ms but saves 1-3s on non-RAG queries.
    ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "gpt-4o-mini")

    # How many previous turns (user+assistant pairs) to include in the
    # conversation context window sent to small-talk and summarisation routes.
    # Deep-RAG route does NOT use conversation history (stateless retrieval).
    CONVERSATION_MEMORY_TURNS: int = int(os.getenv("CONVERSATION_MEMORY_TURNS", "6"))

    # Maximum number of concurrent conversation sessions kept in memory.
    # Oldest sessions are evicted when this limit is reached.
    CONVERSATION_MAX_SESSIONS: int = int(os.getenv("CONVERSATION_MAX_SESSIONS", "1000"))

    # ── Semantic cache ───────────────────────────────────────────────────────
    # Set to "false" to disable the cache entirely (useful during development
    # when you want every request to go through the full pipeline).
    SEMANTIC_CACHE_ENABLED: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"

    # Cosine similarity threshold for a cache hit.
    # 0.92 is a good default: catches paraphrases ("company holidays 2026" ≈
    # "2026 public holidays") but rejects topically different questions.
    # Raise toward 0.99 for exact-only matching; lower toward 0.85 for more
    # aggressive caching (risk of serving a subtly wrong cached answer).
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = float(
        os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.92")
    )

    # How long a cache entry lives in Redis before automatic expiry.
    # Default: 24 hours.  Set to 0 to disable TTL (entries live forever until
    # manually evicted or Redis is flushed).
    SEMANTIC_CACHE_TTL_SECONDS: int = int(
        os.getenv("SEMANTIC_CACHE_TTL_SECONDS", str(24 * 3600))
    )

    # Maximum number of entries per (role, department_filter) namespace.
    # When this limit is reached, the oldest entry is evicted (LRU-style).
    # Default: 10 000 — at 1536 floats × 4 bytes each = ~60 KB per entry,
    # 10 000 entries = ~600 MB RAM.  Reduce if memory is constrained.
    SEMANTIC_CACHE_MAX_ENTRIES: int = int(
        os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "10000")
    )

    # Whether to flush the entire cache after every document ingestion event.
    # Ensures stale answers are never served after a policy document is updated.
    # Small performance cost (cache cold-starts after every ingest).
    INVALIDATE_CACHE_ON_INGEST: bool = (
        os.getenv("INVALIDATE_CACHE_ON_INGEST", "false").lower() == "true"
    )

    # ── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # ── Directories ───────────────────────────────────────────────────────────
    DOCS_DIR: str = os.getenv("DOCS_DIR", "data/sample_docs")
    INDEX_PERSIST_DIR: str = os.getenv("INDEX_PERSIST_DIR", "data/index_store")


def _validate_security_settings(s: Settings) -> None:
    """
    Refuse to start with a signing key that cannot be trusted.

    Without this the fallback is silent: the app boots normally and serves
    traffic while accepting tokens anyone can mint from the placeholder secret
    in the public repo. A forged "admin" token bypasses RBAC completely, and
    nothing in the logs would indicate it. Failing at startup turns a silent
    compromise into an obvious, immediate error.
    """
    problems: list[str] = []

    reason = _jwt_secret_problem(s.JWT_SECRET)
    if reason:
        problems.append(
            f"JWT_SECRET: {reason}.\n"
            "    Tokens are signed with this key, so a guessable value lets "
            "anyone forge a token for any role, including admin.\n"
            '    Generate one:  python -c "import secrets; '
            'print(secrets.token_hex(32))"'
        )

    reason = _demo_password_problem(s.DEMO_USER_PASSWORD)
    if reason:
        problems.append(
            f"DEMO_USER_PASSWORD: {reason}.\n"
            "    Every demo account shares this password, including admin, so "
            "a reachable deployment can simply be logged into.\n"
            '    Generate one:  python -c "import secrets; '
            'print(secrets.token_urlsafe(24))"'
        )

    if not problems:
        return

    listed = "\n\n".join(f"  - {p}" for p in problems)

    if s.ALLOW_INSECURE_AUTH:
        logger.warning(
            "Insecure auth settings accepted because ALLOW_INSECURE_AUTH=true "
            "— never set that outside local development or CI:\n" + listed
        )
        return

    raise ValueError(
        "Refusing to start with insecure auth settings:\n\n"
        + listed
        + "\n\nSet these in your environment (in the ECS task definition for "
        "the deployed service, ideally via secrets rather than plaintext).\n"
        "For local development or CI, where this does not matter, set "
        "ALLOW_INSECURE_AUTH=true instead."
    )


def _jwt_secret_problem(secret: str) -> str | None:
    """
    Describe why `secret` is unusable as a signing key, or None if it is fine.

    Returns a human-readable reason so the startup error can say what is
    actually wrong rather than just refusing.
    """
    if not secret:
        return "it is empty"

    if secret == DEFAULT_JWT_SECRET:
        return "it is the placeholder committed to this repository"

    marker = _placeholder_marker_in(secret)
    if marker:
        return f"it looks like an unfilled template value (contains {marker!r})"

    if len(secret) < MIN_JWT_SECRET_LENGTH:
        return (
            f"it is only {len(secret)} characters; "
            f"at least {MIN_JWT_SECRET_LENGTH} are required"
        )

    return None


def _placeholder_marker_in(value: str) -> str | None:
    """Return the template marker found in `value`, if any."""
    lowered = value.lower()
    for marker in _PLACEHOLDER_MARKERS:
        if marker in lowered:
            return marker
    return None


def _demo_password_problem(password: str) -> str | None:
    """
    Describe why `password` is unusable for the demo accounts, or None.

    Same shape as _jwt_secret_problem, with a lower length bar: this is a
    password a person types, not a signing key.
    """
    if not password:
        return "it is empty"

    if password.lower() in _WEAK_PASSWORDS:
        return f"{password!r} is among the most commonly guessed passwords"

    marker = _placeholder_marker_in(password)
    if marker:
        return f"it looks like an unfilled template value (contains {marker!r})"

    if len(password) < MIN_DEMO_PASSWORD_LENGTH:
        return (
            f"it is only {len(password)} characters; "
            f"at least {MIN_DEMO_PASSWORD_LENGTH} are required"
        )

    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    s = Settings()
    _validate_security_settings(s)
    return s


settings = get_settings()
