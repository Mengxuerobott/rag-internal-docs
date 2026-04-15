"""
auth/rbac.py
────────────
Role-Based Access Control definitions.

Two things live here:

1. ROLE_HIERARCHY — which roles can see which other roles' content.
   A "management" user can see both "management" AND "hr" docs because
   management is a superset role.

2. FOLDER_PERMISSIONS — maps document folder names (departments) to the
   minimum set of roles that may read those chunks.
   This is consulted at INGESTION time to stamp `allowed_roles` onto every
   chunk's Qdrant payload.

Design decisions worth discussing in an interview:
──────────────────────────────────────────────────
• We embed permissions at chunk level rather than document level so a single
  document can contain sections with different ACLs (e.g. a board memo that
  has a public summary and a private appendix in separate pages).

• allowed_roles is stored as an array field in Qdrant's payload.  Qdrant's
  `MatchAny` filter can do "does this array contain X?" in O(1) with a
  payload index — no post-retrieval filtering needed.

• Role hierarchy is resolved at query time (expand_roles), not stored in the
  chunk, so changing the hierarchy doesn't require re-ingestion.

• The `employee` role is the baseline — all authenticated users are at least
  employees.  Unauthenticated requests are rejected by the API before they
  reach the retrieval layer.
"""

from __future__ import annotations

# ── Role registry ─────────────────────────────────────────────────────────────
# All valid role names in the system.
ALL_ROLES: set[str] = {
    "employee",      # every authenticated user; sees general / public docs
    "engineering",   # engineers; sees engineering runbooks, ADRs
    "hr",            # HR team; sees leave policy, salary bands, headcount
    "finance",       # finance team; sees budgets, expense limits, forecasts
    "legal",         # legal team; sees contracts, compliance, data retention
    "management",    # director+; supersedes hr + finance + general
    "admin",         # IT/ops admin; sees everything
}

# ── Role hierarchy ────────────────────────────────────────────────────────────
# Maps role → set of roles whose content this role may also access.
# "admin" can see all roles; "management" can see hr, finance, general, etc.
#
# expand_roles() uses this to turn a single JWT role like "management" into
# the full set of allowed_roles that the Qdrant filter should match against.
ROLE_HIERARCHY: dict[str, set[str]] = {
    "employee":    {"employee"},
    "engineering": {"employee", "engineering"},
    "hr":          {"employee", "hr"},
    "finance":     {"employee", "finance"},
    "legal":       {"employee", "legal"},
    "management":  {"employee", "hr", "finance", "management"},
    "admin":       ALL_ROLES,                    # sees everything
}


def expand_roles(role: str) -> list[str]:
    """
    Return the full list of allowed_roles values this role can query.

    Example:
        expand_roles("management") → ["employee", "hr", "finance", "management"]
        expand_roles("engineering") → ["employee", "engineering"]
        expand_roles("admin")      → [all roles]
    """
    return list(ROLE_HIERARCHY.get(role, {"employee"}))


# ── Folder → allowed_roles mapping ───────────────────────────────────────────
# Consulted by ingestion/loader.py when stamping chunks.
# Key   = subdirectory name under data/sample_docs/
# Value = list of roles stored in chunk payload as `allowed_roles`
#
# A chunk is visible to a user if the intersection of:
#   expand_roles(user.role)  AND  chunk.allowed_roles
# is non-empty.
FOLDER_PERMISSIONS: dict[str, list[str]] = {
    "general":     ["employee"],                          # everyone
    "engineering": ["employee", "engineering"],           # engineers + employee-level
    "hr":          ["hr", "management", "admin"],         # HR-restricted
    "finance":     ["finance", "management", "admin"],    # finance-restricted
    "legal":       ["legal", "management", "admin"],      # legal-restricted
    "executive":   ["management", "admin"],               # board / exec only
}

# Fallback for documents not in any recognised folder
DEFAULT_ALLOWED_ROLES: list[str] = ["employee"]


def get_allowed_roles_for_path(folder_name: str) -> list[str]:
    """
    Return the allowed_roles list for a given folder name.
    Falls back to DEFAULT_ALLOWED_ROLES if the folder isn't mapped.

    Args:
        folder_name: The immediate parent directory of the document
                     (e.g. "hr", "finance", "general").
    """
    return FOLDER_PERMISSIONS.get(folder_name.lower(), DEFAULT_ALLOWED_ROLES)


# ── Access check ──────────────────────────────────────────────────────────────
def can_access(user_role: str, chunk_allowed_roles: list[str]) -> bool:
    """
    Return True if a user with `user_role` may read a chunk with
    `chunk_allowed_roles`.

    Used in unit tests and as a debug helper; the actual enforcement at
    query time is done via Qdrant's MatchAny pre-filter, not this function.
    """
    user_expanded = set(expand_roles(user_role))
    chunk_set = set(chunk_allowed_roles)
    return bool(user_expanded & chunk_set)
