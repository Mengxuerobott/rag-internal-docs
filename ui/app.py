"""
ui/app.py
─────────
Streamlit chat interface for the Internal Docs RAG assistant.

Features (updated for agentic routing):
  - Login screen (JWT auth — required by all query endpoints)
  - Route type badge on every answer: 💬 small_talk | 📄 summarization | 🔍 deep_rag
  - [ROUTE] SSE event detected before first token, shown as a routing indicator
  - Streaming chat with token-by-token display
  - Source citation panel (expander below each answer)
  - Department filter sidebar
  - Conversation history controls (view / clear via API endpoints)
  - Document index browser
  - Re-ingest button (admin only)
  - Session history persisted in Streamlit session state

Run locally:
    streamlit run ui/app.py

Requires API to be running:
    uvicorn api.main:app --reload
"""

import json
import os
import time
from typing import Optional

import httpx
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
APP_TITLE = "Internal Docs Assistant"
APP_ICON = "📄"

PLACEHOLDER_QUESTIONS = [
    "What is our parental leave policy?",
    "How do I submit an expense report?",
    "What are the onboarding steps for new engineers?",
    "What is our data retention policy?",
]

ROUTE_BADGES = {
    "small_talk":    ("💬", "Conversational", "#2196F3"),
    "summarization": ("📄", "Full-doc summary", "#FF9800"),
    "deep_rag":      ("🔍", "Deep retrieval", "#4CAF50"),
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("department_filter", None),
    ("indexed_docs", []),
    ("jwt_token", None),
    ("username", None),
    ("user_role", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Auth helpers ──────────────────────────────────────────────────────────────
def login(username: str, password: str) -> Optional[str]:
    """Exchange credentials for a JWT. Returns the token or None on failure."""
    try:
        r = httpx.post(
            f"{API_BASE}/auth/token",
            data={"username": username, "password": password},
            timeout=10.0,
        )
        if r.status_code == 200:
            data = r.json()
            st.session_state.jwt_token = data["access_token"]
            st.session_state.user_role = data.get("role", "employee")
            return data["access_token"]
        return None
    except Exception:
        return None


def _auth_headers() -> dict:
    token = st.session_state.get("jwt_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def is_logged_in() -> bool:
    return bool(st.session_state.get("jwt_token"))


# ── Login screen ──────────────────────────────────────────────────────────────
if not is_logged_in():
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Sign in with your company account to access internal documents.")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="e.g. alice, bob, admin")
        password = st.text_input("Password", type="password", placeholder="secret")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        with st.spinner("Signing in…"):
            token = login(username, password)
        if token:
            st.session_state.username = username
            st.success(f"Welcome, {username}! Role: **{st.session_state.user_role}**")
            st.rerun()
        else:
            st.error("Invalid credentials. Demo accounts: alice/bob/carol/dave/eve/frank/admin (password: secret)")

    with st.expander("Demo accounts"):
        st.markdown("""
| Username | Role | Can access |
|----------|------|------------|
| alice | hr | HR docs |
| bob | engineering | Engineering docs |
| carol | finance | Finance docs |
| frank | legal | Legal docs |
| dave | management | HR + Finance + Management |
| eve | employee | General docs only |
| admin | admin | Everything |

All passwords: `secret`
        """)
    st.stop()


# ── API helpers (require auth) ────────────────────────────────────────────────
def stream_query(question: str, department_filter: Optional[str] = None):
    """
    Call POST /query/stream and yield (event_type, data) tuples.

    Event types yielded:
        ("route",   route_type_string)     — first event, before any tokens
        ("token",   token_string)          — answer tokens
        ("sources", list_of_source_dicts)  — final event
        ("error",   error_message)         — on failure
    """
    payload = {
        "question": question,
        "session_id": st.session_state.get("username", "streamlit"),
        "department_filter": department_filter,
    }

    try:
        with httpx.Client(timeout=90.0) as client:
            with client.stream(
                "POST",
                f"{API_BASE}/query/stream",
                json=payload,
                headers=_auth_headers(),
            ) as response:
                if response.status_code == 401:
                    st.session_state.jwt_token = None  # force re-login
                    yield ("error", "Session expired — please sign in again.")
                    return
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]  # strip "data: "

                    if chunk == "[DONE]":
                        return

                    if chunk.startswith("[ROUTE]"):
                        yield ("route", chunk[len("[ROUTE]"):])

                    elif chunk.startswith("[SOURCES]"):
                        try:
                            sources = json.loads(chunk[len("[SOURCES]"):]).get("sources", [])
                        except json.JSONDecodeError:
                            sources = []
                        yield ("sources", sources)

                    elif chunk.startswith("[ERROR]"):
                        yield ("error", chunk[7:])

                    else:
                        yield ("token", chunk)

    except httpx.TimeoutException:
        yield ("error", "Request timed out. The document may be very large.")
    except Exception as e:
        yield ("error", str(e))


def fetch_docs_list() -> list[dict]:
    try:
        r = httpx.get(f"{API_BASE}/docs-list", headers=_auth_headers(), timeout=10.0)
        return r.json().get("documents", [])
    except Exception:
        return []


def trigger_ingest(force: bool = False) -> str:
    try:
        r = httpx.post(
            f"{API_BASE}/ingest",
            json={"force_rebuild": force},
            headers=_auth_headers(),
            timeout=10.0,
        )
        if r.status_code == 403:
            return "❌ Admin role required"
        return r.json().get("status", "unknown")
    except Exception as e:
        return f"Error: {e}"


def clear_conversation_history() -> None:
    """Clear server-side conversation memory for this session."""
    try:
        httpx.delete(f"{API_BASE}/query/history", headers=_auth_headers(), timeout=5.0)
    except Exception:
        pass


def check_api_health() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5.0)
        return r.json().get("index_ready", False)
    except Exception:
        return False


def _route_badge(route_type: str) -> str:
    icon, label, _ = ROUTE_BADGES.get(route_type, ("⚙️", route_type, "#888"))
    return f"{icon} *{label}*"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")

    # User info
    st.markdown(
        f"**Signed in as:** `{st.session_state.username}`  \n"
        f"**Role:** `{st.session_state.user_role}`"
    )
    if st.button("Sign out"):
        for key in ["jwt_token", "username", "user_role", "messages"]:
            st.session_state[key] = None if key == "jwt_token" else []
        st.rerun()

    api_ok = check_api_health()
    st.markdown(f"**API:** {'🟢 Ready' if api_ok else '🔴 Offline'}")
    st.divider()

    # Routing legend
    st.subheader("Route types")
    for rt, (icon, label, color) in ROUTE_BADGES.items():
        st.markdown(f"{icon} **{label}** — `{rt}`")
    st.caption(
        "The router classifies each query and dispatches it to the cheapest "
        "appropriate pipeline. Small talk bypasses vector search entirely."
    )
    st.divider()

    # Department filter
    st.subheader("Filter by department")
    dept_options = ["All documents", "hr", "engineering", "finance", "legal", "general"]
    selected_dept = st.selectbox("Department", dept_options, index=0)
    st.session_state.department_filter = (
        None if selected_dept == "All documents" else selected_dept
    )

    st.divider()

    # Indexed documents
    st.subheader("Indexed documents")
    if st.button("🔄 Refresh list"):
        st.session_state.indexed_docs = fetch_docs_list()

    if not st.session_state.indexed_docs:
        st.session_state.indexed_docs = fetch_docs_list()

    if st.session_state.indexed_docs:
        for doc in st.session_state.indexed_docs[:20]:
            st.markdown(f"- `{doc['source']}` · *{doc['department']}*")
        if len(st.session_state.indexed_docs) > 20:
            st.caption(f"…and {len(st.session_state.indexed_docs) - 20} more")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # Admin controls
    st.subheader("Manage index")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Re-ingest"):
            st.success(trigger_ingest(force=False))
    with col2:
        if st.button("🔁 Rebuild"):
            st.warning(trigger_ingest(force=True))

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        clear_conversation_history()
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.header(f"{APP_ICON} Ask your internal docs")

if st.session_state.department_filter:
    st.info(f"🔍 Filtering to **{st.session_state.department_filter}** documents")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Route badge (only on assistant messages that have one)
        if msg["role"] == "assistant" and msg.get("route_type"):
            st.caption(_route_badge(msg["route_type"]))
        st.markdown(msg["content"])

        # Sources expander
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} docs)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source']}** · `{src['department']}` · "
                        f"score: `{src['score']}`"
                    )
                    st.caption(f"> {src.get('text_snippet', '')}…")
                    st.divider()

# Suggested starter questions
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(PLACEHOLDER_QUESTIONS):
        if cols[i % 2].button(q, key=f"suggestion_{i}"):
            st.session_state._prefill = q
            st.rerun()

prefill = st.session_state.pop("_prefill", None)
user_input = st.chat_input("Ask about company docs...") or prefill

if user_input and api_ok:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        route_placeholder = st.empty()
        answer_placeholder = st.empty()

        accumulated = ""
        final_sources = []
        final_route = "deep_rag"
        start = time.perf_counter()

        with st.spinner("Thinking…"):
            for event_type, data in stream_query(
                user_input,
                department_filter=st.session_state.department_filter,
            ):
                if event_type == "route":
                    final_route = data
                    # Show routing badge immediately before first token
                    route_placeholder.caption(_route_badge(final_route))

                elif event_type == "token":
                    accumulated += data
                    answer_placeholder.markdown(accumulated + "▌")

                elif event_type == "sources":
                    final_sources = data

                elif event_type == "error":
                    st.error(f"Error: {data}")
                    break

        latency = time.perf_counter() - start
        answer_placeholder.markdown(accumulated)
        route_placeholder.caption(
            f"{_route_badge(final_route)}  ·  ⏱ {latency:.1f}s"
        )

        if final_sources:
            with st.expander(f"📚 Sources ({len(final_sources)} docs)", expanded=False):
                for src in final_sources:
                    st.markdown(
                        f"**{src['source']}** · `{src['department']}` · "
                        f"score: `{src['score']}`"
                    )
                    st.caption(f"> {src.get('text_snippet', '')}…")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": accumulated,
        "sources": final_sources,
        "route_type": final_route,
    })

elif user_input and not api_ok:
    st.error("❌ API is offline. Make sure the FastAPI server is running.")
