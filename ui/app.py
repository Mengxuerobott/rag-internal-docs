"""
ui/app.py
─────────
Streamlit chat interface for the Internal Docs RAG assistant.

Features:
  - Streaming chat with token-by-token display
  - Source citation panel (expander below each answer)
  - Department filter sidebar (filter searches to a specific team's docs)
  - Document index browser (sidebar — lists all ingested files)
  - Re-ingest button (triggers /ingest endpoint)
  - Session history (persisted in Streamlit session state)

Run locally:
    streamlit run ui/app.py

Requires API to be running:
    uvicorn api.main:app --reload
"""

import json
import os
import time

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "department_filter" not in st.session_state:
    st.session_state.department_filter = None
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None


# ── Auth helpers ──────────────────────────────────────────────────────────────
def _auth_headers() -> dict:
    token = st.session_state.get("jwt_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


# ── Helper: call the API ──────────────────────────────────────────────────────
def stream_query(question: str, department_filter: str | None = None):
    """
    Call POST /query/stream and yield (token, sources) tuples.
    Sources arrive in the last SSE event.
    """
    payload = {
        "question": question,
        "session_id": st.session_state.get("session_id", "streamlit"),
        "department_filter": department_filter,
    }
    sources = []

    with httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            f"{API_BASE}/query/stream",
            json=payload,
            headers=_auth_headers(),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]

                if chunk == "[DONE]":
                    break

                if chunk.startswith("[SOURCES]"):
                    raw = chunk[len("[SOURCES]"):]
                    try:
                        sources = json.loads(raw).get("sources", [])
                    except json.JSONDecodeError:
                        pass
                    yield "", sources

                elif chunk.startswith("[ERROR]"):
                    st.error(f"API error: {chunk[7:]}")
                    break

                else:
                    yield chunk, []


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
        return r.json().get("status", "unknown")
    except Exception as e:
        return f"Error: {e}"


def check_api_health() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5.0)
        return r.json().get("index_ready", False)
    except Exception:
        return False


# ── Login (simple — shown if no token) ───────────────────────────────────────
if not st.session_state.jwt_token:
    st.title(f"{APP_ICON} {APP_TITLE}")
    with st.form("login"):
        username = st.text_input("Username", placeholder="alice, bob, admin…")
        password = st.text_input("Password", type="password", value="secret")
        if st.form_submit_button("Sign in"):
            try:
                r = httpx.post(
                    f"{API_BASE}/auth/token",
                    data={"username": username, "password": password},
                    timeout=10.0,
                )
                if r.status_code == 200:
                    st.session_state.jwt_token = r.json()["access_token"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials. Password for all demo accounts: secret")
            except Exception as e:
                st.error(f"Could not reach API: {e}")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")

    api_ok = check_api_health()
    st.markdown(f"**API status:** {'🟢 Ready' if api_ok else '🔴 Offline'}")
    st.divider()

    # Department filter
    st.subheader("Filter by department")
    dept_options = ["All documents", "hr", "engineering", "finance", "legal", "general"]
    selected_dept = st.selectbox("Department", dept_options, index=0)
    st.session_state.department_filter = (
        None if selected_dept == "All documents" else selected_dept
    )

    st.divider()

    # Indexed documents browser
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

    st.subheader("Manage index")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Re-ingest"):
            status = trigger_ingest(force=False)
            st.success(f"Status: {status}")
    with col2:
        if st.button("🔁 Rebuild"):
            status = trigger_ingest(force=True)
            st.warning(f"Status: {status}")

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.header(f"{APP_ICON} Ask your internal docs")

if st.session_state.department_filter:
    st.info(f"🔍 Searching only: **{selected_dept}** department docs")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} documents)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source']}** · `{src['department']}` · "
                        f"score: `{src['score']}`"
                    )
                    st.caption(f"> {src['text_snippet']}…")
                    st.divider()

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
        answer_placeholder = st.empty()
        accumulated = ""
        final_sources = []
        start = time.perf_counter()

        with st.spinner("Searching docs…"):
            for token, sources in stream_query(
                user_input,
                department_filter=st.session_state.department_filter,
            ):
                if sources:
                    final_sources = sources
                else:
                    accumulated += token
                    answer_placeholder.markdown(accumulated + "▌")

        latency = time.perf_counter() - start
        answer_placeholder.markdown(accumulated)
        st.caption(f"⏱ {latency:.1f}s")

        if final_sources:
            with st.expander(f"📚 Sources ({len(final_sources)} documents)", expanded=False):
                for src in final_sources:
                    st.markdown(
                        f"**{src['source']}** · `{src['department']}` · "
                        f"score: `{src['score']}`"
                    )
                    st.caption(f"> {src['text_snippet']}…")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": accumulated,
        "sources": final_sources,
    })

elif user_input and not api_ok:
    st.error("❌ API is offline. Make sure the FastAPI server is running.")
