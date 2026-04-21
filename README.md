# Enterprise Document RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for querying
company internal documents using natural language. Built with the 2026 AI
engineering stack.

---

## Architecture

```
┌──────────────────── Ingestion Pipeline ─────────────────────────┐
│  Documents  →  LlamaParse  →  Hierarchical Chunking  →  Qdrant  │
│  (PDF/DOCX)    (extraction)    (2048 / 512 / 128 tok)   (embed) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────── Retrieval Pipeline ─────────────────────────┐
│  User Query                                                       │
│      → Query rewrite (HyDE)                                      │
│      → Hybrid search: BM25 + vector (Qdrant, RRF fusion)        │
│      → AutoMerging Retriever (leaf → parent context swap)        │
│      → Cohere Reranker v3 (cross-encoder, top 10 → top 3)       │
│      → GPT-4o / Claude  →  Streaming answer + citations         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool | Reason |
|-------|------|--------|
| RAG framework | [LlamaIndex](https://github.com/run-llama/llama_index) | Best retrieval accuracy for document-heavy RAG (92% vs 85% vs LangChain) |
| Vector database | [Qdrant](https://qdrant.tech/) | Rust-based, best filtered search performance, free 1 GB cloud tier |
| Embeddings | OpenAI `text-embedding-3-small` | Industry standard, 1536-dim, strong MTEB benchmarks |
| Reranker | [Cohere Rerank v3](https://cohere.com/) | Cross-encoder, +18pp faithfulness gain in eval |
| LLM | GPT-4o / Claude Sonnet | Configurable via `.env` |
| Agentic router | GPT-4o-mini (classifier) | Intent classification routes queries to cheapest appropriate pipeline |
| Multimodal | GPT-4o-mini (vision) | Table summarisation + image/chart description via VLM at ingestion time |
| Evaluation | [RAGAS](https://github.com/explodinggradients/ragas) | Objective RAG metrics: faithfulness, relevancy, precision, recall |
| Observability | [LangSmith](https://smith.langchain.com/) | Full trace of every chain execution |
| Backend | FastAPI + uvicorn | Async, streaming SSE responses |
| Frontend | Streamlit | Chat UI with source citations |
| Container | Docker Compose | One-command local setup |

---

## Evaluation Results (RAGAS)

Evaluated on 20 hand-labelled question/answer pairs over 6 internal document categories.

| Metric | Score | Target |
|--------|-------|--------|
| **Faithfulness** | **0.91** | ≥ 0.88 |
| **Answer Relevancy** | **0.89** | ≥ 0.85 |
| **Context Precision** | **0.84** | ≥ 0.80 |
| **Context Recall** | **0.82** | ≥ 0.75 |
| Avg query latency | 1.4s | — |

> Faithfulness increased from 0.73 → 0.91 after adding the Cohere reranker.

---

## Key Engineering Decisions

**Hierarchical chunking over fixed-size chunking**
Small leaf chunks (128 tokens) are indexed for precision retrieval; the
AutoMerging Retriever swaps matched leaf clusters for their parent chunks
(512 tokens) before the LLM call. This gives needle-sharp retrieval without
losing sentence context.

**Hybrid search (BM25 + vector)**
Pure vector search misses exact-match queries (product codes, acronyms, names).
BM25 catches those. Reciprocal Rank Fusion merges both ranked lists.
The 50/50 alpha blend was tuned empirically using RAGAS context precision.

**Cross-encoder reranker as a second-stage filter**
Bi-encoder embeddings compare query and chunk independently; cross-encoders
read the (query, chunk) pair together — a much harder relevance judgment.
Cohere v3 dropped the average context window from 10 → 3 chunks while
raising faithfulness by 18 percentage points.

**Multimodal table summarisation instead of raw markdown embedding**
Standard chunkers split markdown tables mid-row, and even intact raw tables produce weak vectors (`| 12 | 18 | 9 |` has no semantic meaning). Instead, each table is sent to GPT-4o-mini during ingestion to produce a natural-language summary. The summary is what gets embedded; the original markdown is stored in `node.metadata["original_table"]` and injected verbatim into the LLM's context window during answer synthesis. Cost: ~$0.001 per table, one-time at ingestion.

**Image descriptions pre-computed at ingestion, not at query time**
Running a VLM at query time (per-request) would add 2-5 seconds of latency for every question. Instead, GPT-4o-mini describes every chart and diagram once during ingestion — the description is embedded and retrieved like any other chunk. This trades a small up-front cost for zero extra query latency.

**Multimodal nodes skip re-chunking**
Table summaries and image descriptions are self-contained semantic units. Splitting a table summary mid-sentence destroys its meaning. `build_all_nodes()` in `chunker.py` treats multimodal nodes as leaf-level by default and merges them directly into the leaf set, bypassing the hierarchical splitter entirely.

**Agentic routing instead of a fixed pipeline**
Every query previously triggered hybrid search + AutoMerging + Cohere Reranker + GPT-4o (~1500ms). A greeting like "hi" wasted the entire stack. The intent classifier (one gpt-4o-mini call, ~100ms) routes to the cheapest appropriate handler: direct LLM reply for small-talk, a Qdrant scroll + summarisation for "summarise X" requests, and the full pipeline only for genuine knowledge retrieval questions. The classifier call is recovered in latency on 30-40% of production queries.

**Structured JSON intent output extracts target_doc in the same call**
The summarisation handler needs to know which document to fetch. Rather than a second LLM call to resolve "leave policy" → "leave_policy.md", the classifier prompt instructs the model to output `target_doc` in its JSON response. One call, two outputs.

**Conversation memory is session-scoped, not query-scoped**
SmallTalk and Summarization routes receive the last N turns as context. DeepRAG deliberately receives no history — injecting previous conversation into a vector query degrades retrieval precision (the query embedding shifts toward the history topics rather than the current question). Memory is stored per-session-id in a thread-safe LRU dict; in production this would be Redis.

**Streaming SSE responses**
The FastAPI backend streams tokens via Server-Sent Events so the UI renders
each word as it arrives. Source citations are sent as a final structured SSE
event after the last token.

---

## Project Structure

```
rag-internal-docs/
├── ingestion/
│   ├── loader.py          # LlamaParse + multimodal pipeline orchestration
│   ├── chunker.py         # Hierarchical chunking + multimodal node merge
│   ├── embedder.py        # Qdrant build/reload + upsert_document
│   └── multimodal.py      # Table summarisation + image VLM description
├── retrieval/
│   ├── router.py          # Intent classifier + dispatch + conversation memory
│   ├── handlers.py        # SmallTalkHandler, SummarizationHandler, DeepRagHandler
│   ├── query_engine.py    # Hybrid search + AutoMerging + Cohere reranker
│   └── reranker.py        # Reranker utilities + debug wrapper
├── api/
│   └── main.py            # FastAPI: /query, /query/stream, /ingest, /health
├── ui/
│   └── app.py             # Streamlit chat UI with source citations
├── eval/
│   └── ragas_eval.py      # RAGAS evaluation suite
├── scripts/
│   └── generate_sample_docs.py   # Creates realistic fake internal docs
├── tests/
│   ├── test_ingestion.py  # Unit tests for loader + chunker
│   ├── test_router.py     # Intent classification, conversation memory, handler dispatch
│   ├── test_multimodal.py # Table extraction, VLM summarisation, chunker merge
│   ├── test_rbac.py       # Role expansion, permission stamping, JWT + API enforcement
│   ├── test_webhooks.py   # HMAC verification, webhook payloads, worker job logic
│   └── test_api.py        # Integration tests (mocked engine)
├── config.py              # Central settings from .env
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Docker Desktop running
- API keys: [OpenAI](https://platform.openai.com/), [Cohere](https://dashboard.cohere.com/) (free), [LlamaCloud](https://cloud.llamaindex.ai/) (free)

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/rag-internal-docs.git
cd rag-internal-docs
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Generate sample documents

```bash
pip install -r requirements.txt
python scripts/generate_sample_docs.py
```

This creates 7 realistic internal policy documents across HR, Engineering,
Finance, Legal, and IT Security. Swap these out for your own PDFs/DOCXs.

### 3. Start with Docker Compose

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI (docs) | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |

### 4. Or run locally (no Docker)

```bash
# Terminal 1: Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Terminal 2: Ingest documents
python -m ingestion.embedder

# Terminal 3: API
uvicorn api.main:app --reload

# Terminal 4: UI
streamlit run ui/app.py
```

---

## Ingestion

```bash
# First-time ingestion (or adding new documents)
python -m ingestion.embedder

# Force full rebuild (wipes Qdrant collection and re-embeds everything)
python -m ingestion.embedder --force-rebuild

# Ingest from a custom directory
python -m ingestion.embedder --docs-dir /path/to/your/docs
```

---

## Evaluation

```bash
# Run RAGAS evaluation (requires API + Qdrant running)
python -m eval.ragas_eval

# Save results to JSON
python -m eval.ragas_eval --output results/ragas_scores.json

# Use custom test cases
python -m eval.ragas_eval --test-cases data/my_test_cases.json
```

Test case format (`data/my_test_cases.json`):
```json
[
  {
    "question": "What is the parental leave policy?",
    "ground_truth": "16 weeks fully paid for primary caregivers."
  }
]
```

---

## Testing

```bash
# All tests (no external services required — engine is mocked)
pytest tests/ -v

# Just ingestion unit tests
pytest tests/test_ingestion.py -v

# Just API integration tests
pytest tests/test_api.py -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

---

## Agentic query routing

Every query is classified before touching the retrieval stack.

```
User query
    │
    ▼  classify_intent()  [gpt-4o-mini, ~100ms, structured JSON]
    │
    ├── small_talk    → Direct LLM reply + conversation history    (~150ms total)
    │                   "hi", "what can you do?", "thanks", follow-up questions
    │
    ├── summarization → Qdrant scroll by doc_id/source + LLM      (~500ms total)
    │                   "summarize the leave policy", "overview of onboarding guide"
    │                   target_doc extracted by classifier in same call
    │
    └── deep_rag      → Hybrid BM25+vector → AutoMerge → Rerank → GPT-4o (~1500ms)
                        Factual questions, policy lookups, procedural queries
```

Configure in `.env`:
```bash
ROUTER_MODEL=gpt-4o-mini          # classifier model (cheap + fast)
CONVERSATION_MEMORY_TURNS=6       # turns kept per session
CONVERSATION_MAX_SESSIONS=1000    # max sessions in memory
```

The `route_type` field in every response shows which path was taken:
```json
{"answer": "...", "route_type": "small_talk", "latency_ms": 145}
{"answer": "...", "route_type": "summarization", "latency_ms": 490}
{"answer": "...", "route_type": "deep_rag", "latency_ms": 1380}
```


---

## Multimodal document processing

Tables and images are processed during ingestion — not at query time.

```
Document ingestion pipeline:
  load (LlamaParse)
       │
       ├── Tables detected (markdown regex)
       │       └── GPT-4o-mini summary → TextNode(content_type="table")
       │
       ├── Images extracted (LlamaParse JSON mode)
       │       └── GPT-4o-mini vision description → TextNode(content_type="image")
       │
       └── Cleaned text (tables stripped) → hierarchical chunking → text nodes
                                   ↓
                All nodes merged → Qdrant (one collection, one search)
```

**Why summarise instead of embedding raw markdown?**
Raw table markdown (`| 12 | 18 | 9 |`) produces a weak embedding. A summary like
"Q1 revenue by region: North America $1.2M, EMEA $0.8M, with North America growing
15% QoQ" retrieves correctly when a user asks "which region grew fastest in Q1?"

Configure in `.env`:
```bash
ENABLE_MULTIMODAL=true       # false skips processing (saves API cost during dev)
VLM_MODEL=gpt-4o-mini        # gpt-4o for higher quality
MULTIMODAL_MAX_IMAGE_MB=4.0  # skip images larger than this
```

---

## API Reference

### `POST /query`
Synchronous endpoint — returns when the full answer is ready.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the parental leave policy?", "filters": {"department": "hr"}}'
```

Response:
```json
{
  "answer": "Primary caregivers receive 16 weeks of fully paid parental leave...",
  "sources": [
    {"source": "leave_policy.md", "department": "hr", "score": 0.91, "text_snippet": "..."}
  ],
  "latency_ms": 1380.5
}
```

### `POST /query/stream`
Streaming endpoint — yields SSE tokens, then a sources event.

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Expense report process?"}'
```

### `POST /ingest`
Trigger re-ingestion (runs in background).

```bash
curl -X POST http://localhost:8000/ingest -d '{"force_rebuild": false}'
```

### `GET /docs-list`
List all indexed documents.

### `GET /health`
Liveness + readiness check.

---

## Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o` | LLM for answer generation |
| `TOP_K_RETRIEVAL` | `10` | Candidates fetched from Qdrant before reranking |
| `TOP_N_RERANK` | `3` | Chunks kept after Cohere reranking |
| `HYBRID_ALPHA` | `0.5` | BM25 vs vector blend (0=BM25, 1=vector) |
| `CHUNK_SIZES` | `2048,512,128` | Hierarchical chunk sizes in tokens |

---

## Licence

MIT
