# Enterprise Document RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for querying
company internal documents using natural language.

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

**Streaming SSE responses**
The FastAPI backend streams tokens via Server-Sent Events so the UI renders
each word as it arrives. Source citations are sent as a final structured SSE
event after the last token.

---

## Project Structure

```
rag-internal-docs/
├── ingestion/
│   ├── loader.py          # LlamaParse + SimpleDirectoryReader
│   ├── chunker.py         # Hierarchical + flat chunking strategies
│   └── embedder.py        # Qdrant vector store build + reload
├── retrieval/
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
git clone https://github.com/Mengxuerobott/rag-internal-docs.git
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
