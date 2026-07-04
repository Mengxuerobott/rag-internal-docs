# Enterprise Document RAG System

A Retrieval-Augmented Generation service for asking natural-language questions about a company's internal documents. It returns answers with source citations and enforces role-based access control, so a user only ever sees answers drawn from documents their role is allowed to read.

The retrieval side uses hybrid search (keyword + vector) with an AutoMerging retriever and a cross-encoder reranker. The service is containerized and runs on AWS ECS Fargate against a managed Qdrant vector store.

## How it works

Ingestion loads documents, chunks them hierarchically (parent and leaf sizes), embeds them with OpenAI, and stores dense and sparse vectors in Qdrant. Each chunk is tagged with the roles allowed to see it and the department it came from.

At query time, the request is authenticated (JWT) and routed by intent. A knowledge question goes through the full pipeline: hybrid BM25 + vector retrieval with Reciprocal Rank Fusion, an AutoMerging step that swaps matched leaf chunks for their larger parent chunks to restore context, a Cohere cross-encoder rerank that keeps the top few, and finally answer synthesis with inline citations. RBAC is applied as a Qdrant payload filter before retrieval, not after, so restricted documents never enter the candidate set.

## Stack

- Retrieval framework: LlamaIndex
- Vector database: Qdrant (Qdrant Cloud free tier in the deployed version)
- Embeddings: OpenAI `text-embedding-3-small` (1536-dim)
- Sparse model: SPLADE (`prithvida/Splade_PP_en_v1`) for the keyword side of hybrid search
- Reranker: Cohere Rerank v3 (cross-encoder)
- LLM: OpenAI (configurable in `.env`)
- Evaluation: RAGAS
- API: FastAPI with streaming (SSE) responses
- UI: Streamlit
- Auth: JWT with role-based access control
- Packaging: Docker; deployed on AWS ECS Fargate with the image in Amazon ECR
- CI/CD: GitHub Actions (OIDC auth), build and push to ECR, deploy to Fargate on push to `main`

## Evaluation

Measured with RAGAS on a set of hand-labelled question/answer pairs across the sample document categories.

| Metric | Score |
| --- | --- |
| Faithfulness | 0.91 |
| Answer relevancy | 0.89 |
| Context precision | 0.84 |
| Context recall | 0.82 |

Adding the Cohere reranker was the single biggest change: faithfulness went from 0.73 to 0.91. The reranker pushes the genuinely relevant chunks to the top, so the model sees cleaner context and has less room to make things up.

Numbers are from this project's own eval set and are meant to show the relative effect of each component, not to be a general benchmark.

## Design notes

**Hierarchical chunking.** Small leaf chunks are indexed so retrieval can match precisely, but small chunks lose surrounding context. The AutoMerging retriever handles this: when several leaf chunks under the same parent are retrieved, it returns the parent instead, so the model gets the full passage rather than a fragment.

**Hybrid search.** Vector search alone misses exact-string matches like product codes, acronyms, and proper nouns. BM25 catches those. Reciprocal Rank Fusion merges the two ranked lists without needing the scores to be on the same scale.

**Cross-encoder rerank.** The first-stage retriever embeds the query and each chunk separately, which is fast but approximate. The reranker reads each (query, chunk) pair together, which is a better relevance judgment, so it runs only on the shortlist from the first stage.

**RBAC as a pre-filter.** Access rules are stored on each chunk and applied as a Qdrant filter before the vector search runs, so a user's query is only ever compared against documents they are allowed to see.

## Deployment (AWS ECS Fargate)

The API is packaged as a container and runs on Fargate. A few notes that matter if you deploy it yourself:

- Build the image for `linux/amd64`. If you build on an Apple Silicon Mac, pass `--platform linux/amd64` or the container will not run on Fargate.
- The SPLADE sparse model is downloaded during the image build and baked into the image, so the container does not fetch it at startup. Downloading it at runtime inside the container was unreliable.
- Secrets (API keys, Qdrant URL) are passed as environment variables in the ECS task definition, not baked into the image. `.env` and the task definition file are gitignored.
- Hybrid mode loads the SPLADE model in addition to building the index, so the task needs more memory than a minimal container; 4 GB was enough, 2 GB was not.
- The semantic cache (Redis) is enabled locally but turned off in the deployed version to avoid running a managed Redis instance. The caching code is still in the repo.

### CI/CD

Pushing to `main` triggers a GitHub Actions workflow (`.github/workflows/deploy.yml`) that builds the image for `linux/amd64`, pushes it to ECR (tagged both `latest` and the commit SHA), and triggers a new Fargate deployment.

Authentication to AWS uses OIDC rather than a stored access key: GitHub and AWS establish a trust relationship, and each workflow run assumes an IAM role to get short-lived credentials. The role's trust policy is scoped to this repository only, so no long-lived AWS credentials are stored in GitHub. Tagging each image with its commit SHA means every deployed image traces back to an exact commit.

## Project layout

```
rag-internal-docs/
  ingestion/
    loader.py        document loading, RBAC tagging
    chunker.py       hierarchical chunking
    embedder.py      collection setup, Qdrant vector store build/reload
  retrieval/
    query_engine.py  hybrid search + AutoMerging + reranker
    reranker.py      reranker helpers
    handlers.py      per-intent query handlers
    router.py        intent routing
  api/
    main.py          FastAPI app: /query, /query/stream, /ingest, /health
  auth/
    jwt_handler.py   JWT auth
    rbac.py          role/department access rules
  cache/
    semantic_cache.py
  ui/
    app.py           Streamlit chat UI
  eval/
    ragas_eval.py    RAGAS evaluation
  scripts/
    generate_sample_docs.py
  tests/
  config.py
  docker-compose.yml
  Dockerfile.api
  Dockerfile.ui
  requirements.txt
```

## Running it locally

You need Docker Desktop and API keys for OpenAI, Cohere (free tier), and LlamaCloud (free tier).

Clone and configure:

```
git clone https://github.com/Mengxuerobott/rag-internal-docs.git
cd rag-internal-docs
cp .env.example .env
# fill in your API keys in .env
```

Generate the sample documents (7 short internal policy docs across HR, engineering, finance, legal, and IT security):

```
pip install -r requirements.txt
python scripts/generate_sample_docs.py
```

Start everything with Docker Compose:

```
docker compose up --build
```

| Service | URL |
| --- | --- |
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |

Or run the pieces by hand:

```
# Qdrant
docker run -p 6333:6333 qdrant/qdrant

# ingest documents
python -m ingestion.embedder

# API
uvicorn api.main:app --reload

# UI
streamlit run ui/app.py
```

The demo users are defined in `auth/jwt_handler.py`, each with a different role, all with the password `secret`. Logging in as different users shows RBAC in action: an HR user can see the leave policy, a general employee cannot.

## Ingestion

```
# ingest (or add new documents)
python -m ingestion.embedder

# wipe the collection and re-embed everything
python -m ingestion.embedder --force-rebuild

# ingest from a different directory
python -m ingestion.embedder --docs-dir /path/to/docs
```

## Evaluation

```
python -m eval.ragas_eval
python -m eval.ragas_eval --output results/ragas_scores.json
python -m eval.ragas_eval --test-cases data/my_test_cases.json
```

Test case format:

```
[
  {
    "question": "What is the parental leave policy?",
    "ground_truth": "16 weeks fully paid for primary caregivers."
  }
]
```

## Tests

```
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

The engine is mocked, so the tests do not need external services running.

## API

`POST /query` — full answer once ready.

```
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the parental leave policy?"}'
```

`POST /query/stream` — streams tokens over SSE, then a sources event.

`POST /ingest` — triggers re-ingestion in the background.

`GET /health` — liveness/readiness check (no auth), used as the container health check.

Get a token from `POST /token` (or `/auth/token`) with a demo username and the password `secret`.

## Configuration

Everything is set in `.env`. The main options:

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o` | model used for answer generation |
| `TOP_K_RETRIEVAL` | `10` | candidates fetched before reranking |
| `TOP_N_RERANK` | `3` | chunks kept after reranking |
| `HYBRID_ALPHA` | `0.5` | BM25 vs vector blend (0 = BM25, 1 = vector) |
| `CHUNK_SIZES` | `2048,512,128` | hierarchical chunk sizes in tokens |
| `SEMANTIC_CACHE_ENABLED` | `true` | Redis semantic cache; set to `false` if you are not running Redis |

## Known limitations

- The container re-ingests documents on startup so the Qdrant collection and the in-memory docstore are always built together with matching IDs. This is fine for a demo but re-embeds on every deploy; separating ingestion from serving would be the next step.
- The deployed version runs dense retrieval quality with hybrid enabled; the semantic cache is off in the cloud to avoid a managed Redis instance.
- Access is over the task's public IP, which changes on restart. A load balancer would give a stable URL.

## License

MIT
