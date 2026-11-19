# DocuMind

Upload PDFs and ask questions about them. Answers are grounded in the documents and cite the filename and page.
Every citation is checked against the pages that were actually sent to the model, and invented ones are removed.

## Architecture

```
Streamlit UI  ->  FastAPI  ->  PostgreSQL   (documents, ingestion status, chunk text)
                          ->  Qdrant       (dense + BM25 keyword vectors per chunk)
                          ->  OpenAI       (embeddings + answer generation)
                          ->  Cross-encoder reranker (runs locally)
```

Postgres is the source of truth; Qdrant is a search index that can be rebuilt from it
(`cd backend && python -m app.reindex`).
Chunk IDs are deterministic (hash of file + position), so re-uploading a file never creates duplicates.

### Asking a question

1. **Hybrid retrieval**: the question is searched by meaning (dense) and by exact words (BM25) in Qdrant,
   and the two result lists are merged with Reciprocal Rank Fusion (up to 100 candidates).
2. **Reranking**: a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-sorts the candidates; the best few are kept.
3. **Context expansion**: each match is widened with its neighbouring chunks on the same page, within a character budget.
4. **Answer**: the LLM answers from that context with citations.
5. **Citation check**: citations that point to pages not in the context are removed.

See [`backend/HOW_IT_WORKS.md`](backend/HOW_IT_WORKS.md) for a step-by-step walkthrough with the code locations.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/documents` | Upload a PDF (max 20 MB) and ingest it |
| `GET` | `/documents` | List documents and their status |
| `GET` | `/documents/{id}` | Get one document |
| `DELETE` | `/documents/{id}` | Delete a document and its chunks |
| `POST` | `/ask` | Ask a question, get an answer with citations |

Interactive docs: http://localhost:8000/docs

## Run locally

1. Create a `.env` file at the repo root:
   ```
   OPENAI_API_KEY=...
   DATABASE_URL=postgresql+psycopg://documind:documind@localhost:5432/documind
   ```
   `DATABASE_URL` is required (the value above matches `docker-compose.yml`); `postgres://` and `postgresql://`
   URLs from hosts such as Railway are converted automatically.
2. Start Postgres and Qdrant: `docker compose up -d`
3. Backend:
   ```
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
4. Frontend (new terminal):
   ```
   cd frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Configuration

Settings live in `backend/app/config.py` and can be overridden with environment variables or `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | – | OpenAI key for embeddings and answers |
| `DATABASE_URL` | **required** | Postgres connection URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | empty | Qdrant API key (for hosted Qdrant) |
| `CHAT_MODEL` | `gpt-4o-mini` | Answer model |
| `OPENAI_EMBEDDING_MODEL` / `EMBEDDING_DIM` | `text-embedding-3-small` / `1536` | Embeddings (reindex after changing) |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`, `dense` or `bm25` |
| `RETRIEVAL_TOP_K` | `10` | Chunks the LLM reads |
| `RETRIEVAL_CANDIDATES` | `100` | Candidates passed to the reranker |
| `RERANK_ENABLED` | `true` | Turn the reranker on/off |
| `CONTEXT_WINDOW` | `1` | Neighbouring chunks added around a match (0 = none) |
| `CONTEXT_CHAR_BUDGET` | `6000` | Max characters sent to the LLM |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Chunking (re-upload documents after changing) |

Frontend variables:

| Variable | Default | Meaning |
|---|---|---|
| `DOCUMIND_API_URL` | `http://localhost:8000` | Where the Streamlit app finds the backend |
| `APP_PASSWORD` | unset | If set, the UI asks for this password before showing anything |

## Evaluation

`backend/eval/` measures retrieval quality (hit-rate@5 and MRR) for dense, BM25, hybrid, and reranked
configurations, plus answer completeness for different context-window sizes:

```
cd backend
python -m eval.run_eval --docs /path/to/folder/with/the/pdfs
```

Questions are in `backend/eval/questions.json`.

## Tests

```
cd backend && python -m pytest
cd frontend && python -m pytest tests
```

## Deployment (Railway)

Both services have a Dockerfile (`backend/Dockerfile`, `frontend/Dockerfile`).
The backend image uses CPU-only PyTorch, pre-downloads the reranker model, runs as a non-root user and listens on `$PORT`.
Set `DATABASE_URL`, `QDRANT_URL`, `QDRANT_API_KEY` and `OPENAI_API_KEY` on the backend, and
`DOCUMIND_API_URL` (plus optionally `APP_PASSWORD`) on the frontend.
