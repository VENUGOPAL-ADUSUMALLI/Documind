# DocuMind

Upload PDFs and ask questions about them. Answers are grounded in the documents and cite the filename and page.

## Architecture

```
Streamlit UI  ->  FastAPI  ->  PostgreSQL   (documents, ingestion status, chunk text)
                          ->  Qdrant       (chunk vectors, searched by meaning)
                          ->  OpenAI       (embeddings + answer generation)
```

Postgres is the source of truth; Qdrant is a search index that can be rebuilt from it.
Chunk IDs are deterministic (hash of file + position), so re-uploading a file never creates duplicates.

## Run

1. Put `OPENAI_API_KEY=...` in a `.env` file at the repo root.
2. Start the databases: `docker compose up -d`
3. Backend:
   ```
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
   API docs: http://localhost:8000/docs
4. Frontend (new terminal):
   ```
   cd frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Tests

```
cd backend && python -m pytest
cd frontend && python -m pytest tests
```
