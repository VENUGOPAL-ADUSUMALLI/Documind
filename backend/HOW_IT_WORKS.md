# How the backend works

Read this first if you come back after a few months.

## The big picture

Two stores, each with one job:

- **Postgres**: the real data. Documents, their status, and the text of every chunk.
- **Qdrant**: a search index. One vector (list of numbers meaning "what this text is about") per chunk.
  It can always be rebuilt from Postgres.

## Flow 1: uploading a PDF   (`POST /documents`)

| Step | What happens | Code |
|---|---|---|
| 1 | Check size and that it is a PDF | `main.py` `upload_document` |
| 2 | Fingerprint the file; skip if already done | `ingestion.py` `ingest_pdf` |
| 3 | Save a "processing" record in Postgres | `ingestion.py` `ingest_pdf` |
| 4 | Read the text, page by page | `chunking.py` `extract_pages` |
| 5 | Cut text into ~800-character chunks | `chunking.py` `split_pages` |
| 6 | Turn each chunk into a vector | `embeddings.py` `embed_documents` |
| 7 | Save chunk text in Postgres, vectors in Qdrant | `ingestion.py` `process_document` |
| 8 | Mark the document "done" (or "failed" + error) | `ingestion.py` |

## Flow 2: asking a question   (`POST /ask`)

| Step | What happens | Code |
|---|---|---|
| 1 | Turn the question into a vector | `rag.py` `search_qdrant` |
| 2 | Qdrant returns the 100 closest chunks | `vector_store.py` `search` |
| 3 | Load those chunks' text from Postgres | `rag.py` `load_chunk_text` |
| 4 | Reranker re-sorts them; keep the best 10 | `rag.py` `rerank`, `reranker.py` |
| 5 | Put them into one labelled text block | `rag.py` `build_context` |
| 6 | The LLM writes an answer with citations | `llm.py` |

## Where to change things

All settings are in `app/config.py` (and can be overridden with environment variables):

| Want to change | Setting |
|---|---|
| Chunk size | `chunk_size`, `chunk_overlap` (re-upload documents afterwards) |
| How many chunks the LLM reads | `retrieval_top_k` |
| How many candidates the reranker sees | `retrieval_candidates` |
| Turn the reranker off | `rerank_enabled` |
| Embedding provider (openai / gemini) | `embedding_provider` |
| Answer model | `chat_model` |

## Why some choices were made

- **Same chunk id for the same file + position**: uploading a file twice never creates duplicates.
- **Status column on documents**: a crash or API failure is visible and retryable.
- **Reranker**: vector search alone missed "teammate" vs "Student Name(s)"; the reranker reads
  question and passage together and fixed it.

## Run the tests

    cd backend && python -m pytest
