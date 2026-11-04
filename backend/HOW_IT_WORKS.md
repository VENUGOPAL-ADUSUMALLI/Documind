# How the backend works

Read this first if you come back after a few months.

## The big picture

Two stores, each with one job:

- **Postgres**: the real data. Documents, their status, and the text of every chunk.
- **Qdrant**: a search index. Each chunk has two vectors: a **dense** one (numbers meaning "what this text is about")
  and a **bm25** one (weights for the exact words in it). It can always be rebuilt from Postgres (`python -m app.reindex`).

## Flow 1: uploading a PDF   (`POST /documents`)

| Step | What happens | Code |
|---|---|---|
| 1 | Check size and that it is a PDF | `main.py` `upload_document` |
| 2 | Fingerprint the file; skip if already done | `ingestion.py` `ingest_pdf` |
| 3 | Save a "processing" record in Postgres | `ingestion.py` `ingest_pdf` |
| 4 | Read the text, page by page | `chunking.py` `extract_pages` |
| 5 | Cut text into ~800-character chunks | `chunking.py` `split_pages` |
| 6 | Turn each chunk into a dense vector | `embeddings.py` `embed_documents` |
| 7 | Save chunk text in Postgres; dense + keyword (BM25) vectors in Qdrant | `ingestion.py`, `vector_store.py` `upsert_chunks`, `sparse.py` |
| 8 | Mark the document "done" (or "failed" + error) | `ingestion.py` |

## Flow 2: asking a question   (`POST /ask`)

| Step | What happens | Code |
|---|---|---|
| 1 | Turn the question into a dense vector and a keyword vector | `rag.py` `search_qdrant`, `sparse.py` |
| 2 | Qdrant runs both searches and merges them with RRF; returns 100 candidates | `vector_store.py` `search` |
| 3 | Load those chunks' text from Postgres | `rag.py` `load_chunk_text` |
| 4 | Reranker re-sorts them; keep the best few | `rag.py` `rerank`, `reranker.py` |
| 5 | Widen each match with its neighbouring chunks (same page), merge overlaps, stay within the character budget | `rag.py` `build_passages` |
| 6 | Put the passages into one labelled text block | `rag.py` `build_context` |
| 7 | The LLM writes an answer with citations | `llm.py` |
| 8 | Check every citation against the pages we sent; remove invented ones | `citations.py` |

## Where to change things

All settings are in `app/config.py` (and can be overridden with environment variables):

| Want to change | Setting |
|---|---|
| Chunk size | `chunk_size`, `chunk_overlap` (re-upload documents afterwards) |
| How many chunks the LLM reads | `retrieval_top_k` |
| How many candidates the reranker sees | `retrieval_candidates` |
| Search mode: `hybrid`, `dense` or `bm25` | `retrieval_mode` |
| How many neighbouring chunks to add around a match (0 = none) | `context_window` |
| Maximum characters sent to the LLM | `context_char_budget` |
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

## Hybrid search in one paragraph

Dense search finds chunks by meaning (good for paraphrases, bad for exact terms). BM25 finds chunks by
exact words (good for names, codes, rare terms; bad for paraphrases). Hybrid runs both and merges the
two ranked lists with Reciprocal Rank Fusion: each chunk gets `1 / (60 + rank)` from each list and the
scores are added, so a chunk that is high in both lists wins. Filler words ("what", "the") are ignored
for BM25 (`sparse.py` `STOPWORDS`).

## After changing the Qdrant layout or embedding model

Postgres keeps all chunk text, so the PDFs are not needed again:

    cd backend && python -m app.reindex

## Evaluate a change

    cd backend && python -m eval.run_eval --docs /folder/with/pdfs

## Context window (neighbour expansion)

A small chunk matches well but can cut an answer in half. After reranking, each matched chunk is widened
with `context_window` chunks before and after it, taken from the same page. Windows that overlap are merged,
and the repeated overlap text between neighbouring chunks is removed. Passages are added best-first until
`context_char_budget` is reached; a passage that does not fit is skipped.

Limit: windows stay on one page (so page citations stay exact), so a list that continues onto the next page
is only completed if the next page's chunks are also retrieved.

Measured on 6 list/fact questions (see `python -m eval.run_eval`): matched chunk only 48% of required facts,
window +-1 85%, window +-2 81%. Small question set: treat as a direction, not a precise number.
