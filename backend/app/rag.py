"""Answering a question:  question -> find relevant chunks -> ask the LLM -> answer.

Finding the chunks has three steps:
  1. SEARCH   Qdrant returns many candidate chunks (by similarity of meaning).
  2. LOAD     Qdrant only knows ids and scores, so we load the real text from Postgres.
  3. RERANK   A second, more careful model re-sorts the candidates; we keep the best few.
"""

from sqlalchemy import select

from app import vector_store
from app.config import settings
from app.models import Chunk

NO_RESULTS_ANSWER = (
    "I couldn't find relevant text in the uploaded documents. "
    "Try rephrasing the question or uploading a text-based PDF."
)


def answer_question(session, qdrant, embedder, llm, question, document_id=None, reranker=None):
    """Returns {"answer": text, "sources": [the chunks that were used]}."""
    sources = retrieve(session, qdrant, embedder, question, document_id, reranker)

    # Nothing found: don't waste an LLM call.
    if not sources:
        return {"answer": NO_RESULTS_ANSWER, "sources": []}

    context = build_context(sources)
    answer = llm.generate(question, context)
    return {"answer": answer, "sources": sources}


def retrieve(session, qdrant, embedder, question, document_id=None, reranker=None):
    """Find the best chunks for a question. Returns a list of 'source' dicts."""
    # With a reranker we fetch many candidates (it will pick the best ones).
    # Without one, we fetch only as many as we will finally use.
    how_many = settings.retrieval_candidates if reranker else settings.retrieval_top_k

    hits = search_qdrant(qdrant, embedder, question, document_id, how_many)  # step 1
    sources = load_chunk_text(session, hits)                                  # step 2
    if reranker:
        sources = rerank(reranker, question, sources)                         # step 3

    return sources[: settings.retrieval_top_k]


def search_qdrant(qdrant, embedder, question, document_id, how_many):
    """Step 1: turn the question into a vector and find the closest chunks."""
    vector_store.ensure_collection(qdrant)
    question_vector = embedder.embed_query(question)
    return vector_store.search(
        qdrant, question_vector, limit=how_many, document_id=document_id
    )


def load_chunk_text(session, hits):
    """Step 2: attach the real text (from Postgres) to each Qdrant hit."""
    if not hits:
        return []

    chunk_ids = [str(hit.id) for hit in hits]
    rows = session.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids)))
    chunk_by_id = {chunk.id: chunk for chunk in rows}

    sources = []
    for hit in hits:
        chunk = chunk_by_id.get(str(hit.id))
        if chunk is None:
            continue  # the vector exists but its row was deleted; ignore it
        sources.append(
            {
                "document_id": chunk.document_id,
                "filename": hit.payload["filename"],
                "page": chunk.page,
                "chunk_id": chunk.id,
                "score": round(hit.score, 4),
                "excerpt": chunk.text,
            }
        )
    return sources


def rerank(reranker, question, sources):
    """Step 3: re-score every candidate against the question, best first.

    The score is replaced by the reranker's score (higher is better;
    it can be negative, only the order matters).
    """
    if not sources:
        return sources

    texts = [source["excerpt"] for source in sources]
    scores = reranker.score(question, texts)
    for source, score in zip(sources, scores):
        source["score"] = round(score, 4)

    return sorted(sources, key=lambda source: source["score"], reverse=True)


def build_context(sources):
    """Put the chunks into one text block, each labelled so the LLM can cite it."""
    parts = []
    for number, source in enumerate(sources, start=1):
        label = f"[Source {number}: {source['filename']}, page {source['page']}]"
        passage = f"<document_passage>\n{source['excerpt']}\n</document_passage>"
        parts.append(f"{label}\n{passage}")
    return "\n\n".join(parts)
