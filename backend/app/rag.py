
from sqlalchemy import select

from app import citations, vector_store
from app.config import settings
from app.models import Chunk

NO_RESULTS_ANSWER = (
    "I couldn't find relevant text in the uploaded documents. "
    "Try rephrasing the question or uploading a text-based PDF."
)


def answer_question(session, qdrant, embedder, llm, question, document_id=None, reranker=None):
    sources = retrieve(session, qdrant, embedder, question, document_id, reranker)

    if not sources:
        return {"answer": NO_RESULTS_ANSWER, "sources": [], "invalid_citations": []}

    passages = build_passages(session, sources)
    answer = llm.generate(question, build_context(passages))
    answer, invalid_citations = citations.check_citations(answer, passages)
    return {"answer": answer, "sources": passages, "invalid_citations": invalid_citations}


def retrieve(session, qdrant, embedder, question, document_id=None, reranker=None):
    how_many = settings.retrieval_candidates if reranker else settings.retrieval_top_k
    hits = search_qdrant(qdrant, embedder, question, document_id, how_many)  # step 1
    sources = load_chunk_text(session, hits)                                  # step 2
    if reranker:
        sources = rerank(reranker, question, sources)                         # step 3

    return sources[: settings.retrieval_top_k]


def search_qdrant(qdrant, embedder, question, document_id, how_many):
    vector_store.ensure_collection(qdrant)
    question_vector = embedder.embed_query(question)
    return vector_store.search(
        qdrant, question_vector, question, limit=how_many, document_id=document_id
    )


def load_chunk_text(session, hits):
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
                "chunk_index": chunk.chunk_index,
                "score": round(hit.score, 4),
                "excerpt": chunk.text,
            }
        )
    return sources


def rerank(reranker, question, sources):
    if not sources:
        return sources

    texts = [source["excerpt"] for source in sources]
    scores = reranker.score(question, texts)
    for source, score in zip(sources, scores):
        source["score"] = round(score, 4)

    return sorted(sources, key=lambda source: source["score"], reverse=True)


def build_passages(session, sources):
    windows = merge_windows(sources)

    passages = []
    total_chars = 0
    for window in windows:
        text = load_window_text(session, window)
        if not text:
            continue
        if passages and total_chars + len(text) > settings.context_char_budget:
            continue
        total_chars += len(text)
        passages.append(
            {
                "document_id": window["document_id"],
                "filename": window["filename"],
                "page": window["page"],
                "chunk_id": window["chunk_id"],
                "score": window["score"],
                "excerpt": text,
            }
        )
    return passages


def merge_windows(sources):
    reach = settings.context_window
    windows = []
    for source in sources:
        lo = source["chunk_index"] - reach
        hi = source["chunk_index"] + reach

        for window in windows:
            same_page = (
                window["document_id"] == source["document_id"]
                and window["page"] == source["page"]
            )
            if same_page and lo <= window["hi"] and hi >= window["lo"]:
                window["lo"] = min(window["lo"], lo)
                window["hi"] = max(window["hi"], hi)
                break
        else:
            windows.append(
                {
                    "document_id": source["document_id"],
                    "filename": source["filename"],
                    "page": source["page"],
                    "chunk_id": source["chunk_id"],
                    "score": source["score"],
                    "lo": lo,
                    "hi": hi,
                }
            )
    return windows


def load_window_text(session, window):
    rows = session.scalars(
        select(Chunk)
        .where(
            Chunk.document_id == window["document_id"],
            Chunk.page == window["page"],
            Chunk.chunk_index.between(window["lo"], window["hi"]),
        )
        .order_by(Chunk.chunk_index)
    )
    return join_chunks([row.text for row in rows])


def join_chunks(texts):
    if not texts:
        return ""
    result = texts[0]
    for text in texts[1:]:
        shared = shared_length(result, text)
        result += text[shared:] if shared else "\n" + text
    return result


MIN_SHARED = 10


def shared_length(first, second):
    longest = min(len(first), len(second), settings.chunk_overlap + 50)
    for length in range(longest, MIN_SHARED - 1, -1):
        if first.endswith(second[:length]):
            return length
    return 0


def build_context(sources):
    parts = []
    for number, source in enumerate(sources, start=1):
        label = f"[Source {number}: {source['filename']}, page {source['page']}]"
        passage = f"<document_passage>\n{source['excerpt']}\n</document_passage>"
        parts.append(f"{label}\n{passage}")
    return "\n\n".join(parts)
