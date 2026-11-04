"""Answering a question:  question -> find relevant chunks -> ask the LLM -> answer.

Finding the chunks has three steps:
  1. SEARCH   Qdrant returns many candidate chunks (hybrid: meaning + keywords, merged).
  2. LOAD     Qdrant only knows ids and scores, so we load the real text from Postgres.
  3. RERANK   A second, more careful model re-sorts the candidates; we keep the best few.
"""

from sqlalchemy import select

from app import citations, vector_store
from app.config import settings
from app.models import Chunk

NO_RESULTS_ANSWER = (
    "I couldn't find relevant text in the uploaded documents. "
    "Try rephrasing the question or uploading a text-based PDF."
)


def answer_question(session, qdrant, embedder, llm, question, document_id=None, reranker=None):
    """Returns the answer, the chunks that were used, and any citations that were removed."""
    sources = retrieve(session, qdrant, embedder, question, document_id, reranker)

    # Nothing found: don't waste an LLM call.
    if not sources:
        return {"answer": NO_RESULTS_ANSWER, "sources": [], "invalid_citations": []}

    # Give the model the matched chunks plus their neighbours (see build_passages).
    passages = build_passages(session, sources)
    answer = llm.generate(question, build_context(passages))

    # The model can cite pages it was never shown. Remove those and report them.
    answer, invalid_citations = citations.check_citations(answer, passages)
    return {"answer": answer, "sources": passages, "invalid_citations": invalid_citations}


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
    """Step 1: search by meaning AND by keywords (see settings.retrieval_mode)."""
    vector_store.ensure_collection(qdrant)
    question_vector = embedder.embed_query(question)
    return vector_store.search(
        qdrant, question_vector, question, limit=how_many, document_id=document_id
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
                "chunk_index": chunk.chunk_index,
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


def build_passages(session, sources):
    """Widen each matched chunk into a passage: the chunk plus its neighbours.

    Why: a small chunk matches well but may cut an answer in half. The chunks
    just before and after it (same page) usually hold the rest.

    Steps:
      1. Turn each matched chunk into a window of chunk positions [lo, hi].
      2. Merge windows that overlap, so the same text is not sent twice.
      3. Load the text, best match first, adding each passage that still fits the
         character budget (a passage that is too big is skipped, smaller ones may follow).
    """
    windows = merge_windows(sources)

    passages = []
    total_chars = 0
    for window in windows:
        text = load_window_text(session, window)
        if not text:
            continue  # the chunks are gone (for example the document was deleted)
        if passages and total_chars + len(text) > settings.context_char_budget:
            continue  # does not fit; a later, smaller passage still might (best one is always kept)
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
    """One window per matched chunk: chunk positions [lo, hi] on one page.

    Windows on the same page that overlap are merged into one.
    The windows keep the order of the sources (best first).
    """
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
    """Read the chunks in a window from Postgres and join them into one text."""
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
    """Join consecutive chunks into one text.

    Neighbouring chunks share a few repeated characters (the chunk overlap).
    We detect that repeated part and keep it only once.
    """
    if not texts:
        return ""
    result = texts[0]
    for text in texts[1:]:
        shared = shared_length(result, text)
        result += text[shared:] if shared else "\n" + text
    return result


MIN_SHARED = 10  # ignore accidental matches shorter than this


def shared_length(first, second):
    """How many characters at the end of `first` are repeated at the start of `second`."""
    longest = min(len(first), len(second), settings.chunk_overlap + 50)
    for length in range(longest, MIN_SHARED - 1, -1):
        if first.endswith(second[:length]):
            return length
    return 0


def build_context(sources):
    """Put the passages into one text block, each labelled so the LLM can cite it."""
    parts = []
    for number, source in enumerate(sources, start=1):
        label = f"[Source {number}: {source['filename']}, page {source['page']}]"
        passage = f"<document_passage>\n{source['excerpt']}\n</document_passage>"
        parts.append(f"{label}\n{passage}")
    return "\n\n".join(parts)
