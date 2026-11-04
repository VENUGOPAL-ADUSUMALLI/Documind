"""Rebuild the Qdrant index from what is stored in Postgres.

Qdrant is only a search index; Postgres holds the real chunk text. So whenever the
index layout or the embedding model changes, we can rebuild it without the PDFs:

    cd backend && python -m app.reindex
"""

from sqlalchemy import select

from app import models, vector_store
from app.deps import get_qdrant, get_session_factory, get_text_embedder
from app.ingestion import make_points
from app.models import Document


def reindex_all(session, qdrant, embedder):
    """Re-embed and re-save every finished document. Returns how many were done."""
    vector_store.ensure_collection(qdrant)

    documents = session.scalars(select(Document).where(Document.status == models.DONE)).all()
    for document in documents:
        chunks = sorted(document.chunks, key=lambda chunk: chunk.chunk_index)
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        vector_store.upsert_chunks(qdrant, make_points(document, chunks, vectors))
        print(f"reindexed {document.filename}: {len(chunks)} chunks")
    return len(documents)


if __name__ == "__main__":
    with get_session_factory()() as session:
        count = reindex_all(session, get_qdrant(), get_text_embedder())
    print(f"done: {count} documents")
