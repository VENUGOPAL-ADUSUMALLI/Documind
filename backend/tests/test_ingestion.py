import hashlib

import pymupdf
import pytest
from sqlalchemy import select

from app import models, vector_store
from app.config import settings
from app.db import Base, make_engine, make_session_factory
from app.ingestion import ingest_pdf
from app.models import Chunk, Document


class FakeEmbedder:
    """Bag-of-words hashing: texts sharing words get similar vectors. No API needed."""

    def _vec(self, text):
        vector = [0.0] * settings.embedding_dim
        for word in text.lower().split():
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % settings.embedding_dim
            vector[slot] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def make_pdf(pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def env():
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = make_session_factory(engine)()
    qdrant = vector_store.make_qdrant_client(":memory:")
    return session, qdrant, FakeEmbedder()


PDF = make_pdf(
    [
        "Photosynthesis converts sunlight into chemical energy inside chloroplasts.",
        "The French Revolution began in 1789 and ended the monarchy.",
    ]
)


def test_ingest_stores_document_chunks_and_vectors(env):
    session, qdrant, embedder = env
    document = ingest_pdf(session, qdrant, embedder, "bio.pdf", PDF)

    assert document.status == models.DONE
    assert document.page_count == 2
    chunks = session.scalars(select(Chunk)).all()
    assert len(chunks) == 2
    assert qdrant.count(settings.qdrant_collection).count == 2


def test_search_returns_relevant_page_with_metadata(env):
    session, qdrant, embedder = env
    document = ingest_pdf(session, qdrant, embedder, "bio.pdf", PDF)

    hits = vector_store.search(qdrant, embedder.embed_query("chloroplasts sunlight energy"), limit=1)
    assert hits[0].payload["page"] == 1
    assert hits[0].payload["filename"] == "bio.pdf"

    # Metadata filter: another document id matches nothing.
    none = vector_store.search(qdrant, embedder.embed_query("sunlight"), document_id="other")
    assert none == []
    some = vector_store.search(qdrant, embedder.embed_query("sunlight"), document_id=document.id)
    assert len(some) == 2


def test_reupload_same_file_is_idempotent(env):
    session, qdrant, embedder = env
    first = ingest_pdf(session, qdrant, embedder, "bio.pdf", PDF)
    second = ingest_pdf(session, qdrant, embedder, "copy-of-bio.pdf", PDF)

    assert first.id == second.id
    assert len(session.scalars(select(Document)).all()) == 1
    assert qdrant.count(settings.qdrant_collection).count == 2


def test_failed_ingestion_is_recorded_and_retry_works(env):
    session, qdrant, embedder = env

    class Boom(FakeEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("embedding API down")

    with pytest.raises(RuntimeError):
        ingest_pdf(session, qdrant, Boom(), "bio.pdf", PDF)
    document = session.scalar(select(Document))
    assert document.status == models.FAILED
    assert "embedding API down" in document.error

    retried = ingest_pdf(session, qdrant, embedder, "bio.pdf", PDF)
    assert retried.id == document.id
    assert retried.status == models.DONE
    assert qdrant.count(settings.qdrant_collection).count == 2


def test_pdf_without_text_fails_cleanly(env):
    session, qdrant, embedder = env
    with pytest.raises(ValueError, match="No extractable text"):
        ingest_pdf(session, qdrant, embedder, "blank.pdf", make_pdf([""]))
    assert session.scalar(select(Document)).status == models.FAILED
