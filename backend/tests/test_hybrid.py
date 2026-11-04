from sqlalchemy import select

from app import sparse, vector_store
from app.config import settings
from app.db import Base, make_engine, make_session_factory
from app.ingestion import ingest_pdf
from app.models import Chunk
from app.reindex import reindex_all
from tests.test_ingestion import FakeEmbedder, make_pdf


class UselessEmbedder(FakeEmbedder):
    """Gives every text the same vector, so 'meaning' search cannot tell chunks apart."""

    def _vec(self, text):
        return [1.0] + [0.0] * (settings.embedding_dim - 1)


def make_env():
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    return make_session_factory(engine)(), vector_store.make_qdrant_client(":memory:")


PDF = make_pdf(
    [
        "General notes about the project schedule and the weekly meetings.",
        "Error code ERR_4021 means the encryption key has expired.",
        "Budget planning and office supplies for the next quarter.",
    ]
)


def test_sparse_query_has_one_entry_per_distinct_word():
    query = sparse.encode_query("Key key KEY expired")
    assert len(query.indices) == 2
    assert query.values == [1.0, 1.0]


def test_sparse_word_ids_are_stable():
    assert sparse.word_id("encryption") == sparse.word_id("encryption")
    assert sparse.encode_document("a b a").indices == sparse.encode_document("a b a").indices


def test_repeated_words_weigh_more_but_with_diminishing_returns():
    once = sparse.encode_document("alpha beta").values[0]
    twice = sparse.encode_document("alpha alpha beta").values[0]
    ten_times = sparse.encode_document("alpha " * 10 + "beta").values[0]
    assert once < twice < ten_times
    assert (twice - once) > (ten_times - twice) / 8  # gains shrink as repeats grow


def test_keyword_search_finds_exact_term_when_meaning_search_cannot():
    session, qdrant = make_env()
    embedder = UselessEmbedder()
    ingest_pdf(session, qdrant, embedder, "notes.pdf", PDF)

    query = "ERR_4021"
    vector = embedder.embed_query(query)

    for mode in ("bm25", "hybrid"):
        hits = vector_store.search(qdrant, vector, query, limit=1, mode=mode)
        assert hits[0].payload["page"] == 2, mode


def test_dense_mode_ignores_keywords_and_bm25_mode_ignores_meaning():
    session, qdrant = make_env()
    embedder = FakeEmbedder()
    ingest_pdf(session, qdrant, embedder, "notes.pdf", PDF)

    # A query with no usable words has no keyword results at all.
    assert vector_store.search(qdrant, embedder.embed_query("???"), "???", mode="bm25") == []
    # Dense mode still returns something for the same query.
    assert vector_store.search(qdrant, embedder.embed_query("???"), "???", mode="dense")


def test_hybrid_respects_document_filter():
    session, qdrant = make_env()
    embedder = FakeEmbedder()
    first = ingest_pdf(session, qdrant, embedder, "notes.pdf", PDF)
    ingest_pdf(session, qdrant, embedder, "other.pdf", make_pdf(["ERR_4021 appears here too."]))

    hits = vector_store.search(
        qdrant, embedder.embed_query("ERR_4021"), "ERR_4021", limit=10, document_id=first.id, mode="hybrid"
    )
    assert hits and {hit.payload["document_id"] for hit in hits} == {first.id}


def test_reindex_rebuilds_the_search_index_from_postgres():
    session, qdrant = make_env()
    embedder = FakeEmbedder()
    ingest_pdf(session, qdrant, embedder, "notes.pdf", PDF)

    # Lose the whole index (as if the collection layout had changed).
    qdrant.delete_collection(settings.qdrant_collection)

    assert reindex_all(session, qdrant, embedder) == 1
    assert qdrant.count(settings.qdrant_collection).count == len(session.scalars(select(Chunk)).all())
    hits = vector_store.search(qdrant, embedder.embed_query("ERR_4021"), "ERR_4021", limit=1)
    assert hits[0].payload["page"] == 2
