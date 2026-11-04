import pytest
from sqlalchemy import select

from app import rag
from app.config import settings
from app.db import Base, make_engine, make_session_factory
from app.ingestion import ingest_pdf
from app.models import Chunk
from app.vector_store import make_qdrant_client
from tests.test_ingestion import FakeEmbedder, make_pdf


def source(chunk_index, page=1, document_id="d1", score=1.0):
    return {
        "document_id": document_id, "filename": "a.pdf", "page": page,
        "chunk_id": f"c{chunk_index}", "chunk_index": chunk_index, "score": score,
    }


# ---- joining chunks -------------------------------------------------------

def test_join_keeps_shared_overlap_only_once():
    first = "The system stores keys. The server never sees the plaintext"
    second = "server never sees the plaintext of any file. Revocation is fast."
    assert rag.join_chunks([first, second]) == (
        "The system stores keys. The server never sees the plaintext of any file. Revocation is fast."
    )


def test_join_without_overlap_uses_a_newline():
    assert rag.join_chunks(["alpha beta", "gamma delta"]) == "alpha beta\ngamma delta"


def test_join_of_nothing_is_empty():
    assert rag.join_chunks([]) == ""


# ---- merging windows ------------------------------------------------------

def test_window_zero_is_just_the_matched_chunk(monkeypatch):
    monkeypatch.setattr(settings, "context_window", 0)
    windows = rag.merge_windows([source(4), source(9)])
    assert [(w["lo"], w["hi"]) for w in windows] == [(4, 4), (9, 9)]


def test_window_one_adds_a_neighbour_each_side(monkeypatch):
    monkeypatch.setattr(settings, "context_window", 1)
    windows = rag.merge_windows([source(4)])
    assert [(w["lo"], w["hi"]) for w in windows] == [(3, 5)]


def test_overlapping_windows_on_one_page_are_merged(monkeypatch):
    monkeypatch.setattr(settings, "context_window", 1)
    windows = rag.merge_windows([source(4, score=9), source(5, score=5), source(20)])
    assert [(w["lo"], w["hi"]) for w in windows] == [(3, 6), (19, 21)]
    assert windows[0]["score"] == 9  # keeps the best match's score


def test_windows_on_different_pages_are_never_merged(monkeypatch):
    monkeypatch.setattr(settings, "context_window", 1)
    windows = rag.merge_windows([source(4, page=1), source(5, page=2)])
    assert len(windows) == 2


# ---- against a real (in-memory) database ----------------------------------

@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 120)
    monkeypatch.setattr(settings, "chunk_overlap", 20)
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = make_session_factory(engine)()
    qdrant = make_qdrant_client(":memory:")
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(1, 25))
    ingest_pdf(session, qdrant, FakeEmbedder(), "long.pdf", make_pdf([text, "Second page content here."]))
    return session


def matched(session, index):
    chunk = session.scalar(select(Chunk).where(Chunk.page == 1, Chunk.chunk_index == index))
    return {
        "document_id": chunk.document_id, "filename": "long.pdf", "page": 1,
        "chunk_id": chunk.id, "chunk_index": index, "score": 1.0,
    }


def test_window_adds_neighbour_text_and_stays_on_the_page(small_chunks, monkeypatch):
    session = small_chunks
    monkeypatch.setattr(settings, "context_window", 0)
    alone = rag.build_passages(session, [matched(session, 2)])[0]["excerpt"]

    monkeypatch.setattr(settings, "context_window", 1)
    widened = rag.build_passages(session, [matched(session, 2)])[0]["excerpt"]

    assert len(widened) > len(alone)
    assert alone[:30] in widened
    assert "Second page" not in widened


def test_window_text_has_no_repeated_overlap(small_chunks, monkeypatch):
    session = small_chunks
    monkeypatch.setattr(settings, "context_window", 2)
    text = rag.build_passages(session, [matched(session, 3)])[0]["excerpt"]
    # Each sentence number should appear once even though chunks overlap.
    assert text.count("Sentence number 5 ") <= 1


def test_budget_limits_how_many_passages_are_sent(small_chunks, monkeypatch):
    session = small_chunks
    monkeypatch.setattr(settings, "context_window", 1)
    monkeypatch.setattr(settings, "context_char_budget", 300)
    sources = [matched(session, 1), matched(session, 4), matched(session, 7)]
    passages = rag.build_passages(session, sources)
    assert 1 <= len(passages) < 3
    assert passages[0]["chunk_id"] == sources[0]["chunk_id"]  # best match always kept
