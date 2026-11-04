import pytest
from fastapi.testclient import TestClient

from app import vector_store
from app.db import Base, make_engine, make_session_factory
from app.deps import get_chat_model, get_db, get_qdrant, get_reranker, get_text_embedder
from app.main import app
from tests.test_ingestion import FakeEmbedder, make_pdf


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.answer = "Photosynthesis happens in chloroplasts [bio.pdf, page 1]."

    def generate(self, question, context):
        self.calls.append((question, context))
        return self.answer


@pytest.fixture
def client(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    qdrant = vector_store.make_qdrant_client(":memory:")
    llm = FakeLLM()

    def override_db():
        with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_qdrant] = lambda: qdrant
    app.dependency_overrides[get_text_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_chat_model] = lambda: llm
    app.dependency_overrides[get_reranker] = lambda: None

    # No `with` block, so the startup hook (which targets the real database) is skipped.
    test_client = TestClient(app)
    test_client.llm = llm
    yield test_client
    app.dependency_overrides.clear()


PDF = make_pdf(
    [
        "Photosynthesis converts sunlight into chemical energy inside chloroplasts.",
        "The French Revolution began in 1789 and ended the monarchy.",
    ]
)


def upload(client, data=PDF, name="bio.pdf"):
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_list_get_delete(client):
    created = upload(client)
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "done" and body["page_count"] == 2

    assert [d["id"] for d in client.get("/documents").json()] == [body["id"]]
    assert client.get(f"/documents/{body['id']}").json()["filename"] == "bio.pdf"

    assert client.delete(f"/documents/{body['id']}").status_code == 204
    assert client.get(f"/documents/{body['id']}").status_code == 404
    assert client.get("/documents").json() == []


def test_upload_rejects_non_pdf(client):
    response = upload(client, data=b"just some text", name="notes.txt")
    assert response.status_code == 400


def test_upload_pdf_without_text_returns_422(client):
    response = upload(client, data=make_pdf([""]), name="blank.pdf")
    assert response.status_code == 422
    assert "No extractable text" in response.json()["detail"]


def test_ask_returns_answer_with_sources(client):
    upload(client)
    response = client.post("/ask", json={"question": "Where does photosynthesis happen? chloroplasts"})
    assert response.status_code == 200
    body = response.json()

    assert "chloroplasts" in body["answer"]
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["filename"] == "bio.pdf"
    assert "[Source 1: bio.pdf, page 1]" in client.llm.calls[0][1]


def test_ask_scoped_to_one_document(client):
    first = upload(client).json()
    other = upload(client, data=make_pdf(["Volcanoes erupt molten rock."]), name="geo.pdf").json()

    response = client.post(
        "/ask", json={"question": "volcanoes rock", "document_id": other["id"]}
    )
    assert {s["document_id"] for s in response.json()["sources"]} == {other["id"]}
    assert first["id"] != other["id"]


def test_ask_with_nothing_indexed_skips_the_llm(client):
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert client.llm.calls == []


def test_ask_validation_and_unknown_document(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "document_id": "nope"}).status_code == 404


class FakeReranker:
    """Ranks passages that mention 'student' first, whatever vector search said."""

    def score(self, question, texts):
        return [10.0 if "student" in text.lower() else 0.0 for text in texts]


def test_reranker_rescues_a_chunk_that_vector_search_ranks_low(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "retrieval_top_k", 1)
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()

    upload(client, data=make_pdf(["Student Name(s): Ana Rao, Ben Cho", "Teammate names are not written anywhere in plain words here."]))
    response = client.post("/ask", json={"question": "what are the teammate names?"})

    sources = response.json()["sources"]
    assert len(sources) == 1
    assert "Student Name(s)" in sources[0]["excerpt"]


def test_invented_citation_is_removed_and_reported(client):
    upload(client)
    client.llm.answer = (
        "Chloroplasts make energy [bio.pdf, page 1]. "
        "The revolution began in 1789 [bio.pdf, page 9]."
    )
    body = client.post("/ask", json={"question": "chloroplasts energy"}).json()

    assert "[bio.pdf, page 1]" in body["answer"]
    assert "page 9" not in body["answer"]
    assert body["invalid_citations"] == ["bio.pdf, page 9"]


def test_valid_answer_has_no_invalid_citations(client):
    upload(client)
    body = client.post("/ask", json={"question": "chloroplasts energy"}).json()
    assert body["invalid_citations"] == []
