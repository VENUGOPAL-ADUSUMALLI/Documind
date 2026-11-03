import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

import api_client  # noqa: E402

DOC = {
    "id": "d1",
    "filename": "bio.pdf",
    "status": "done",
    "error": None,
    "page_count": 2,
    "created_at": "2026-11-02T10:00:00Z",
}
ANSWER = {
    "answer": "Chloroplasts [bio.pdf, page 1].",
    "sources": [
        {"document_id": "d1", "filename": "bio.pdf", "page": 1,
         "chunk_id": "c1", "score": 0.9, "excerpt": "Photosynthesis text"}
    ],
}


def run_app():
    return AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=10).run()


def test_empty_state_asks_for_upload(monkeypatch):
    monkeypatch.setattr(api_client, "list_documents", lambda: [])
    app = run_app()
    assert not app.exception
    assert any("Upload a PDF" in info.value for info in app.info)
    assert app.chat_input[0].disabled


def test_backend_down_shows_error(monkeypatch):
    def boom():
        raise api_client.ApiError("Cannot reach the DocuMind API")

    monkeypatch.setattr(api_client, "list_documents", boom)
    app = run_app()
    assert not app.exception
    assert "Cannot reach" in app.error[0].value


def test_asking_a_question_shows_answer_and_sources(monkeypatch):
    calls = []

    def fake_ask(question, document_id=None):
        calls.append((question, document_id))
        return ANSWER

    monkeypatch.setattr(api_client, "list_documents", lambda: [DOC])
    monkeypatch.setattr(api_client, "ask", fake_ask)

    app = run_app()
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value("What happens in chloroplasts?").run()

    assert not app.exception
    assert calls == [("What happens in chloroplasts?", None)]
    texts = [m.value for m in app.markdown]
    assert any("Chloroplasts [bio.pdf, page 1]." in t for t in texts)
    assert any("Sources" in e.label for e in app.expander)


def test_scope_selection_is_sent_to_the_api(monkeypatch):
    calls = []
    monkeypatch.setattr(api_client, "list_documents", lambda: [DOC])
    monkeypatch.setattr(api_client, "ask", lambda q, document_id=None: calls.append(document_id) or ANSWER)

    app = run_app()
    app.selectbox[0].select("bio.pdf").run()
    app.chat_input[0].set_value("hello").run()
    assert calls == ["d1"]


def test_api_error_during_ask_is_shown(monkeypatch):
    def fail(question, document_id=None):
        raise api_client.ApiError("Document not found.")

    monkeypatch.setattr(api_client, "list_documents", lambda: [DOC])
    monkeypatch.setattr(api_client, "ask", fail)

    app = run_app()
    app.chat_input[0].set_value("hello").run()
    assert not app.exception
    assert any("Document not found." in e.value for e in app.error)
