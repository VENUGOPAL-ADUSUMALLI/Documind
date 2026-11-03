"""The web API. Each function below is one URL the frontend can call.

  POST   /documents        upload a PDF
  GET    /documents        list all documents
  GET    /documents/{id}   one document (to check its status)
  DELETE /documents/{id}   delete a document
  POST   /ask              ask a question
  GET    /health           is the server alive?

The endpoints are plain `def` (not `async def`) on purpose: the database and
AI calls are blocking, and FastAPI runs plain functions in worker threads,
so one slow request does not freeze the others.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy import select

from app import rag, vector_store
from app.config import settings
from app.db import Base
from app.deps import (
    get_chat_model,
    get_db,
    get_engine,
    get_qdrant,
    get_reranker,
    get_text_embedder,
)
from app.ingestion import ingest_pdf
from app.models import Document
from app.schemas import AskRequest, AskResponse, DocumentOut


@asynccontextmanager
async def lifespan(app):
    """Runs once when the server starts: create the database tables if missing."""
    Base.metadata.create_all(get_engine())
    yield


app = FastAPI(title="DocuMind", lifespan=lifespan)


def get_document_or_404(session, document_id):
    """Fetch a document, or answer the request with a 404 'not found' error."""
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found.")
    return document


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    session=Depends(get_db),
    qdrant=Depends(get_qdrant),
    embedder=Depends(get_text_embedder),
):
    # Read the file, but never more than the size limit (+1 byte to detect "too big").
    pdf_bytes = file.file.read(settings.max_upload_bytes + 1)
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise HTTPException(413, "File is too large.")

    # Every real PDF starts with the bytes "%PDF".
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(400, "Only PDF files are supported.")

    try:
        return ingest_pdf(session, qdrant, embedder, file.filename or "upload.pdf", pdf_bytes)
    except ValueError as error:
        # A problem with the file itself (for example no text in it).
        raise HTTPException(422, str(error))
    except Exception:
        # Something else broke (API down, etc). The document row is already
        # marked 'failed' with the real error, so we show a short message here.
        raise HTTPException(502, "Ingestion failed. Check the document status for details.")


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(session=Depends(get_db)):
    newest_first = select(Document).order_by(Document.created_at.desc())
    return session.scalars(newest_first).all()


@app.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, session=Depends(get_db)):
    return get_document_or_404(session, document_id)


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    session=Depends(get_db),
    qdrant=Depends(get_qdrant),
):
    document = get_document_or_404(session, document_id)

    # Remove the vectors from Qdrant, then the rows from Postgres.
    vector_store.ensure_collection(qdrant)
    vector_store.delete_document_points(qdrant, document_id)
    session.delete(document)  # its chunks are deleted with it
    session.commit()


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    session=Depends(get_db),
    qdrant=Depends(get_qdrant),
    embedder=Depends(get_text_embedder),
    llm=Depends(get_chat_model),
    reranker=Depends(get_reranker),
):
    question = request.question.strip()
    if not question:
        raise HTTPException(422, "Please enter a question.")

    # If the user limited the search to one document, make sure it exists.
    if request.document_id:
        get_document_or_404(session, request.document_id)

    return rag.answer_question(
        session, qdrant, embedder, llm, question, request.document_id, reranker
    )
