
from sqlalchemy import select

from app import chunking, models, vector_store
from app.models import Chunk, Document

NO_TEXT_MESSAGE = (
    "No extractable text found. Please upload a text-based PDF. "
    "Scanned PDFs may require OCR."
)


def ingest_pdf(session, qdrant, embedder, filename, pdf_bytes):

    file_hash = chunking.hash_file(pdf_bytes)
    document = session.scalar(select(Document).where(Document.file_hash == file_hash))
    if document and document.status == models.DONE:
        return document
    if document is None:
        document = Document(filename=filename, file_hash=file_hash)
        session.add(document)
    document.status = models.PROCESSING
    document.error = None
    session.commit()

    try:
        process_document(session, qdrant, embedder, document, pdf_bytes)
    except Exception as error:
        session.rollback()
        document.status = models.FAILED
        document.error = str(error)
        session.commit()
        raise

    return document


def chunk_payload(document, chunk):
    return {
        "document_id": document.id,
        "filename": document.filename,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
    }


def make_points(document, chunks, vectors):
    return [
        (chunk.id, chunk.text, vector, chunk_payload(document, chunk))
        for chunk, vector in zip(chunks, vectors)
    ]


def process_document(session, qdrant, embedder, document, pdf_bytes):
    pages, page_count = chunking.extract_pages(pdf_bytes)
    if not pages:
        raise ValueError(NO_TEXT_MESSAGE)
    chunks = chunking.split_pages(pages)

    vectors = embedder.embed_documents([text for _, text in chunks])
    document.chunks.clear()
    session.flush()
    for index, (page, text) in enumerate(chunks):
        chunk_id = chunking.chunk_id(document.file_hash, index)
        document.chunks.append(Chunk(id=chunk_id, chunk_index=index, page=page, text=text))
    vector_store.ensure_collection(qdrant)
    vector_store.upsert_chunks(qdrant, make_points(document, document.chunks, vectors))
    document.page_count = page_count
    document.status = models.DONE
    session.commit()
