"""The upload pipeline:  PDF  ->  text  ->  chunks  ->  embeddings  ->  stored.

Where things are stored:
  - Postgres: the document record (with its status) and every chunk's text.
  - Qdrant:   one vector per chunk, used for searching by meaning.

The document's status shows progress:  processing  ->  done   (or failed).
"""

from sqlalchemy import select

from app import chunking, models, vector_store
from app.models import Chunk, Document

NO_TEXT_MESSAGE = (
    "No extractable text found. Please upload a text-based PDF. "
    "Scanned PDFs may require OCR."
)


def ingest_pdf(session, qdrant, embedder, filename, pdf_bytes):
    """Process one PDF and return its Document record.

    It is safe to call twice with the same file:
      - already finished  -> returns the existing record straight away
      - earlier attempt failed -> tries again
    """
    # Step 1: have we seen this exact file before?
    file_hash = chunking.hash_file(pdf_bytes)
    document = session.scalar(select(Document).where(Document.file_hash == file_hash))
    if document and document.status == models.DONE:
        return document

    # Step 2: save a record saying "processing started".
    # If the program crashes later, this record shows what happened.
    if document is None:
        document = Document(filename=filename, file_hash=file_hash)
        session.add(document)
    document.status = models.PROCESSING
    document.error = None
    session.commit()

    # Step 3: do the real work. If anything goes wrong, record the error.
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
    """Extra info stored next to a chunk's vector in Qdrant.

    It lets us filter searches (for example "only this document")
    and show the filename and page without asking Postgres.
    """
    return {
        "document_id": document.id,
        "filename": document.filename,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
    }


def make_points(document, chunks, vectors):
    """Pair each chunk with its vector, in the shape vector_store.upsert_chunks expects."""
    return [
        (chunk.id, chunk.text, vector, chunk_payload(document, chunk))
        for chunk, vector in zip(chunks, vectors)
    ]


def process_document(session, qdrant, embedder, document, pdf_bytes):
    """Extract text, split it, embed it, and save everything."""
    # Read the PDF and cut it into chunks.
    pages, page_count = chunking.extract_pages(pdf_bytes)
    if not pages:
        raise ValueError(NO_TEXT_MESSAGE)
    chunks = chunking.split_pages(pages)  # [(page_number, chunk_text), ...]

    # Turn every chunk's text into a vector (one API call per batch).
    vectors = embedder.embed_documents([text for _, text in chunks])

    # Delete chunks left over from an earlier failed attempt.
    document.chunks.clear()
    session.flush()

    # Build the rows for Postgres.
    for index, (page, text) in enumerate(chunks):
        chunk_id = chunking.chunk_id(document.file_hash, index)
        document.chunks.append(Chunk(id=chunk_id, chunk_index=index, page=page, text=text))

    # Save the vectors in Qdrant.
    vector_store.ensure_collection(qdrant)
    vector_store.upsert_chunks(qdrant, make_points(document, document.chunks, vectors))

    # Everything worked: mark the document as finished.
    document.page_count = page_count
    document.status = models.DONE
    session.commit()
