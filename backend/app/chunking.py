"""Turn a PDF file into small pieces of text (chunks) that can be searched."""

import hashlib
import uuid

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

# A fixed random-looking number. It only has to stay the same forever,
# so that the same input always produces the same chunk id.
CHUNK_NAMESPACE = uuid.UUID("6f1d3c2e-8a54-4b7e-9d0a-2c5e7f1a9b34")


def hash_file(pdf_bytes):
    """A fingerprint of the file. The same file always gives the same fingerprint."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def chunk_id(file_hash, chunk_index):
    """Id for one chunk, made from the file fingerprint + the chunk's position.

    Same file + same position = same id. So uploading a file twice
    overwrites its chunks instead of creating duplicates.
    """
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{file_hash}:{chunk_index}"))


def extract_pages(pdf_bytes):
    """Read the PDF. Returns (pages, total_page_count).

    pages is a list like [(1, "text of page 1"), (2, "text of page 2")].
    Pages with no text (for example scanned images) are left out.
    """
    pages = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for index, page in enumerate(pdf):
            text = page.get_text().strip()
            if text:
                pages.append((index + 1, text))  # page numbers start at 1
        total_pages = len(pdf)
    return pages, total_pages


def split_pages(pages):
    """Cut each page's text into overlapping chunks.

    Returns a list like [(1, "chunk text"), (1, "next chunk"), (2, "...")]:
    each chunk remembers which page it came from, so answers can cite pages.
    The overlap means a sentence cut at a chunk edge still appears whole in one chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = []
    for page_number, page_text in pages:
        for chunk_text in splitter.split_text(page_text):
            chunks.append((page_number, chunk_text))
    return chunks
