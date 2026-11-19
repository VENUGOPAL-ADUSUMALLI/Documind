
import hashlib
import uuid

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

CHUNK_NAMESPACE = uuid.UUID("6f1d3c2e-8a54-4b7e-9d0a-2c5e7f1a9b34")


def hash_file(pdf_bytes):
    return hashlib.sha256(pdf_bytes).hexdigest()


def chunk_id(file_hash, chunk_index):
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{file_hash}:{chunk_index}"))


def extract_pages(pdf_bytes):
    pages = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for index, page in enumerate(pdf):
            text = page.get_text().strip()
            if text:
                pages.append((index + 1, text))
        total_pages = len(pdf)
    return pages, total_pages


def split_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = []
    for page_number, page_text in pages:
        for chunk_text in splitter.split_text(page_text):
            chunks.append((page_number, chunk_text))
    return chunks
