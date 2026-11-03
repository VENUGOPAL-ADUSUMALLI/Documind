import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# Document status values.
PENDING = "pending"
PROCESSING = "processing"
DONE = "done"
FAILED = "failed"


def _now():
    return datetime.now(timezone.utc)


class Document(Base):
    """One uploaded file and where it is in the ingestion pipeline."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255))
    # SHA-256 of the file bytes; the same file uploaded twice is recognised by this.
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default=PENDING)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A piece of a document's text. The full text lives here, vectors live in Qdrant."""

    __tablename__ = "chunks"

    # Deterministic: the same file and chunk position always give the same id,
    # and the same id is used as the Qdrant point id.
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")
