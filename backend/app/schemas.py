from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    error: str | None
    page_count: int
    created_at: datetime


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None


class SourceOut(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    # Citations the model wrote that did not match any retrieved page; removed from the answer.
    invalid_citations: list[str] = []
