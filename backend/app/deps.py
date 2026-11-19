
from functools import lru_cache

from app import embeddings
from app.config import settings
from app.db import make_engine, make_session_factory
from app.llm import OpenAIChat
from app.reranker import CrossEncoderReranker
from app.vector_store import make_qdrant_client


@lru_cache
def get_engine():
    return make_engine()


@lru_cache
def get_session_factory():
    return make_session_factory(get_engine())


def get_db():
    with get_session_factory()() as session:
        yield session


@lru_cache
def get_qdrant():
    return make_qdrant_client()


@lru_cache
def get_text_embedder():
    return embeddings.get_embedder()


@lru_cache
def get_chat_model():
    return OpenAIChat()


@lru_cache
def get_reranker():
    return CrossEncoderReranker() if settings.rerank_enabled else None
