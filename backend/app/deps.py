"""Shared objects the API endpoints need (database, Qdrant, AI models).

Why @lru_cache?  It makes each function run only ONCE and then remember the
result. So the database engine, the Qdrant connection and the AI clients are
created one time and reused by every request, instead of rebuilt each time.

FastAPI's Depends(...) calls these functions and hands the result to an endpoint.
"""

from functools import lru_cache

from app import embeddings
from app.config import settings
from app.db import make_engine, make_session_factory
from app.llm import OpenAIChat
from app.reranker import CrossEncoderReranker
from app.vector_store import make_qdrant_client


@lru_cache
def get_engine():
    """The connection to the database (Postgres)."""
    return make_engine()


@lru_cache
def get_session_factory():
    """A factory that makes database sessions."""
    return make_session_factory(get_engine())


def get_db():
    """Give each request its own database session, and close it afterwards.

    (No @lru_cache here on purpose: every request needs a fresh session.)
    """
    with get_session_factory()() as session:
        yield session


@lru_cache
def get_qdrant():
    """The connection to Qdrant."""
    return make_qdrant_client()


@lru_cache
def get_text_embedder():
    """Turns text into vectors (OpenAI or Gemini, set in config)."""
    return embeddings.get_embedder()


@lru_cache
def get_chat_model():
    """The LLM that writes the final answer."""
    return OpenAIChat()


@lru_cache
def get_reranker():
    """Re-sorts search results more carefully. None if turned off in config."""
    return CrossEncoderReranker() if settings.rerank_enabled else None
