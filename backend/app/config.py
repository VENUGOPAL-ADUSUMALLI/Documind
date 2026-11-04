from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """App configuration, read from environment variables or the root .env file."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )

    openai_api_key: str = ""
    gemini_api_key: str = ""

    database_url: str = "postgresql+psycopg://documind:documind@localhost:5432/documind"
    qdrant_url: str = "http://localhost:6333"

    # "openai" or "gemini"
    embedding_provider: str = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 1536

    chat_model: str = "gpt-4o-mini"
    # "hybrid" (dense + BM25 fused), "dense" (meaning only) or "bm25" (keywords only)
    retrieval_mode: str = "hybrid"
    retrieval_top_k: int = 10
    # Cast a wide net with vector search, then let the reranker pick the best few.
    retrieval_candidates: int = 100
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many neighbouring chunks (before and after) to add around each matched chunk.
    # 0 = send only the matched chunk. 1 scored best in eval/run_eval.py.
    context_window: int = 1
    # Stop adding passages once the text sent to the LLM reaches this many characters.
    context_char_budget: int = 6000
    max_upload_bytes: int = 20 * 1024 * 1024

    chunk_size: int = 800
    chunk_overlap: int = 150

    @property
    def qdrant_collection(self):
        # Vectors from different models/sizes must never share a collection.
        return f"documind_hybrid_{self.embedding_provider}_{self.embedding_dim}"


settings = Settings()
