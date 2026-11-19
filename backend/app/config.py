from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        extra="ignore",
        # Keep secrets (API keys) out of validation error messages and logs.
        hide_input_in_errors=True,
    )

    openai_api_key: str = ""

    # Required: there is no default, so a missing value fails at startup with a clear message.
    database_url: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    chat_model: str = "gpt-4o-mini"
    retrieval_mode: str = "hybrid"
    retrieval_top_k: int = 10
    retrieval_candidates: int = 100
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    context_window: int = 1
    context_char_budget: int = 6000
    max_upload_bytes: int = 20 * 1024 * 1024
    chunk_size: int = 800
    chunk_overlap: int = 150

    @field_validator("database_url")
    @classmethod
    def use_psycopg_driver(cls, value):
        # Hosts such as Railway give "postgresql://..." (or "postgres://..."),
        # but SQLAlchemy must be told to use the psycopg 3 driver we install.
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value[len(prefix):]
        return value

    @property
    def qdrant_collection(self):
        return f"documind_hybrid_openai_{self.embedding_dim}"


settings = Settings()
