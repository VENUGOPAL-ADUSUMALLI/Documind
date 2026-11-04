from google import genai
from google.genai import types
from openai import OpenAI

from app.config import settings

# The APIs accept many texts per request, but only up to a limit.
# We send 100 at a time, which is safe for both OpenAI and Gemini.
BATCH_SIZE = 100


class OpenAIEmbedder:
    """Turns text into vectors with an OpenAI embedding model."""

    def __init__(self, api_key=None):
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY was not found. Please check your .env file.")
        self.client = OpenAI(api_key=api_key)

    def _embed(self, texts):
        vectors = []
        # Go through the texts 100 at a time: texts[0:100], texts[100:200], ...
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            response = self.client.embeddings.create(
                model=settings.openai_embedding_model,
                input=batch,
                dimensions=settings.embedding_dim,
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_documents(self, texts):
        """Vectors for many chunks (used when uploading)."""
        return self._embed(texts)

    def embed_query(self, text):
        """The vector for one question (used when asking)."""
        return self._embed([text])[0]


class GeminiEmbedder:
    """Turns text into vectors with the Gemini embedding model."""

    def __init__(self, api_key=None):
        api_key = api_key or settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY was not found. Please check your .env file.")
        self.client = genai.Client(api_key=api_key)

    def _embed(self, texts, task_type):
        vectors = []
        # Go through the texts 100 at a time: texts[0:100], texts[100:200], ...
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            response = self.client.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dim,
                ),
            )
            vectors.extend(item.values for item in response.embeddings)
        return vectors

    def embed_documents(self, texts):
        """Vectors for many chunks (used when uploading)."""
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        """The vector for one question (used when asking)."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def get_embedder():
    """Return the embedder chosen by EMBEDDING_PROVIDER (default: openai)."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder()
    if settings.embedding_provider == "gemini":
        return GeminiEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
