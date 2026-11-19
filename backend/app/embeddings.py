from openai import OpenAI

from app.config import settings

BATCH_SIZE = 100


class OpenAIEmbedder:

    def __init__(self, api_key=None):
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY was not found. Please check your .env file.")
        self.client = OpenAI(api_key=api_key)

    def _embed(self, texts):
        vectors = []
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
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


def get_embedder():
    return OpenAIEmbedder()
