from sentence_transformers import CrossEncoder

from app.config import settings


class CrossEncoderReranker:
    """Scores (question, passage) pairs together, which catches matches that
    embedding similarity misses (e.g. 'teammate' vs 'Student Name(s)')."""

    def __init__(self, model_name=None):
        self.model_name = model_name or settings.rerank_model
        self._model = None  # loaded on first use so the API starts quickly

    def score(self, question, texts):
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(question, text) for text in texts])
        return [float(score) for score in scores]
