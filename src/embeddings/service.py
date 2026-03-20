from __future__ import annotations

from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_MODEL_CACHE_LOCK = Lock()


def _get_or_create_model(model_name: str) -> SentenceTransformer:
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = model
        return model


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = _get_or_create_model(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode_texts(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=normalize_embeddings)
        return np.array(vectors, dtype="float32")