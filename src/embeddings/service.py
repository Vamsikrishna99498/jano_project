from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Final

import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_MODEL_CACHE_LOCK = Lock()
_MAX_EMBED_CACHE_SIZE: Final[int] = 2000


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
        self._cache: OrderedDict[tuple[str, bool], np.ndarray] = OrderedDict()
        self._cache_lock = Lock()

    @property
    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Embedding model did not report vector dimension.")
        return int(dimension)

    def encode_texts(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=normalize_embeddings)
        return np.array(vectors, dtype="float32")

    def encode_text_cached(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        key = (text, normalize_embeddings)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached.copy()

        vector = self.encode_texts([text], normalize_embeddings=normalize_embeddings)[0]
        with self._cache_lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            if len(self._cache) > _MAX_EMBED_CACHE_SIZE:
                self._cache.popitem(last=False)
        return vector.copy()

    def encode_texts_cached(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        out: list[np.ndarray | None] = [None] * len(texts)
        missing_by_text: dict[str, list[int]] = {}

        with self._cache_lock:
            for idx, text in enumerate(texts):
                key = (text, normalize_embeddings)
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    out[idx] = cached.copy()
                else:
                    missing_by_text.setdefault(text, []).append(idx)

        if missing_by_text:
            missing_texts = list(missing_by_text.keys())
            missing_vectors = self.encode_texts(missing_texts, normalize_embeddings=normalize_embeddings)

            with self._cache_lock:
                for text, vector in zip(missing_texts, missing_vectors):
                    key = (text, normalize_embeddings)
                    vector32 = np.array(vector, dtype="float32")
                    self._cache[key] = vector32
                    self._cache.move_to_end(key)
                    if len(self._cache) > _MAX_EMBED_CACHE_SIZE:
                        self._cache.popitem(last=False)

                    for idx in missing_by_text[text]:
                        out[idx] = vector32.copy()

        stacked = np.vstack([v for v in out if v is not None]).astype("float32")
        return stacked