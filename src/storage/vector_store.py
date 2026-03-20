from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissVectorStore:
    def __init__(self, index_path: str, meta_path: str, embedding_model: str) -> None:
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = self._load_or_create_index()
        self.metadata = self._load_meta()

    def add_resume_text(self, resume_id: int, text: str) -> None:
        embedding = self.model.encode([text], normalize_embeddings=True)
        vector = np.array(embedding, dtype="float32")
        self.index.add(vector)
        self.metadata.append({"resume_id": resume_id, "preview": text[:300]})
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        qv = np.array(q, dtype="float32")
        scores, indices = self.index.search(qv, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def _load_or_create_index(self):
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatIP(self.dimension)

    def _load_meta(self) -> list[dict]:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return []

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
