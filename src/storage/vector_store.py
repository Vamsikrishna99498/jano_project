from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.embeddings.service import EmbeddingService


class FaissVectorStore:
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        embedding_model: str | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        if embedder is None and not embedding_model:
            raise ValueError("Either embedder or embedding_model must be provided.")

        self.embedder = embedder or EmbeddingService(str(embedding_model))
        self.dimension = self.embedder.dimension
        self.records = self._load_records()
        self._resume_id_to_record_idx = self._build_resume_id_map()
        self.index = self._rebuild_index()

    def add_resume_text(self, resume_id: int, text: str) -> None:
        vector = self.embedder.encode_texts([text], normalize_embeddings=True)[0]
        existing_idx = self._resume_id_to_record_idx.get(resume_id)

        if existing_idx is not None:
            self.records[existing_idx]["preview"] = text[:300]
            self.records[existing_idx]["embedding"] = vector.tolist()
            self.records[existing_idx]["active"] = True
        else:
            self.records.append(
                {
                    "resume_id": resume_id,
                    "preview": text[:300],
                    "embedding": vector.tolist(),
                    "active": True,
                }
            )

        self._resume_id_to_record_idx = self._build_resume_id_map()
        self.index = self._rebuild_index()
        self._save()

    def delete_resume(self, resume_id: int) -> bool:
        existing_idx = self._resume_id_to_record_idx.get(resume_id)
        if existing_idx is None:
            return False

        self.records[existing_idx]["active"] = False
        self._resume_id_to_record_idx = self._build_resume_id_map()
        self.index = self._rebuild_index()
        self._save()
        return True

    def reindex(self) -> None:
        self._resume_id_to_record_idx = self._build_resume_id_map()
        self.index = self._rebuild_index()
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        qv = self.embedder.encode_texts([query], normalize_embeddings=True)
        scores, indices = self.index.search(qv, top_k)
        active_records = self._active_records()

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(active_records):
                continue
            item = dict(active_records[idx])
            item.pop("embedding", None)
            item.pop("active", None)
            item["score"] = float(score)
            results.append(item)
        return results

    def _rebuild_index(self):
        index = faiss.IndexFlatIP(self.dimension)
        active_records = self._active_records()
        if not active_records:
            return index

        vectors = np.array([r["embedding"] for r in active_records], dtype="float32")
        index.add(vectors)
        return index

    def _load_records(self) -> list[dict[str, Any]]:
        if self.meta_path.exists():
            raw = json.loads(self.meta_path.read_text(encoding="utf-8"))

            # Backward-compatible load for legacy list metadata format.
            if isinstance(raw, list):
                legacy_records: list[dict[str, Any]] = []
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    resume_id = item.get("resume_id")
                    if isinstance(resume_id, int):
                        legacy_records.append(
                            {
                                "resume_id": resume_id,
                                "preview": str(item.get("preview", "")),
                                "embedding": [],
                                "active": True,
                            }
                        )
                return legacy_records

            if isinstance(raw, dict) and isinstance(raw.get("records"), list):
                records: list[dict[str, Any]] = []
                for item in raw["records"]:
                    if not isinstance(item, dict):
                        continue
                    resume_id = item.get("resume_id")
                    embedding = item.get("embedding")
                    if not isinstance(resume_id, int):
                        continue
                    if not isinstance(embedding, list) or len(embedding) != self.dimension:
                        continue
                    records.append(
                        {
                            "resume_id": resume_id,
                            "preview": str(item.get("preview", "")),
                            "embedding": embedding,
                            "active": bool(item.get("active", True)),
                        }
                    )
                return records

        return []

    def _build_resume_id_map(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for idx, record in enumerate(self.records):
            if not record.get("active", True):
                continue
            resume_id = int(record["resume_id"])
            mapping[resume_id] = idx
        return mapping

    def _active_records(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("active", True) and len(r.get("embedding", [])) == self.dimension]

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        payload = {
            "schema_version": 2,
            "records": self.records,
        }
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
