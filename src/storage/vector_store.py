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
        self.index = self._load_or_rebuild_index()

    def add_resume_text(self, resume_id: int, text: str) -> None:
        vector = self.embedder.encode_texts([text], normalize_embeddings=True)[0]
        existing_idx = self._resume_id_to_record_idx.get(resume_id)
        rid = np.array([int(resume_id)], dtype="int64")

        # Incremental update: remove old id (if any) and upsert this vector.
        self.index.remove_ids(rid)
        self.index.add_with_ids(vector.reshape(1, -1), rid)

        if existing_idx is not None:
            self.records[existing_idx]["preview"] = text[:300]
            self.records[existing_idx]["active"] = True
        else:
            self.records.append(
                {
                    "resume_id": resume_id,
                    "preview": text[:300],
                    "active": True,
                }
            )

        self._resume_id_to_record_idx = self._build_resume_id_map()
        self._save()

    def delete_resume(self, resume_id: int) -> bool:
        existing_idx = self._resume_id_to_record_idx.get(resume_id)
        if existing_idx is None:
            return False

        self.records[existing_idx]["active"] = False
        self.index.remove_ids(np.array([int(resume_id)], dtype="int64"))
        self._resume_id_to_record_idx = self._build_resume_id_map()
        self._save()
        return True

    def reindex(self) -> None:
        self._resume_id_to_record_idx = self._build_resume_id_map()
        if not any(isinstance(r.get("embedding"), list) for r in self.records):
            # Compact metadata mode stores vectors only in FAISS; keep current index as source of truth.
            self._save()
            return
        self.index = self._rebuild_index_from_metadata()
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        qv = self.embedder.encode_texts([query], normalize_embeddings=True)
        scores, ids = self.index.search(qv, top_k)

        results = []
        for score, rid in zip(scores[0], ids[0]):
            if rid < 0:
                continue
            record = self._record_by_resume_id(int(rid))
            if record is None or not record.get("active", True):
                continue
            item = dict(record)
            item.pop("active", None)
            item["score"] = float(score)
            results.append(item)
        return results

    def _new_index(self) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))

    def _load_or_rebuild_index(self) -> faiss.IndexIDMap2:
        if self.index_path.exists():
            loaded = faiss.read_index(str(self.index_path))
            if hasattr(loaded, "add_with_ids") and hasattr(loaded, "remove_ids"):
                return loaded

            # Convert legacy flat index by rebuilding from metadata records.
            return self._rebuild_index_from_metadata()
        return self._rebuild_index_from_metadata()

    def _rebuild_index_from_metadata(self) -> faiss.IndexIDMap2:
        index = self._new_index()
        active_records = self._active_records()
        if not active_records:
            return index

        # Rebuild is supported only for legacy metadata that still contains embeddings.
        with_embeddings = [r for r in active_records if isinstance(r.get("embedding"), list)]
        if not with_embeddings:
            return index

        vectors = np.array([r["embedding"] for r in with_embeddings], dtype="float32")
        ids = np.array([int(r["resume_id"]) for r in with_embeddings], dtype="int64")
        index.add_with_ids(vectors, ids)
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
                    if not isinstance(resume_id, int):
                        continue
                    embedding = item.get("embedding")
                    if embedding is not None:
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
        return [r for r in self.records if r.get("active", True)]

    def _record_by_resume_id(self, resume_id: int) -> dict[str, Any] | None:
        idx = self._resume_id_to_record_idx.get(resume_id)
        if idx is None:
            return None
        return self.records[idx]

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        # Persist only lightweight metadata; vectors live in the FAISS index file.
        compact_records = [
            {
                "resume_id": int(r["resume_id"]),
                "preview": str(r.get("preview", "")),
                "active": bool(r.get("active", True)),
            }
            for r in self.records
        ]
        payload = {
            "schema_version": 3,
            "records": compact_records,
        }
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
