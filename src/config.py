from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/resume_ai",
    )
    vector_index_path: str = os.getenv("VECTOR_INDEX_PATH", "./data/faiss_resume.index")
    vector_meta_path: str = os.getenv("VECTOR_META_PATH", "./data/faiss_resume_meta.json")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    llm_mode: str = os.getenv("LLM_MODE", "none").lower()
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


settings = Settings()
