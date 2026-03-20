from __future__ import annotations

from typing import Optional

from src.config import settings
from src.parser.smart_parser import SmartParser
from src.schemas import ParseResult
from src.storage.postgres_store import PostgresStore
from src.storage.vector_store import FaissVectorStore


class ResumeIngestionPipeline:
    def __init__(self) -> None:
        self.parser = SmartParser(min_confidence_for_code_first=0.65)
        self.pg = PostgresStore(settings.database_url)
        self.faiss = FaissVectorStore(
            index_path=settings.vector_index_path,
            meta_path=settings.vector_meta_path,
            embedding_model=settings.embedding_model,
        )
        self.pg.init_db()

    def create_job_description(self, title: str, description: str) -> int:
        return self.pg.add_job_description(title=title, description=description)

    def get_job_descriptions(self) -> list[dict]:
        return self.pg.list_job_descriptions()

    def process_resume(
        self,
        file_name: str,
        content: bytes,
        job_description_id: Optional[int],
    ) -> tuple[int, ParseResult]:
        parse_result = self.parser.parse(file_name=file_name, content=content)
        resume_id = self.pg.add_resume(parse_result=parse_result, job_description_id=job_description_id)
        self.faiss.add_resume_text(resume_id=resume_id, text=parse_result.raw_text)
        return resume_id, parse_result
