from __future__ import annotations

from typing import Optional

from src.config import settings
from src.embeddings.service import EmbeddingService
from src.parser.smart_parser import SmartParser
from src.scoring.engine import ResumeScoringEngine
from src.schemas import ParseResult, ParsedResume, ResumeScoreResult, ScoringConstraints, ScoringWeights
from src.storage.postgres_store import PostgresStore
from src.storage.vector_store import FaissVectorStore


class ResumeIngestionPipeline:
    def __init__(self) -> None:
        self.parser = SmartParser(min_confidence_for_code_first=0.65)
        self.pg = PostgresStore(settings.database_url)
        self.embedder = EmbeddingService(settings.embedding_model)
        self.faiss = FaissVectorStore(
            index_path=settings.vector_index_path,
            meta_path=settings.vector_meta_path,
            embedder=self.embedder,
        )
        self.scoring = ResumeScoringEngine(embedder=self.embedder)
        self.scoring_version = "phase2.v2"
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

    def score_resumes_for_job(
        self,
        job_description_id: int,
        weights: ScoringWeights,
        constraints: ScoringConstraints,
    ) -> list[ResumeScoreResult]:
        jd = self.pg.get_job_description(job_description_id)
        if jd is None:
            raise ValueError(f"Job description id={job_description_id} not found.")

        rows = self.pg.list_resumes(job_description_id=job_description_id)
        results: list[ResumeScoreResult] = []
        for row in rows:
            parsed = ParsedResume.model_validate(row["parsed_json"])
            result = self.scoring.score_resume(
                resume_id=int(row["id"]),
                file_name=str(row["file_name"]),
                resume=parsed,
                raw_text=str(row["raw_text"]),
                jd_text=str(jd["description"]),
                weights=weights,
                constraints=constraints,
            )
            results.append(result)
            self.pg.add_resume_score(
                score=result,
                job_description_id=job_description_id,
                weights=weights,
                constraints=constraints,
                scoring_version=self.scoring_version,
            )

        results.sort(key=lambda x: x.total_score, reverse=True)
        return results
