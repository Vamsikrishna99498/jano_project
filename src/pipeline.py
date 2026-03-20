from __future__ import annotations

from time import perf_counter
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
        self.scoring_page_size = 500
        self.scoring_db_write_batch_size = 200
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
        started = perf_counter()
        parse_started = perf_counter()
        parse_result = self.parser.parse(file_name=file_name, content=content)
        parse_ms = (perf_counter() - parse_started) * 1000.0

        db_started = perf_counter()
        resume_id, job_id = self.pg.add_resume_with_vector_job(
            parse_result=parse_result,
            job_description_id=job_description_id,
        )
        db_insert_ms = (perf_counter() - db_started) * 1000.0

        vector_started = perf_counter()
        vector_synced = self._attempt_vector_sync(job_id=job_id, resume_id=resume_id, text=parse_result.raw_text)
        vector_upsert_ms = (perf_counter() - vector_started) * 1000.0
        if not vector_synced:
            parse_result.diagnostics.reasons.append("vector_sync_pending_retry")

        parse_result.diagnostics.timing_ms = {
            "parse": round(parse_ms, 2),
            "db_insert": round(db_insert_ms, 2),
            "vector_upsert": round(vector_upsert_ms, 2),
            "total": round((perf_counter() - started) * 1000.0, 2),
        }
        return resume_id, parse_result

    def retry_pending_vector_sync_jobs(self, limit: int = 25, max_attempts: int = 5) -> tuple[int, int]:
        rows = self.pg.list_vector_sync_jobs(status="pending", limit=limit, max_attempts=max_attempts)
        success = 0
        failed = 0

        for row in rows:
            ok = self._attempt_vector_sync(
                job_id=int(row["id"]),
                resume_id=int(row["resume_id"]),
                text=str(row["raw_text"]),
            )
            if ok:
                success += 1
            else:
                failed += 1

        return success, failed

    def get_vector_sync_summary(self) -> dict[str, int]:
        return self.pg.get_vector_sync_summary()

    def _attempt_vector_sync(self, job_id: int, resume_id: int, text: str) -> bool:
        try:
            self.faiss.add_resume_text(resume_id=resume_id, text=text)
            self.pg.mark_vector_sync_job_success(job_id)
            return True
        except Exception as exc:
            self.pg.mark_vector_sync_job_failure(job_id, str(exc))
            return False

    def score_resumes_for_job(
        self,
        job_description_id: int,
        weights: ScoringWeights,
        constraints: ScoringConstraints,
        persist_scores: bool = True,
        max_resumes: int | None = None,
    ) -> list[ResumeScoreResult]:
        started = perf_counter()
        jd = self.pg.get_job_description(job_description_id)
        if jd is None:
            raise ValueError(f"Job description id={job_description_id} not found.")

        results: list[ResumeScoreResult] = []
        fetched_rows_total = 0
        fetch_rows_ms = 0.0
        semantic_batch_ms = 0.0
        scoring_total_ms = 0.0
        db_write_total_ms = 0.0

        offset = 0
        max_rows = max_resumes if max_resumes is None else max(0, int(max_resumes))
        while True:
            if max_rows is not None and fetched_rows_total >= max_rows:
                break

            page_limit = self.scoring_page_size
            if max_rows is not None:
                page_limit = min(page_limit, max_rows - fetched_rows_total)
                if page_limit <= 0:
                    break

            fetch_started = perf_counter()
            rows = self.pg.list_resumes_page(
                job_description_id=job_description_id,
                limit=page_limit,
                offset=offset,
            )
            fetch_rows_ms += (perf_counter() - fetch_started) * 1000.0
            if not rows:
                break

            fetched_rows_total += len(rows)
            parsed_by_id = {
                int(row["id"]): ParsedResume.model_validate(row["parsed_json"])
                for row in rows
            }

            semantic_started = perf_counter()
            semantic_overrides = self.scoring.batch_semantic_similarity_scores(
                resumes_by_id=parsed_by_id,
                jd_text=str(jd["description"]),
            )
            semantic_batch_ms += (perf_counter() - semantic_started) * 1000.0

            pending_db_rows: list[ResumeScoreResult] = []
            for row in rows:
                resume_id = int(row["id"])
                parsed = parsed_by_id[resume_id]

                score_started = perf_counter()
                result = self.scoring.score_resume(
                    resume_id=resume_id,
                    file_name=str(row["file_name"]),
                    resume=parsed,
                    raw_text=str(row["raw_text"]),
                    jd_text=str(jd["description"]),
                    weights=weights,
                    constraints=constraints,
                    semantic_override=semantic_overrides.get(resume_id),
                )
                score_ms = (perf_counter() - score_started) * 1000.0
                scoring_total_ms += score_ms

                result.timing_ms = {
                    "score_compute": round(score_ms, 2),
                }
                results.append(result)
                pending_db_rows.append(result)

                if persist_scores and len(pending_db_rows) >= self.scoring_db_write_batch_size:
                    db_write_started = perf_counter()
                    self.pg.add_resume_scores_bulk(
                        scores=pending_db_rows,
                        job_description_id=job_description_id,
                        weights=weights,
                        constraints=constraints,
                        scoring_version=self.scoring_version,
                    )
                    db_write_total_ms += (perf_counter() - db_write_started) * 1000.0
                    pending_db_rows = []

            if persist_scores and pending_db_rows:
                db_write_started = perf_counter()
                self.pg.add_resume_scores_bulk(
                    scores=pending_db_rows,
                    job_description_id=job_description_id,
                    weights=weights,
                    constraints=constraints,
                    scoring_version=self.scoring_version,
                )
                db_write_total_ms += (perf_counter() - db_write_started) * 1000.0

            offset += self.scoring_page_size

        results.sort(key=lambda x: x.total_score, reverse=True)

        total_ms = (perf_counter() - started) * 1000.0
        run_metrics = {
            "rows": fetched_rows_total,
            "fetch_rows": round(fetch_rows_ms, 2),
            "semantic_batch": round(semantic_batch_ms, 2),
            "score_compute_total": round(scoring_total_ms, 2),
            "db_write_total": round(db_write_total_ms, 2),
            "total": round(total_ms, 2),
        }
        for result in results:
            result.timing_ms.update(run_metrics)

        return results
