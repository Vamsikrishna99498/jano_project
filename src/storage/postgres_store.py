from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, func, select
from sqlalchemy.engine import Engine

from src.schemas import ParseResult, ResumeScoreResult, ScoringConstraints, ScoringWeights


metadata = MetaData()
MAX_VECTOR_SYNC_ATTEMPTS = 5

job_descriptions = Table(
    "job_descriptions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", String(255), nullable=False),
    Column("description", Text, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

resumes = Table(
    "resumes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("file_name", String(255), nullable=False),
    Column("file_type", String(16), nullable=False),
    Column("job_description_id", Integer, nullable=True),
    Column("raw_text", Text, nullable=False),
    Column("parsed_json", JSON, nullable=False),
    Column("parser_mode", String(64), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("reasons", Text, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

resume_scores = Table(
    "resume_scores",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("resume_id", Integer, nullable=False),
    Column("job_description_id", Integer, nullable=False),
    Column("file_name", String(255), nullable=False),
    Column("candidate_name", String(255), nullable=True),
    Column("total_score", Float, nullable=False),
    Column("rejected", Integer, nullable=False),
    Column("rejection_reasons", Text, nullable=False),
    Column("dimension_scores", JSON, nullable=False),
    Column("recruiter_explanation", Text, nullable=False),
    Column("weights_json", JSON, nullable=False),
    Column("constraints_json", JSON, nullable=False),
    Column("scoring_version", String(64), nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

vector_sync_jobs = Table(
    "vector_sync_jobs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("resume_id", Integer, nullable=False),
    Column("raw_text", Text, nullable=False),
    Column("status", String(32), nullable=False),
    Column("attempt_count", Integer, nullable=False, default=0),
    Column("last_error", Text, nullable=True),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
    Column("updated_at", DateTime, nullable=False, default=datetime.utcnow),
)


class PostgresStore:
    def __init__(self, database_url: str) -> None:
        self.engine: Engine = create_engine(database_url, future=True)

    def init_db(self) -> None:
        metadata.create_all(self.engine)

    def add_job_description(self, title: str, description: str) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                job_descriptions.insert().values(
                    title=title.strip(),
                    description=description.strip(),
                    created_at=datetime.utcnow(),
                )
            )
            inserted_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
            if inserted_id is None:
                raise RuntimeError("Failed to insert job description: missing inserted primary key.")
            return int(inserted_id)

    def list_job_descriptions(self) -> list[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(job_descriptions.select().order_by(job_descriptions.c.id.desc()))
            return [dict(row._mapping) for row in rows]

    def get_job_description(self, jd_id: int) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                job_descriptions.select().where(job_descriptions.c.id == jd_id)
            ).fetchone()
            if row is None:
                return None
            return dict(row._mapping)

    def add_resume(self, parse_result: ParseResult, job_description_id: Optional[int]) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                resumes.insert().values(
                    file_name=parse_result.file_name,
                    file_type=parse_result.file_type,
                    job_description_id=job_description_id,
                    raw_text=parse_result.raw_text,
                    parsed_json=json.loads(parse_result.resume.model_dump_json()),
                    parser_mode=parse_result.diagnostics.parser_mode,
                    confidence=parse_result.diagnostics.confidence,
                    reasons=",".join(parse_result.diagnostics.reasons),
                    created_at=datetime.utcnow(),
                )
            )
            inserted_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
            if inserted_id is None:
                raise RuntimeError("Failed to insert resume: missing inserted primary key.")
            return int(inserted_id)

    def add_resume_with_vector_job(
        self,
        parse_result: ParseResult,
        job_description_id: Optional[int],
    ) -> tuple[int, int]:
        with self.engine.begin() as conn:
            resume_result = conn.execute(
                resumes.insert().values(
                    file_name=parse_result.file_name,
                    file_type=parse_result.file_type,
                    job_description_id=job_description_id,
                    raw_text=parse_result.raw_text,
                    parsed_json=json.loads(parse_result.resume.model_dump_json()),
                    parser_mode=parse_result.diagnostics.parser_mode,
                    confidence=parse_result.diagnostics.confidence,
                    reasons=",".join(parse_result.diagnostics.reasons),
                    created_at=datetime.utcnow(),
                )
            )
            resume_id = resume_result.inserted_primary_key[0] if resume_result.inserted_primary_key else None
            if resume_id is None:
                raise RuntimeError("Failed to insert resume: missing inserted primary key.")

            job_result = conn.execute(
                vector_sync_jobs.insert().values(
                    resume_id=int(resume_id),
                    raw_text=parse_result.raw_text,
                    status="pending",
                    attempt_count=0,
                    last_error=None,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )
            job_id = job_result.inserted_primary_key[0] if job_result.inserted_primary_key else None
            if job_id is None:
                raise RuntimeError("Failed to insert vector sync job: missing inserted primary key.")

            return int(resume_id), int(job_id)

    def add_resumes_bulk(self, resume_rows: list[dict]) -> int:
        if not resume_rows:
            return 0
        with self.engine.begin() as conn:
            conn.execute(resumes.insert(), resume_rows)
        return len(resume_rows)

    def list_resumes(self, job_description_id: Optional[int] = None) -> list[dict]:
        query = resumes.select().order_by(resumes.c.id.desc())
        if job_description_id is not None:
            query = query.where(resumes.c.job_description_id == job_description_id)
        with self.engine.begin() as conn:
            rows = conn.execute(query)
            return [dict(row._mapping) for row in rows]

    def count_resumes(self, job_description_id: Optional[int] = None) -> int:
        query = select(func.count()).select_from(resumes)
        if job_description_id is not None:
            query = query.where(resumes.c.job_description_id == job_description_id)
        with self.engine.begin() as conn:
            value = conn.execute(query).scalar_one()
            return int(value)

    def list_resumes_page(
        self,
        job_description_id: int,
        limit: int,
        offset: int,
    ) -> list[dict]:
        query = (
            resumes.select()
            .where(resumes.c.job_description_id == job_description_id)
            .order_by(resumes.c.id.desc())
            .limit(max(1, int(limit)))
            .offset(max(0, int(offset)))
        )
        with self.engine.begin() as conn:
            rows = conn.execute(query)
            return [dict(row._mapping) for row in rows]

    def list_vector_sync_jobs(
        self,
        status: str = "pending",
        limit: int = 50,
        max_attempts: int = MAX_VECTOR_SYNC_ATTEMPTS,
    ) -> list[dict]:
        query = vector_sync_jobs.select().where(vector_sync_jobs.c.status == status)
        if status == "pending":
            query = query.where(vector_sync_jobs.c.attempt_count < max(1, int(max_attempts)))

        query = query.order_by(vector_sync_jobs.c.id.asc()).limit(max(1, int(limit)))
        with self.engine.begin() as conn:
            rows = conn.execute(query)
            return [dict(row._mapping) for row in rows]

    def mark_vector_sync_job_success(self, job_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                vector_sync_jobs.update()
                .where(vector_sync_jobs.c.id == job_id)
                .values(
                    status="done",
                    last_error=None,
                    updated_at=datetime.utcnow(),
                )
            )

    def mark_vector_sync_job_failure(self, job_id: int, error: str) -> None:
        with self.engine.begin() as conn:
            row = conn.execute(
                vector_sync_jobs.select().where(vector_sync_jobs.c.id == job_id)
            ).fetchone()
            attempts = 0
            if row is not None:
                attempts = int(row._mapping.get("attempt_count") or 0)

            next_attempts = attempts + 1
            next_status = "failed" if next_attempts >= MAX_VECTOR_SYNC_ATTEMPTS else "pending"

            conn.execute(
                vector_sync_jobs.update()
                .where(vector_sync_jobs.c.id == job_id)
                .values(
                    status=next_status,
                    attempt_count=next_attempts,
                    last_error=error[:1000],
                    updated_at=datetime.utcnow(),
                )
            )

    def get_vector_sync_summary(self) -> dict[str, int]:
        with self.engine.begin() as conn:
            pending = conn.execute(
                select(func.count()).select_from(vector_sync_jobs).where(vector_sync_jobs.c.status == "pending")
            ).scalar_one()
            done = conn.execute(
                select(func.count()).select_from(vector_sync_jobs).where(vector_sync_jobs.c.status == "done")
            ).scalar_one()
            failed = conn.execute(
                select(func.count()).select_from(vector_sync_jobs).where(vector_sync_jobs.c.status == "failed")
            ).scalar_one()
            return {
                "pending": int(pending),
                "done": int(done),
                "failed": int(failed),
            }

    def add_resume_score(
        self,
        score: ResumeScoreResult,
        job_description_id: int,
        weights: ScoringWeights,
        constraints: ScoringConstraints,
        scoring_version: str,
    ) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                resume_scores.insert().values(
                    resume_id=score.resume_id,
                    job_description_id=job_description_id,
                    file_name=score.file_name,
                    candidate_name=score.candidate_name,
                    total_score=score.total_score,
                    rejected=1 if score.rejected else 0,
                    rejection_reasons=",".join(score.rejection_reasons),
                    dimension_scores=[d.model_dump() for d in score.dimension_scores],
                    recruiter_explanation=score.recruiter_explanation,
                    weights_json=weights.model_dump(),
                    constraints_json=constraints.model_dump(),
                    scoring_version=scoring_version,
                    created_at=datetime.utcnow(),
                )
            )
            inserted_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
            if inserted_id is None:
                raise RuntimeError("Failed to insert resume score: missing inserted primary key.")
            return int(inserted_id)

    def add_resume_scores_bulk(
        self,
        scores: list[ResumeScoreResult],
        job_description_id: int,
        weights: ScoringWeights,
        constraints: ScoringConstraints,
        scoring_version: str,
    ) -> int:
        if not scores:
            return 0

        now = datetime.utcnow()
        payload = [
            {
                "resume_id": score.resume_id,
                "job_description_id": job_description_id,
                "file_name": score.file_name,
                "candidate_name": score.candidate_name,
                "total_score": score.total_score,
                "rejected": 1 if score.rejected else 0,
                "rejection_reasons": ",".join(score.rejection_reasons),
                "dimension_scores": [d.model_dump() for d in score.dimension_scores],
                "recruiter_explanation": score.recruiter_explanation,
                "weights_json": weights.model_dump(),
                "constraints_json": constraints.model_dump(),
                "scoring_version": scoring_version,
                "created_at": now,
            }
            for score in scores
        ]

        with self.engine.begin() as conn:
            conn.execute(resume_scores.insert(), payload)
        return len(scores)

    def list_resume_scores(self, job_description_id: Optional[int] = None) -> list[dict]:
        query = resume_scores.select().order_by(resume_scores.c.id.desc())
        if job_description_id is not None:
            query = query.where(resume_scores.c.job_description_id == job_description_id)
        with self.engine.begin() as conn:
            rows = conn.execute(query)
            return [dict(row._mapping) for row in rows]
