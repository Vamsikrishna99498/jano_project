from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.engine import Engine

from src.schemas import ParseResult


metadata = MetaData()

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
            return int(result.inserted_primary_key[0])

    def list_job_descriptions(self) -> list[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(job_descriptions.select().order_by(job_descriptions.c.id.desc()))
            return [dict(row._mapping) for row in rows]

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
            return int(result.inserted_primary_key[0])
