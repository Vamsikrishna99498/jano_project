from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow direct execution: `python scripts/seed_benchmark_resumes.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_runtime_dependencies() -> tuple[Any, Any]:
    from src.config import settings
    from src.storage.postgres_store import PostgresStore

    return settings, PostgresStore


def _pick_jd_id(store: Any, requested: int | None) -> int:
    if requested is not None:
        jd = store.get_job_description(requested)
        if jd is None:
            raise ValueError(f"Job description id={requested} not found.")
        return requested

    rows = store.list_job_descriptions()
    if not rows:
        raise ValueError("No job descriptions found. Create one in the app first.")
    return int(rows[0]["id"])


def _random_skill_block(rng: random.Random) -> list[str]:
    pool = [
        "Python",
        "Java",
        "SQL",
        "PostgreSQL",
        "Docker",
        "Kubernetes",
        "AWS",
        "Kafka",
        "RabbitMQ",
        "AWS Kinesis",
        "FastAPI",
        "Flask",
        "ETL",
        "Machine Learning",
        "NLP",
    ]
    size = rng.randint(6, 10)
    return sorted(rng.sample(pool, k=size))


def _build_resume_row(jd_id: int, i: int, seed: int) -> dict:
    rng = random.Random(seed + i)
    years = rng.randint(1, 12)
    candidate_name = f"Benchmark Candidate {i:05d}"
    skills = _random_skill_block(rng)

    impact = rng.choice(["15%", "22%", "35%", "2x", "3x", "40%"])
    ownership_verb = rng.choice(["led", "owned", "designed", "implemented", "architected"])
    stream_tool = rng.choice(["Kafka", "RabbitMQ", "AWS Kinesis"])

    summary = (
        f"Backend engineer with {years}+ years experience in distributed systems, "
        f"data pipelines, and cloud-native delivery."
    )

    experience = [
        {
            "title": rng.choice(["Software Engineer", "Senior Software Engineer", "Backend Engineer"]),
            "company": rng.choice(["Acme", "NovaTech", "DataOrbit", "CloudLeaf"]),
            "start_date": f"{datetime.now(UTC).year - years}-01",
            "end_date": "Present",
            "description": [
                f"{ownership_verb.title()} event-driven services using {stream_tool} and PostgreSQL.",
                f"Improved reliability by {impact} via observability and autoscaling.",
                "Built APIs and async workers for resume parsing and scoring workflows.",
            ],
        }
    ]

    education = [
        {
            "degree": rng.choice(["B.Tech Computer Science", "B.E Information Technology", "B.Sc Computer Science"]),
            "institution": rng.choice(["State University", "Tech Institute", "City College"]),
            "start_date": "2014",
            "end_date": "2018",
            "details": [],
        }
    ]

    projects = [
        {
            "name": "Resume Match Engine",
            "description": [
                "Built semantic matching between resumes and job descriptions.",
                "Added explainable multi-dimensional scoring for recruiters.",
            ],
            "links": [f"https://github.com/example/candidate-{i:05d}"],
        }
    ]

    parsed_json = {
        "candidate_name": candidate_name,
        "contact": {
            "email": f"candidate{i:05d}@example.com",
            "phone": "+1-555-0100",
            "location": "Remote",
            "linkedin": f"https://www.linkedin.com/in/candidate-{i:05d}",
            "github": f"https://github.com/candidate-{i:05d}",
            "website": None,
        },
        "summary": summary,
        "skills": skills,
        "experience": experience,
        "education": education,
        "projects": projects,
        "certifications": [],
        "raw_sections": {},
    }

    raw_text = "\n".join(
        [
            candidate_name,
            summary,
            "Skills: " + ", ".join(skills),
            experience[0]["title"] + " at " + experience[0]["company"],
            *experience[0]["description"],
        ]
    )

    return {
        "file_name": f"benchmark_resume_{i:05d}.pdf",
        "file_type": "pdf",
        "job_description_id": jd_id,
        "raw_text": raw_text,
        "parsed_json": parsed_json,
        "parser_mode": "benchmark_seed",
        "confidence": 0.95,
        "reasons": "benchmark_seed",
        "created_at": datetime.now(UTC),
    }


def main() -> None:
    settings, PostgresStore = _load_runtime_dependencies()

    parser = argparse.ArgumentParser(description="Seed synthetic resumes for benchmark runs.")
    parser.add_argument("--jd-id", type=int, default=None, help="Target job description id.")
    parser.add_argument("--count", type=int, default=1000, help="Number of synthetic resumes to add.")
    parser.add_argument("--batch-size", type=int, default=500, help="Insert batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be > 0")

    store = PostgresStore(settings.database_url)
    store.init_db()
    jd_id = _pick_jd_id(store, args.jd_id)

    start_index = store.count_resumes(job_description_id=jd_id) + 1
    remaining = args.count
    current = start_index

    while remaining > 0:
        size = min(max(1, args.batch_size), remaining)
        batch = [_build_resume_row(jd_id, i, args.seed) for i in range(current, current + size)]
        store.add_resumes_bulk(batch)
        current += size
        remaining -= size

    total = store.count_resumes(job_description_id=jd_id)
    print(
        f"Seeded {args.count} resumes into jd_id={jd_id}. "
        f"Current total resumes for this JD: {total}."
    )


if __name__ == "__main__":
    main()
