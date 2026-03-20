from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

# Allow direct execution: `python scripts/benchmark_option_a.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    from src.parser.jd_parser import parse_jd_text
    from src.pipeline import ResumeIngestionPipeline
    from src.schemas import ScoringConstraints, ScoringWeights

    return parse_jd_text, ResumeIngestionPipeline, ScoringConstraints, ScoringWeights


def _load_storage_dependencies() -> tuple[Any, Any]:
    from src.config import settings
    from src.storage.postgres_store import PostgresStore

    return settings, PostgresStore


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _pick_jd_id(pipeline: Any, requested: int | None) -> int:
    if requested is not None:
        jd = pipeline.pg.get_job_description(requested)
        if jd is None:
            raise ValueError(f"Job description id={requested} not found.")
        return requested

    rows = pipeline.get_job_descriptions()
    if not rows:
        raise ValueError("No job descriptions found. Create one in the app first.")
    return int(rows[0]["id"])


def _build_constraints(jd_text: str, parse_jd_text: Any, ScoringConstraints: Any) -> Any:
    parsed = parse_jd_text(jd_text)
    return ScoringConstraints(
        min_years_experience=float(parsed.min_years_experience),
        required_degree_keywords=parsed.required_degree_keywords,
        required_skills=parsed.required_skills,
    )


def _run_once(
    pipeline: Any,
    jd_id: int,
    weights: Any,
    constraints: Any,
    persist_scores: bool,
    max_resumes: int | None,
) -> tuple[float, int, dict[str, float]]:
    started = perf_counter()
    results = pipeline.score_resumes_for_job(
        job_description_id=jd_id,
        weights=weights,
        constraints=constraints,
        persist_scores=persist_scores,
        max_resumes=max_resumes,
    )
    elapsed = perf_counter() - started
    rows = len(results)
    timing = results[0].timing_ms if results else {}
    return elapsed, rows, timing


def main() -> None:
    parse_jd_text, ResumeIngestionPipeline, ScoringConstraints, ScoringWeights = _load_runtime_dependencies()
    settings, PostgresStore = _load_storage_dependencies()

    parser = argparse.ArgumentParser(description="Benchmark Option A scoring throughput.")
    parser.add_argument("--jd-id", type=int, default=None, help="Job description id to benchmark.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs.")
    parser.add_argument("--warmups", type=int, default=1, help="Warm-up runs.")
    parser.add_argument(
        "--persist-scores",
        action="store_true",
        help="Persist score rows during benchmark (includes DB write cost).",
    )
    parser.add_argument(
        "--list-jds",
        action="store_true",
        help="List job descriptions and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run progress logs.",
    )
    parser.add_argument(
        "--max-resumes",
        type=int,
        default=None,
        help="Optional cap on resumes scored per run for faster sampling (e.g., 300).",
    )
    args = parser.parse_args()

    store = PostgresStore(settings.database_url)
    store.init_db()

    if args.list_jds:
        rows = store.list_job_descriptions()
        print(json.dumps([{"id": int(r["id"]), "title": str(r["title"])} for r in rows], indent=2))
        return

    pipeline = ResumeIngestionPipeline()

    jd_id = _pick_jd_id(pipeline, args.jd_id)

    jd = pipeline.pg.get_job_description(jd_id)
    if jd is None:
        raise ValueError(f"Job description id={jd_id} not found.")

    total_resumes = pipeline.pg.count_resumes(job_description_id=jd_id)
    if total_resumes <= 0:
        raise ValueError("No resumes attached to this job description. Upload resumes first.")

    weights = ScoringWeights(exact_match=35.0, semantic_similarity=30.0, achievement=20.0, ownership=15.0)
    constraints = _build_constraints(str(jd["description"]), parse_jd_text, ScoringConstraints)

    for _ in range(max(0, args.warmups)):
        _run_once(
            pipeline,
            jd_id,
            weights,
            constraints,
            persist_scores=args.persist_scores,
            max_resumes=args.max_resumes,
        )

    run_seconds: list[float] = []
    run_rows_per_second: list[float] = []
    run_rows: list[int] = []
    sample_timing: dict[str, float] = {}

    total_runs = max(1, args.runs)
    for idx in range(total_runs):
        elapsed, rows, timing = _run_once(
            pipeline,
            jd_id,
            weights,
            constraints,
            persist_scores=args.persist_scores,
            max_resumes=args.max_resumes,
        )
        run_seconds.append(elapsed)
        run_rows.append(rows)
        run_rows_per_second.append((rows / elapsed) if elapsed > 0 else 0.0)
        if not sample_timing and timing:
            sample_timing = timing
        if not args.quiet:
            print(
                f"run {idx + 1}/{total_runs}: rows={rows}, elapsed={elapsed:.2f}s, "
                f"throughput={(rows / elapsed) if elapsed > 0 else 0.0:.2f} rows/s"
            )

    avg_seconds = mean(run_seconds)
    avg_rows = int(round(mean(run_rows)))
    rows_per_second = (avg_rows / avg_seconds) if avg_seconds > 0 else 0.0
    rows_per_hour = rows_per_second * 3600.0
    rows_per_day = rows_per_hour * 24.0

    p50_seconds = _percentile(run_seconds, 0.50)
    p95_seconds = _percentile(run_seconds, 0.95)
    p99_seconds = _percentile(run_seconds, 0.99)
    p95_rows_per_second = _percentile(run_rows_per_second, 0.05)

    report = {
        "jd_id": jd_id,
        "persist_scores": bool(args.persist_scores),
        "runs": len(run_seconds),
        "warmups": max(0, args.warmups),
        "resumes_available_for_jd": total_resumes,
        "max_resumes_per_run": args.max_resumes,
        "resumes_scored_per_run": avg_rows,
        "avg_run_seconds": round(avg_seconds, 3),
        "p50_run_seconds": round(p50_seconds, 3),
        "p95_run_seconds": round(p95_seconds, 3),
        "p99_run_seconds": round(p99_seconds, 3),
        "throughput_rows_per_second": round(rows_per_second, 2),
        "throughput_p95_rows_per_second": round(p95_rows_per_second, 2),
        "throughput_rows_per_hour": round(rows_per_hour, 0),
        "projected_rows_per_day": round(rows_per_day, 0),
        "target_rows_per_day": 10000,
        "meets_target": rows_per_day >= 10000.0,
        "sample_size_warning": total_resumes < 1000,
        "sample_pipeline_timing_ms": sample_timing,
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
