from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    from src.config import settings
    from src.embeddings.service import EmbeddingService
    from src.scoring.engine import ResumeScoringEngine
    from src.schemas import ParsedResume, ScoringConstraints, ScoringWeights

    return settings, EmbeddingService, ResumeScoringEngine, ParsedResume, ScoringConstraints, ScoringWeights


def _load_testset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Testset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _perturb_weights(base: dict[str, float], jitter_pct: float, rng: random.Random) -> dict[str, float]:
    keys = ["exact_match", "semantic_similarity", "achievement", "ownership"]
    out: dict[str, float] = {}
    for key in keys:
        value = float(base.get(key, 0.0))
        jitter = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
        out[key] = max(0.0, value * jitter)

    total = sum(out.values())
    if total <= 0:
        return {"exact_match": 35.0, "semantic_similarity": 30.0, "achievement": 20.0, "ownership": 15.0}

    # Keep the same 100-scale style used in testset for readability.
    return {k: (v / total) * 100.0 for k, v in out.items()}


def _soften_text(text: str) -> str:
    # Reduce strong ownership/achievement cues while keeping text readable.
    replacements = {
        "led": "worked on",
        "owned": "contributed to",
        "architected": "implemented",
        "designed": "helped design",
        "improved": "changed",
        "reduced": "changed",
        "increased": "changed",
    }
    out = text
    for src, dst in replacements.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    return out


def _alias_skill(skill: str, rng: random.Random) -> str:
    # Simulate realistic alternate naming seen in resumes.
    aliases = {
        "Kafka": ["AWS Kinesis", "RabbitMQ"],
        "React": ["ReactJS", "Frontend React"],
        "TypeScript": ["TS"],
        "PyTorch": ["torch"],
        "JavaScript": ["JS"],
        "SQL": ["Structured Query Language"],
    }
    options = aliases.get(skill)
    if not options:
        return skill
    return rng.choice(options)


def _perturb_candidate(
    candidate: dict[str, Any],
    rng: random.Random,
    text_jitter: float,
    required_skills: set[str],
) -> dict[str, Any]:
    cloned = copy.deepcopy(candidate)
    parsed = cloned.get("parsed_resume", {})

    skills = list(parsed.get("skills", []))
    if skills and rng.random() < text_jitter:
        # Randomly rewrite one skill alias.
        mutable_indexes = [
            idx for idx, value in enumerate(skills)
            if str(value).strip().lower() not in required_skills
        ]
        if mutable_indexes:
            idx = rng.choice(mutable_indexes)
            skills[idx] = _alias_skill(str(skills[idx]), rng)

    if len(skills) > 3 and rng.random() < text_jitter:
        # Drop only non-required skills to keep perturbation semantically valid.
        mutable_indexes = [
            idx for idx, value in enumerate(skills)
            if str(value).strip().lower() not in required_skills
        ]
        if mutable_indexes:
            drop_idx = rng.choice(mutable_indexes)
            skills.pop(drop_idx)

    parsed["skills"] = skills

    if rng.random() < text_jitter:
        cloned["raw_text"] = _soften_text(str(cloned.get("raw_text", "")))

    # Perturb experience descriptions lightly.
    experience = parsed.get("experience", [])
    if isinstance(experience, list):
        for exp in experience:
            desc = exp.get("description")
            if isinstance(desc, list):
                exp["description"] = [
                    _soften_text(str(line)) if rng.random() < (text_jitter * 0.6) else str(line)
                    for line in desc
                ]

    cloned["parsed_resume"] = parsed
    return cloned


def _evaluate_once(
    case: dict[str, Any],
    engine: Any,
    ParsedResume: Any,
    ScoringWeights: Any,
    ScoringConstraints: Any,
    weights_override: dict[str, float],
    rng: random.Random,
    perturb_text: bool,
    text_jitter: float,
) -> dict[str, Any]:
    constraints = ScoringConstraints(**case["constraints"])
    weights = ScoringWeights(**weights_override)
    jd_text = str(case["job_description"])

    scored: list[dict[str, Any]] = []
    required_skills = {str(s).strip().lower() for s in case.get("constraints", {}).get("required_skills", [])}
    for candidate in case["candidates"]:
        candidate_eval = (
            _perturb_candidate(candidate, rng, text_jitter, required_skills) if perturb_text else candidate
        )

        resume = ParsedResume.model_validate(candidate_eval["parsed_resume"])
        result = engine.score_resume(
            resume_id=0,
            file_name=f"{candidate['candidate_id']}.pdf",
            resume=resume,
            raw_text=str(candidate_eval["raw_text"]),
            jd_text=jd_text,
            weights=weights,
            constraints=constraints,
        )
        scored.append(
            {
                "candidate_id": candidate["candidate_id"],
                "score": float(result.total_score),
                "rejected": bool(result.rejected),
            }
        )

    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)

    expected = case["expected"]
    expected_top = str(expected["top_candidate_id"])
    must_reject_ids = set(expected.get("must_reject_ids", []))
    top_k = int(expected.get("top_k", 2))
    top_k_contains = set(expected.get("top_k_contains", []))

    actual_top = ranked[0]["candidate_id"] if ranked else None
    top1_hit = actual_top == expected_top

    top_k_ids = {r["candidate_id"] for r in ranked[:top_k]}
    top_k_hit = top_k_contains.issubset(top_k_ids)

    rejected_ids = {r["candidate_id"] for r in ranked if r["rejected"]}
    reject_hit = must_reject_ids.issubset(rejected_ids)

    return {
        "top1_hit": top1_hit,
        "top_k_hit": top_k_hit,
        "reject_hit": reject_hit,
        "actual_top": actual_top,
        "top1_top2_margin": (
            float(ranked[0]["score"] - ranked[1]["score"]) if len(ranked) >= 2 else None
        ),
    }


def main() -> None:
    settings, EmbeddingService, ResumeScoringEngine, ParsedResume, ScoringConstraints, ScoringWeights = (
        _load_runtime_dependencies()
    )

    parser = argparse.ArgumentParser(description="Run repeated randomized stability checks on small scoring testset.")
    parser.add_argument(
        "--testset",
        type=str,
        default="data/testsets/scoring_small_testset.json",
        help="Path to scoring testset JSON file.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="How many randomized runs to execute per test case.",
    )
    parser.add_argument(
        "--weight-jitter",
        type=float,
        default=0.15,
        help="Weight jitter range as ratio. 0.15 means +/-15% perturbation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/testsets/results/scoring_small_stability_report.json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--perturb-text",
        action="store_true",
        help="Enable stricter mode by perturbing candidate text and skills each iteration.",
    )
    parser.add_argument(
        "--text-jitter",
        type=float,
        default=0.20,
        help="Candidate text perturbation intensity in [0, 1]. Used with --perturb-text.",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.weight_jitter < 0:
        raise ValueError("--weight-jitter must be >= 0")
    if args.text_jitter < 0 or args.text_jitter > 1:
        raise ValueError("--text-jitter must be between 0 and 1")

    rng = random.Random(args.seed)
    testset_path = PROJECT_ROOT / args.testset
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _load_testset(testset_path)
    cases = payload.get("test_cases", [])
    if not cases:
        raise ValueError("No test cases found in testset.")

    embedder = EmbeddingService(settings.embedding_model)
    engine = ResumeScoringEngine(embedder=embedder)

    case_reports: list[dict[str, Any]] = []
    for case in cases:
        base_weights = case["weights"]

        top1_hits = 0
        topk_hits = 0
        reject_hits = 0
        all_pass = 0
        top_counter: dict[str, int] = {}
        top1_top2_margins: list[float] = []

        for _ in range(args.iterations):
            perturbed_weights = _perturb_weights(base_weights, args.weight_jitter, rng)
            run = _evaluate_once(
                case,
                engine,
                ParsedResume,
                ScoringWeights,
                ScoringConstraints,
                perturbed_weights,
                rng,
                perturb_text=args.perturb_text,
                text_jitter=args.text_jitter,
            )

            top1_hits += 1 if run["top1_hit"] else 0
            topk_hits += 1 if run["top_k_hit"] else 0
            reject_hits += 1 if run["reject_hit"] else 0
            all_pass += 1 if (run["top1_hit"] and run["top_k_hit"] and run["reject_hit"]) else 0

            if run["actual_top"] is not None:
                top_counter[run["actual_top"]] = top_counter.get(run["actual_top"], 0) + 1
            if run.get("top1_top2_margin") is not None:
                top1_top2_margins.append(float(run["top1_top2_margin"]))

        top1_top2_margins_sorted = sorted(top1_top2_margins)
        p10_margin = 0.0
        if top1_top2_margins_sorted:
            idx = max(0, int(0.1 * (len(top1_top2_margins_sorted) - 1)))
            p10_margin = float(top1_top2_margins_sorted[idx])

        report = {
            "case_id": case["case_id"],
            "iterations": args.iterations,
            "top1_stability": round(top1_hits / args.iterations, 4),
            "topk_stability": round(topk_hits / args.iterations, 4),
            "reject_stability": round(reject_hits / args.iterations, 4),
            "all_checks_pass_rate": round(all_pass / args.iterations, 4),
            "avg_top1_top2_margin": round(mean(top1_top2_margins), 4) if top1_top2_margins else None,
            "p10_top1_top2_margin": round(p10_margin, 4) if top1_top2_margins else None,
            "top_candidate_distribution": dict(sorted(top_counter.items(), key=lambda x: x[1], reverse=True)),
        }
        case_reports.append(report)
        print(
            f"case={report['case_id']} top1_stability={report['top1_stability']} "
            f"topk_stability={report['topk_stability']} reject_stability={report['reject_stability']}"
        )

    overall = {
        "avg_top1_stability": round(mean([c["top1_stability"] for c in case_reports]), 4),
        "avg_topk_stability": round(mean([c["topk_stability"] for c in case_reports]), 4),
        "avg_reject_stability": round(mean([c["reject_stability"] for c in case_reports]), 4),
        "avg_all_checks_pass_rate": round(mean([c["all_checks_pass_rate"] for c in case_reports]), 4),
        "avg_top1_top2_margin": round(
            mean([c["avg_top1_top2_margin"] for c in case_reports if c["avg_top1_top2_margin"] is not None]),
            4,
        ),
    }

    final_report = {
        "testset": args.testset,
        "iterations": args.iterations,
        "weight_jitter": args.weight_jitter,
        "perturb_text": bool(args.perturb_text),
        "text_jitter": args.text_jitter,
        "seed": args.seed,
        "overall": overall,
        "cases": case_reports,
    }

    output_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print("---")
    print(json.dumps(final_report, indent=2))
    print(f"report_written_to={output_path}")


if __name__ == "__main__":
    main()
