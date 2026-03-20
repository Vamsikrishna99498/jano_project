from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    from src.config import settings
    from src.embeddings.service import EmbeddingService
    from src.parser.heuristics import parse_resume_code_first
    from src.scoring.engine import ResumeScoringEngine
    from src.schemas import ParsedResume, ScoringConstraints, ScoringWeights

    return (
        settings,
        EmbeddingService,
        ResumeScoringEngine,
        ParsedResume,
        ScoringConstraints,
        ScoringWeights,
        parse_resume_code_first,
    )


def _load_testset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Testset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_case(
    case: dict[str, Any],
    engine: Any,
    ParsedResume: Any,
    ScoringWeights: Any,
    ScoringConstraints: Any,
) -> dict[str, Any]:
    constraints = ScoringConstraints(**case["constraints"])
    weights = ScoringWeights(**case["weights"])
    jd_text = str(case["job_description"])

    scored: list[dict[str, Any]] = []
    for candidate in case["candidates"]:
        resume = ParsedResume.model_validate(candidate["parsed_resume"])
        result = engine.score_resume(
            resume_id=0,
            file_name=f"{candidate['candidate_id']}.pdf",
            resume=resume,
            raw_text=str(candidate["raw_text"]),
            jd_text=jd_text,
            weights=weights,
            constraints=constraints,
        )
        scored.append(
            {
                "candidate_id": candidate["candidate_id"],
                "name": resume.candidate_name,
                "score": result.total_score,
                "rejected": result.rejected,
                "reasons": result.rejection_reasons,
            }
        )

    ranked = sorted(scored, key=lambda x: float(x["score"]), reverse=True)

    expected = case["expected"]
    top_candidate_id = str(expected["top_candidate_id"])
    must_reject_ids = set(expected.get("must_reject_ids", []))
    top_k = int(expected.get("top_k", 2))
    top_k_contains = set(expected.get("top_k_contains", []))

    actual_top_id = ranked[0]["candidate_id"] if ranked else None
    top1_hit = actual_top_id == top_candidate_id

    actual_top_k = {item["candidate_id"] for item in ranked[:top_k]}
    top_k_hit = top_k_contains.issubset(actual_top_k)

    rejected_ids = {item["candidate_id"] for item in ranked if item["rejected"]}
    reject_hit = must_reject_ids.issubset(rejected_ids)

    return {
        "case_id": case["case_id"],
        "top1_hit": top1_hit,
        "top_k_hit": top_k_hit,
        "reject_hit": reject_hit,
        "expected_top": top_candidate_id,
        "actual_top": actual_top_id,
        "ranked": ranked,
    }


def _build_resume_text_from_parsed(parsed_resume: dict[str, Any]) -> str:
    # Build sectioned text so parser-layer scoring can be exercised in a controlled way.
    lines: list[str] = []
    name = parsed_resume.get("candidate_name") or "Candidate"
    lines.append(str(name))
    lines.append("")

    summary = parsed_resume.get("summary")
    if summary:
        lines.extend(["Summary", str(summary), ""])

    skills = parsed_resume.get("skills") or []
    if skills:
        lines.extend(["Skills", ", ".join([str(s) for s in skills]), ""])

    experience = parsed_resume.get("experience") or []
    if experience:
        lines.append("Experience")
        for exp in experience:
            title = str(exp.get("title", "")).strip()
            company = str(exp.get("company", "")).strip()
            header = f"{title} | {company}" if company else title
            lines.append(header)
            for item in exp.get("description", []):
                lines.append(f"- {str(item)}")
            lines.append("")

    education = parsed_resume.get("education") or []
    if education:
        lines.append("Education")
        for edu in education:
            degree = str(edu.get("degree", "")).strip()
            institution = str(edu.get("institution", "")).strip()
            header = f"{degree} | {institution}" if institution else degree
            lines.append(header)
            lines.append("")

    return "\n".join(lines).strip()


def _evaluate_case_parser_e2e(
    case: dict[str, Any],
    engine: Any,
    parse_resume_code_first: Any,
    ScoringWeights: Any,
    ScoringConstraints: Any,
) -> dict[str, Any]:
    constraints = ScoringConstraints(**case["constraints"])
    weights = ScoringWeights(**case["weights"])
    jd_text = str(case["job_description"])

    scored: list[dict[str, Any]] = []
    for candidate in case["candidates"]:
        synthetic_text = _build_resume_text_from_parsed(candidate["parsed_resume"])
        parsed_resume, diagnostics = parse_resume_code_first(synthetic_text)

        result = engine.score_resume(
            resume_id=0,
            file_name=f"{candidate['candidate_id']}.pdf",
            resume=parsed_resume,
            raw_text=synthetic_text,
            jd_text=jd_text,
            weights=weights,
            constraints=constraints,
        )
        scored.append(
            {
                "candidate_id": candidate["candidate_id"],
                "score": result.total_score,
                "rejected": result.rejected,
                "parser_confidence": diagnostics.confidence,
            }
        )

    ranked = sorted(scored, key=lambda x: float(x["score"]), reverse=True)
    expected_top = str(case["expected"]["top_candidate_id"])
    actual_top = ranked[0]["candidate_id"] if ranked else None

    return {
        "e2e_top1_hit": expected_top == actual_top,
        "e2e_expected_top": expected_top,
        "e2e_actual_top": actual_top,
        "e2e_ranked": ranked,
    }


def main() -> None:
    (
        settings,
        EmbeddingService,
        ResumeScoringEngine,
        ParsedResume,
        ScoringConstraints,
        ScoringWeights,
        parse_resume_code_first,
    ) = (
        _load_runtime_dependencies()
    )

    parser = argparse.ArgumentParser(description="Run small synthetic scoring evaluation.")
    parser.add_argument(
        "--testset",
        type=str,
        default="data/testsets/scoring_small_testset.json",
        help="Path to scoring testset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/testsets/results/scoring_small_eval_report.json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--e2e-parse",
        action="store_true",
        help="Also run parser-layer end-to-end scoring from synthetic resume text.",
    )
    args = parser.parse_args()

    testset_path = PROJECT_ROOT / args.testset
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _load_testset(testset_path)
    cases = payload.get("test_cases", [])
    if not cases:
        raise ValueError("No test cases found in testset.")

    embedder = EmbeddingService(settings.embedding_model)
    engine = ResumeScoringEngine(embedder=embedder)

    case_results: list[dict[str, Any]] = []
    strict_rank_hits: list[bool] = []
    strict_reject_fp_hits: list[bool] = []
    e2e_hits: list[bool] = []
    for case in cases:
        result = _evaluate_case(case, engine, ParsedResume, ScoringWeights, ScoringConstraints)

        expected = case["expected"]
        top_k = int(expected.get("top_k", 2))
        expected_top_k_exact = list(expected.get("top_k_exact", []))
        actual_top_k = [item["candidate_id"] for item in result["ranked"][:top_k]]
        strict_topk_order_hit = (not expected_top_k_exact) or (actual_top_k == expected_top_k_exact)

        must_not_reject = set(expected.get("must_not_reject_ids", []))
        actual_rejected = {item["candidate_id"] for item in result["ranked"] if item["rejected"]}
        reject_false_positive_hit = len(must_not_reject.intersection(actual_rejected)) == 0

        result["strict_topk_order_hit"] = strict_topk_order_hit
        result["reject_false_positive_hit"] = reject_false_positive_hit
        result["actual_top_k"] = actual_top_k

        strict_rank_hits.append(strict_topk_order_hit)
        strict_reject_fp_hits.append(reject_false_positive_hit)

        if args.e2e_parse:
            e2e = _evaluate_case_parser_e2e(
                case,
                engine,
                parse_resume_code_first,
                ScoringWeights,
                ScoringConstraints,
            )
            result.update(e2e)
            e2e_hits.append(bool(e2e["e2e_top1_hit"]))

        case_results.append(result)
        print(
            f"case={result['case_id']} top1={result['top1_hit']} "
            f"topk={result['top_k_hit']} reject={result['reject_hit']} "
            f"strict_topk={result['strict_topk_order_hit']} reject_fp={result['reject_false_positive_hit']} "
            f"expected_top={result['expected_top']} actual_top={result['actual_top']}"
        )

    top1_accuracy = mean([1.0 if r["top1_hit"] else 0.0 for r in case_results])
    topk_accuracy = mean([1.0 if r["top_k_hit"] else 0.0 for r in case_results])
    reject_accuracy = mean([1.0 if r["reject_hit"] else 0.0 for r in case_results])

    report = {
        "testset": str(args.testset),
        "cases": len(case_results),
        "top1_accuracy": round(top1_accuracy, 4),
        "topk_accuracy": round(topk_accuracy, 4),
        "reject_accuracy": round(reject_accuracy, 4),
        "strict_topk_order_accuracy": round(mean([1.0 if x else 0.0 for x in strict_rank_hits]), 4),
        "reject_false_positive_accuracy": round(mean([1.0 if x else 0.0 for x in strict_reject_fp_hits]), 4),
        "e2e_parse_enabled": bool(args.e2e_parse),
        "e2e_top1_accuracy": round(mean([1.0 if x else 0.0 for x in e2e_hits]), 4) if e2e_hits else None,
        "all_cases_passed": all(r["top1_hit"] and r["top_k_hit"] and r["reject_hit"] for r in case_results),
        "all_cases_strict_passed": all(
            r["top1_hit"]
            and r["top_k_hit"]
            and r["reject_hit"]
            and r["strict_topk_order_hit"]
            and r["reject_false_positive_hit"]
            and ((not args.e2e_parse) or r.get("e2e_top1_hit", False))
            for r in case_results
        ),
        "case_results": case_results,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("---")
    print(json.dumps({k: v for k, v in report.items() if k != "case_results"}, indent=2))
    print(f"report_written_to={output_path}")


if __name__ == "__main__":
    main()
