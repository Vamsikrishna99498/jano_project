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


def _load_runtime_dependencies() -> tuple[Any, Any]:
    from src.parser.heuristics import parse_resume_code_first

    return parse_resume_code_first, _normalize_skill


def _normalize_skill(value: str) -> str:
    aliases = {
        "js": "javascript",
        "reactjs": "react",
        "structured query language": "sql",
    }
    key = " ".join(str(value).strip().lower().split())
    return aliases.get(key, key)


def _load_samples(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Parser QA sample file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parse_resume_code_first, normalize_skill = _load_runtime_dependencies()

    parser = argparse.ArgumentParser(description="Run parser QA checks on mildly unstructured text fixtures.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/testsets/parser_qa_samples.json",
        help="Path to parser QA fixtures JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/testsets/results/parser_qa_report.json",
        help="Path to write parser QA report.",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _load_samples(input_path)
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError("No samples found in parser QA input.")

    case_reports: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        raw_text = str(sample["raw_text"])
        expect = sample.get("expect", {})

        parsed, diagnostics = parse_resume_code_first(raw_text)

        parsed_skills = [normalize_skill(s) for s in parsed.skills]
        expected_skills = [normalize_skill(s) for s in expect.get("must_have_skills", [])]
        missing_skills = [s for s in expected_skills if s not in parsed_skills]

        min_experience_items = int(expect.get("min_experience_items", 0))
        require_email = bool(expect.get("require_email", False))
        require_summary = bool(expect.get("require_summary", False))

        checks = {
            "skills_hit": len(missing_skills) == 0,
            "experience_hit": len(parsed.experience) >= min_experience_items,
            "email_hit": (parsed.contact.email is not None) if require_email else True,
            "summary_hit": (parsed.summary is not None and len(parsed.summary.strip()) > 0) if require_summary else True,
        }

        report = {
            "sample_id": sample_id,
            "confidence": round(diagnostics.confidence, 4),
            "reasons": diagnostics.reasons,
            "parsed_skill_count": len(parsed.skills),
            "parsed_experience_count": len(parsed.experience),
            "missing_skills": missing_skills,
            "checks": checks,
            "all_passed": all(checks.values()),
        }
        case_reports.append(report)

        print(
            f"sample={sample_id} all_passed={report['all_passed']} "
            f"confidence={report['confidence']} missing_skills={len(missing_skills)}"
        )

    overall = {
        "samples": len(case_reports),
        "pass_rate": round(mean([1.0 if c["all_passed"] else 0.0 for c in case_reports]), 4),
        "avg_confidence": round(mean([c["confidence"] for c in case_reports]), 4),
        "all_samples_passed": all(c["all_passed"] for c in case_reports),
    }

    final_report = {
        "input": args.input,
        "overall": overall,
        "samples": case_reports,
    }

    output_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print("---")
    print(json.dumps(overall, indent=2))
    print(f"report_written_to={output_path}")


if __name__ == "__main__":
    main()
