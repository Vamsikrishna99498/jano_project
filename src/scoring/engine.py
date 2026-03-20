from __future__ import annotations

import re
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from src.schemas import (
    DimensionScore,
    ParsedResume,
    ResumeScoreResult,
    ScoringConstraints,
    ScoringWeights,
)


class ResumeScoringEngine:
    def __init__(self, embedding_model: str) -> None:
        self.model = SentenceTransformer(embedding_model)

    def score_resume(
        self,
        resume_id: int,
        file_name: str,
        resume: ParsedResume,
        raw_text: str,
        jd_text: str,
        weights: ScoringWeights,
        constraints: ScoringConstraints,
    ) -> ResumeScoreResult:
        normalized_weights = _normalize_weights(weights)

        inferred_years = _infer_experience_years(resume, raw_text)
        rejection_reasons = _check_strict_rejections(
            resume,
            inferred_years,
            constraints,
        )
        rejected = len(rejection_reasons) > 0

        exact_score, exact_note = _exact_match_score(resume, jd_text, constraints.required_skills)
        semantic_score, semantic_note = self._semantic_similarity_score(resume, jd_text)
        impact_score, impact_note = _impact_score(raw_text)
        ownership_score, ownership_note = _ownership_score(raw_text)

        total = (
            normalized_weights.exact_match * exact_score
            + normalized_weights.semantic_similarity * semantic_score
            + normalized_weights.impact * impact_score
            + normalized_weights.ownership * ownership_score
        )

        if rejected:
            total = min(total, 40.0)

        dimensions = [
            DimensionScore(name="Exact Match", score=round(exact_score, 2), note=exact_note),
            DimensionScore(
                name="Semantic Similarity",
                score=round(semantic_score, 2),
                note=semantic_note,
            ),
            DimensionScore(name="Impact", score=round(impact_score, 2), note=impact_note),
            DimensionScore(name="Ownership", score=round(ownership_score, 2), note=ownership_note),
        ]

        explanation = _build_recruiter_explanation(
            candidate_name=resume.candidate_name,
            total_score=round(total, 2),
            inferred_years=inferred_years,
            rejected=rejected,
            rejection_reasons=rejection_reasons,
            exact_note=exact_note,
            semantic_note=semantic_note,
        )

        return ResumeScoreResult(
            resume_id=resume_id,
            file_name=file_name,
            candidate_name=resume.candidate_name,
            total_score=round(total, 2),
            rejected=rejected,
            rejection_reasons=rejection_reasons,
            dimension_scores=dimensions,
            recruiter_explanation=explanation,
        )

    def _semantic_similarity_score(self, resume: ParsedResume, jd_text: str) -> tuple[float, str]:
        resume_text = _build_resume_semantic_text(resume)
        if not resume_text.strip() or not jd_text.strip():
            return 0.0, "Insufficient text for semantic comparison."

        vectors = self.model.encode([resume_text, jd_text], normalize_embeddings=True)
        score = float(np.dot(vectors[0], vectors[1]))
        mapped = max(0.0, min(100.0, (score + 1.0) * 50.0))
        return mapped, f"Semantic alignment score {mapped:.1f}/100 using local embeddings."


def _normalize_weights(weights: ScoringWeights) -> ScoringWeights:
    total = (
        weights.exact_match
        + weights.semantic_similarity
        + weights.impact
        + weights.ownership
    )
    if total <= 0:
        return ScoringWeights(exact_match=0.35, semantic_similarity=0.30, impact=0.20, ownership=0.15)
    return ScoringWeights(
        exact_match=weights.exact_match / total,
        semantic_similarity=weights.semantic_similarity / total,
        impact=weights.impact / total,
        ownership=weights.ownership / total,
    )


def _build_resume_semantic_text(resume: ParsedResume) -> str:
    parts: list[str] = []
    if resume.summary:
        parts.append(resume.summary)
    if resume.skills:
        parts.append("Skills: " + ", ".join(resume.skills))
    for exp in resume.experience:
        parts.append(f"{exp.title} {exp.company} " + " ".join(exp.description))
    for proj in resume.projects:
        parts.append(f"{proj.name} " + " ".join(proj.description))
    return "\n".join(parts)


def _check_strict_rejections(
    resume: ParsedResume,
    inferred_years: float,
    constraints: ScoringConstraints,
) -> list[str]:
    reasons: list[str] = []

    if inferred_years < constraints.min_years_experience:
        reasons.append(
            f"Experience {inferred_years:.1f}y is below minimum {constraints.min_years_experience:.1f}y."
        )

    degree_text = " ".join([f"{e.degree} {e.institution} {' '.join(e.details)}" for e in resume.education]).lower()
    for keyword in constraints.required_degree_keywords:
        if not _degree_constraint_matched(keyword, degree_text):
            reasons.append(f"Missing required degree constraint: {keyword}")

    cert_text = " ".join(resume.certifications).lower()
    for cert in constraints.required_certifications:
        if cert.lower() not in cert_text:
            reasons.append(f"Missing required certification: {cert}")

    return reasons


def _exact_match_score(
    resume: ParsedResume,
    jd_text: str,
    required_skills: list[str],
) -> tuple[float, str]:
    resume_skills = {s.lower().strip() for s in resume.skills if s.strip()}
    resume_skill_text = " | ".join(resume_skills)

    if required_skills:
        needed = [s.lower().strip() for s in required_skills if s.strip()]
        matched = [s for s in needed if _skill_present(s, resume_skills, resume_skill_text)]
        if not needed:
            return 0.0, "No required skills provided."
        score = 100.0 * len(matched) / len(needed)
        return score, f"Matched {len(matched)}/{len(needed)} required skills."

    jd_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{1,}", jd_text.lower()))
    jd_terms = {t for t in jd_terms if len(t) > 2}
    overlap = resume_skills.intersection(jd_terms)
    if not jd_terms:
        return 0.0, "JD terms unavailable for exact matching."
    base = 100.0 * len(overlap) / max(1, min(len(jd_terms), 40))
    score = min(100.0, base)
    return score, f"Skill-term overlap found on {len(overlap)} JD terms."


def _impact_score(raw_text: str) -> tuple[float, str]:
    lowered = raw_text.lower()
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", lowered)
    impact_words = ["improved", "increased", "reduced", "saved", "optimized", "grew", "scale"]
    verbs = [w for w in impact_words if w in lowered]

    number_signal = min(1.0, len(numbers) / 10.0)
    verb_signal = min(1.0, len(verbs) / len(impact_words))
    score = 100.0 * (0.65 * number_signal + 0.35 * verb_signal)
    return score, f"Detected {len(numbers)} quantified signals and {len(verbs)} impact terms."


def _ownership_score(raw_text: str) -> tuple[float, str]:
    lowered = raw_text.lower()
    strong = ["led", "owned", "architected", "designed", "mentored", "drove", "end-to-end"]
    weak = ["assisted", "helped", "supported"]

    strong_hits = sum(1 for t in strong if t in lowered)
    weak_hits = sum(1 for t in weak if t in lowered)
    net = max(0, strong_hits - weak_hits)

    score = min(100.0, (net / 6.0) * 100.0)
    return score, f"Ownership signals: strong={strong_hits}, support={weak_hits}."


def _infer_experience_years(resume: ParsedResume, raw_text: str) -> float:
    years_from_dates = _years_from_experience_dates(resume)
    years_from_text = _years_from_text(raw_text)
    return max(years_from_dates, years_from_text)


def _years_from_experience_dates(resume: ParsedResume) -> float:
    total_months = 0
    for exp in resume.experience:
        if not exp.start_date:
            continue
        start_year = _extract_year(exp.start_date)
        end_year = _extract_year(exp.end_date or "")
        if start_year is None:
            continue
        if end_year is None:
            end_year = datetime.utcnow().year
        if end_year < start_year:
            continue
        total_months += (end_year - start_year) * 12
    return round(total_months / 12.0, 1)


def _years_from_text(raw_text: str) -> float:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", raw_text.lower())
    if not matches:
        return 0.0
    values = [float(m) for m in matches]
    return round(max(values), 1)


def _extract_year(value: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", value)
    if not match:
        return None
    return int(match.group(0))


def _build_recruiter_explanation(
    candidate_name: str | None,
    total_score: float,
    inferred_years: float,
    rejected: bool,
    rejection_reasons: list[str],
    exact_note: str,
    semantic_note: str,
) -> str:
    name = candidate_name or "Candidate"
    if rejected:
        return (
            f"{name} scored {total_score:.1f}/100 but was auto-rejected due to strict criteria: "
            + "; ".join(rejection_reasons)
        )
    return (
        f"{name} scored {total_score:.1f}/100 with inferred experience {inferred_years:.1f} years. "
        f"{exact_note} {semantic_note}"
    )


def _skill_present(required_skill: str, resume_skills: set[str], resume_skill_text: str) -> bool:
    # Direct exact match first.
    if required_skill in resume_skills:
        return True

    # Then allow substring and token-boundary style matches for prefixed skill strings.
    escaped = re.escape(required_skill)
    pattern = rf"(^|[^a-z0-9]){escaped}([^a-z0-9]|$)"
    return re.search(pattern, resume_skill_text) is not None


def _degree_constraint_matched(keyword: str, degree_text: str) -> bool:
    key = keyword.lower().strip()
    if not key:
        return True

    # Direct match.
    if key in degree_text:
        return True

    alias_groups = {
        "b.tech": ["btech", "b. tech", "bachelor of technology"],
        "b.e": ["be", "b.e", "bachelor of engineering"],
        "m.tech": ["mtech", "m. tech", "master of technology"],
    }

    for anchor, aliases in alias_groups.items():
        if key == anchor or key in aliases:
            if anchor in degree_text:
                return True
            for alias in aliases:
                if alias in degree_text:
                    return True

    return False
