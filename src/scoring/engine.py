from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime

import numpy as np

from src.embeddings.service import EmbeddingService
from src.schemas import (
    DimensionScore,
    ParsedResume,
    ResumeScoreResult,
    ScoringConstraints,
    ScoringWeights,
)


class ResumeScoringEngine:
    def __init__(
        self,
        embedding_model: str | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        if embedder is None and not embedding_model:
            raise ValueError("Either embedder or embedding_model must be provided.")
        self.embedder = embedder or EmbeddingService(str(embedding_model))

    def score_resume(
        self,
        resume_id: int,
        file_name: str,
        resume: ParsedResume,
        raw_text: str,
        jd_text: str,
        weights: ScoringWeights,
        constraints: ScoringConstraints,
        semantic_override: tuple[float, str] | None = None,
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
        semantic_score, semantic_note = semantic_override or self._semantic_similarity_score(resume, jd_text)
        achievement_score, achievement_note = _achievement_score(raw_text)
        ownership_score, ownership_note = _ownership_score(resume)

        total = (
            normalized_weights.exact_match * exact_score
            + normalized_weights.semantic_similarity * semantic_score
            + normalized_weights.achievement * achievement_score
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
            DimensionScore(name="Achievement", score=round(achievement_score, 2), note=achievement_note),
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

        vectors = self.embedder.encode_texts([resume_text, jd_text], normalize_embeddings=True)
        score = float(np.dot(vectors[0], vectors[1]))
        mapped = max(0.0, min(100.0, (score + 1.0) * 50.0))
        return mapped, f"Semantic alignment score {mapped:.1f}/100 using local embeddings."

    def batch_semantic_similarity_scores(
        self,
        resumes_by_id: Mapping[int, ParsedResume],
        jd_text: str,
    ) -> dict[int, tuple[float, str]]:
        if not jd_text.strip() or not resumes_by_id:
            return {resume_id: (0.0, "Insufficient text for semantic comparison.") for resume_id in resumes_by_id}

        resume_ids: list[int] = []
        resume_texts: list[str] = []
        for resume_id, resume in resumes_by_id.items():
            resume_ids.append(resume_id)
            resume_texts.append(_build_resume_semantic_text(resume))

        jd_vector = self.embedder.encode_text_cached(jd_text, normalize_embeddings=True)
        resume_vectors = self.embedder.encode_texts_cached(resume_texts, normalize_embeddings=True)

        results: dict[int, tuple[float, str]] = {}
        for idx, resume_id in enumerate(resume_ids):
            resume_text = resume_texts[idx]
            if not resume_text.strip():
                results[resume_id] = (0.0, "Insufficient text for semantic comparison.")
                continue

            score = float(np.dot(resume_vectors[idx], jd_vector))
            mapped = max(0.0, min(100.0, (score + 1.0) * 50.0))
            results[resume_id] = (
                mapped,
                f"Semantic alignment score {mapped:.1f}/100 using local embeddings.",
            )
        return results


def _normalize_weights(weights: ScoringWeights) -> ScoringWeights:
    total = (
        weights.exact_match
        + weights.semantic_similarity
        + weights.achievement
        + weights.ownership
    )
    if total <= 0:
        return ScoringWeights(exact_match=0.35, semantic_similarity=0.30, achievement=0.20, ownership=0.15)
    return ScoringWeights(
        exact_match=weights.exact_match / total,
        semantic_similarity=weights.semantic_similarity / total,
        achievement=weights.achievement / total,
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


def _achievement_score(raw_text: str) -> tuple[float, str]:
    lowered = raw_text.lower()
    quantified = re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|x|k|m|b)?\b", lowered)
    achievement_terms = [
        "improved",
        "increased",
        "reduced",
        "saved",
        "optimized",
        "delivered",
        "launched",
        "achieved",
        "boosted",
        "grew",
        "scaled",
    ]
    terms_found = [w for w in achievement_terms if w in lowered]

    quantified_signal = min(1.0, len(quantified) / 12.0)
    term_signal = min(1.0, len(terms_found) / 6.0)
    score = 100.0 * (0.7 * quantified_signal + 0.3 * term_signal)
    return score, (
        f"Detected {len(quantified)} quantified indicators and "
        f"{len(terms_found)} achievement action terms."
    )


def _ownership_score(resume: ParsedResume) -> tuple[float, str]:
    if not resume.experience:
        return 0.0, "Ownership scoring requires experience entries."

    strong_verbs = [
        "architected",
        "designed",
        "led",
        "owned",
        "drove",
        "implemented",
        "mentored",
        "built",
    ]
    support_verbs = ["assisted", "helped", "supported"]

    role_scores: list[float] = []
    total_strong = 0
    total_support = 0

    for exp in resume.experience:
        title = (exp.title or "").lower()
        description_text = " ".join(exp.description).lower()

        strong_hits = sum(1 for verb in strong_verbs if verb in description_text)
        support_hits = sum(1 for verb in support_verbs if verb in description_text)
        total_strong += strong_hits
        total_support += support_hits

        # Penalize support-only phrasing within the same role context.
        net_signal = max(0.0, float(strong_hits) - (0.8 * float(support_hits)))
        base_score = min(100.0, (net_signal / 3.0) * 100.0)
        weighted_role_score = min(100.0, base_score * _role_weight(title))
        role_scores.append(weighted_role_score)

    count = len(role_scores)
    recent_count = max(1, (count + 1) // 2)

    # Assumes experience list is ordered from most recent to oldest.
    recent_scores = role_scores[:recent_count]
    older_scores = role_scores[recent_count:]

    recent_avg = sum(recent_scores) / len(recent_scores)
    if older_scores:
        older_avg = sum(older_scores) / len(older_scores)
        final_score = (0.6 * recent_avg) + (0.4 * older_avg)
    else:
        final_score = recent_avg

    note = (
        f"Ownership signals by role context: strong={total_strong}, support={total_support}; "
        f"recent-role weighting=60%."
    )
    return final_score, note


def _role_weight(title: str) -> float:
    if any(token in title for token in ["lead", "manager", "head", "principal", "staff"]):
        return 1.2
    if any(token in title for token in ["senior", "sr"]):
        return 1.1
    if any(token in title for token in ["intern", "trainee"]):
        return 0.85
    return 1.0


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
        "bachelor degree": [
            "bachelor",
            "bachelor's",
            "undergraduate",
            "b.tech",
            "btech",
            "bachelor of technology",
            "b.e",
            "be",
            "bachelor of engineering",
        ],
        "b.tech": ["btech", "b. tech", "bachelor of technology"],
        "b.e": ["be", "b.e", "bachelor of engineering"],
        "bachelor": ["bachelor", "bachelor's", "undergraduate"],
        "computer science": ["computer science", "cse", "cs"],
        "information technology": ["information technology", "it"],
        "m.tech": ["mtech", "m. tech", "master of technology"],
        "mca": ["mca", "master of computer applications"],
        "b.sc": ["bsc", "b.sc", "bachelor of science"],
        "m.sc": ["msc", "m.sc", "master of science"],
    }

    for anchor, aliases in alias_groups.items():
        if key == anchor or key in aliases:
            if anchor in degree_text:
                return True
            for alias in aliases:
                if alias in degree_text:
                    return True

    return False
