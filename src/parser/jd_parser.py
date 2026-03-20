from __future__ import annotations

import re

from src.schemas import JDParsedDetails


COMMON_SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "postgresql",
    "mysql",
    "mongodb",
    "fastapi",
    "flask",
    "django",
    "spring",
    "react",
    "node",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "machine learning",
    "deep learning",
    "nlp",
    "pytorch",
    "tensorflow",
    "langchain",
    "rag",
]


DEGREE_ALIASES = {
    "Bachelor Degree": [
        "b.tech",
        "btech",
        "bachelor of technology",
        "b.e",
        "be",
        "bachelor of engineering",
        "bachelor's",
        "bachelor",
        "undergraduate",
    ],
    "Computer Science": ["computer science", "cse", "cs"],
    "Information Technology": ["information technology", "it"],
    "M.Tech": ["m.tech", "mtech", "master of technology"],
    "MCA": ["mca", "master of computer applications"],
    "B.Sc": ["b.sc", "bsc", "bachelor of science"],
    "M.Sc": ["m.sc", "msc", "master of science"],
}


def parse_jd_text(jd_text: str) -> JDParsedDetails:
    text = jd_text.strip()
    lowered = text.lower()

    title_hint = _extract_title_hint(text)
    required_skills = _extract_required_skills(lowered)
    min_years = _extract_min_years(lowered)
    degrees = _extract_degrees(lowered)

    return JDParsedDetails(
        title_hint=title_hint,
        required_skills=required_skills,
        min_years_experience=min_years,
        required_degree_keywords=degrees,
    )


def _extract_title_hint(text: str) -> str | None:
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not first_line:
        return None
    if len(first_line) > 90:
        return None
    return first_line


def _extract_required_skills(lowered: str) -> list[str]:
    found = []
    for skill in COMMON_SKILLS:
        pattern = rf"(^|[^a-z0-9]){re.escape(skill)}([^a-z0-9]|$)"
        if re.search(pattern, lowered):
            found.append(_title_case_skill(skill))
    return sorted(set(found))


def _extract_min_years(lowered: str) -> float:
    patterns = [
        r"(?:minimum|min|at least)\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s*(?:of)?\s*experience",
    ]
    values: list[float] = []
    for pattern in patterns:
        for match in re.findall(pattern, lowered):
            try:
                values.append(float(match))
            except ValueError:
                continue
    return max(values) if values else 0.0


def _extract_degrees(lowered: str) -> list[str]:
    found = []
    for label, aliases in DEGREE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            found.append(label)
    return sorted(set(found))


def _title_case_skill(skill: str) -> str:
    keep_upper = {"sql", "aws", "gcp", "nlp", "rag"}
    if skill in keep_upper:
        return skill.upper()
    if skill == "postgresql":
        return "PostgreSQL"
    if skill == "javascript":
        return "JavaScript"
    if skill == "typescript":
        return "TypeScript"
    if skill == "fastapi":
        return "FastAPI"
    return skill.title()
