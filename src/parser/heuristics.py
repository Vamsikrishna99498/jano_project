from __future__ import annotations

import re
from typing import Dict, List, Tuple

from src.schemas import (
    ContactInfo,
    EducationItem,
    ExperienceItem,
    ParseDiagnostics,
    ParsedResume,
    ProjectItem,
)


SECTION_HEADERS = {
    "summary": ["summary", "profile", "objective", "professional summary", "about"],
    "skills": [
        "skills",
        "technical skills",
        "core competencies",
        "technical stack",
        "tech stack",
        "skills and tools",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "work history",
    ],
    "education": ["education", "academic", "academic background"],
    "projects": ["projects", "project experience", "personal projects"],
    "certifications": ["certifications", "licenses", "courses"],
}


def split_sections(text: str) -> Dict[str, str]:
    lines = [ln.strip() for ln in text.splitlines()]
    sections: Dict[str, List[str]] = {k: [] for k in SECTION_HEADERS}
    current = "summary"

    for line in lines:
        if not line:
            continue
        matched, inline_content = _match_section_header(line)
        if matched:
            current = matched
            if inline_content:
                sections[current].append(inline_content)
            continue
        sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def extract_contact(text: str) -> ContactInfo:
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"\+?[0-9][0-9\-\s()]{7,}[0-9]", text)

    linkedin_match = re.search(r"https?://(?:www\.)?linkedin\.com/[^\s]+", text)
    github_match = re.search(r"https?://(?:www\.)?github\.com/[^\s]+", text)
    website_match = re.search(r"https?://[^\s]+", text)

    return ContactInfo(
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0).strip() if phone_match else None,
        linkedin=linkedin_match.group(0) if linkedin_match else None,
        github=github_match.group(0) if github_match else None,
        website=website_match.group(0) if website_match else None,
    )


def parse_skills(section: str) -> List[str]:
    if not section.strip():
        return []

    cleaned = re.sub(r"(?i)^skills\s*[:\-]\s*", "", section.strip())
    cleaned = cleaned.replace("\u2022", "|")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    tokens = re.split(r"[,|;\n]", cleaned)
    skills = []
    for token in tokens:
        parts = [token]
        # Split slash-separated compounds only when they look like list separators.
        if "/" in token and not re.search(r"c\+\+|ci/cd|a/b", token, flags=re.IGNORECASE):
            parts = token.split("/")

        for part in parts:
            value = part.strip(" -:\t")
            if value:
                skills.append(value)

    unique = []
    seen = set()
    for skill in skills:
        key = _normalize_skill_key(skill)
        if key not in seen:
            seen.add(key)
            unique.append(skill)
    return unique


def parse_experience(section: str) -> List[ExperienceItem]:
    if not section:
        return []
    blocks = [b.strip() for b in section.split("\n\n") if b.strip()]
    items: list[ExperienceItem] = []
    for block in blocks:
        lines = [ln.strip(" -") for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        header = lines[0]
        title, company = _split_title_company(header)
        items.append(
            ExperienceItem(
                title=title,
                company=company,
                description=lines[1:] if len(lines) > 1 else [],
            )
        )
    return items


def parse_education(section: str) -> List[EducationItem]:
    if not section:
        return []
    blocks = [b.strip() for b in section.split("\n\n") if b.strip()]
    items: list[EducationItem] = []
    for block in blocks:
        lines = [ln.strip(" -") for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        degree, institution = _split_title_company(lines[0])
        items.append(EducationItem(degree=degree, institution=institution, details=lines[1:]))
    return items


def parse_projects(section: str) -> List[ProjectItem]:
    if not section:
        return []
    blocks = [b.strip() for b in section.split("\n\n") if b.strip()]
    items: list[ProjectItem] = []
    for block in blocks:
        lines = [ln.strip(" -") for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        name = lines[0]
        links = re.findall(r"https?://[^\s]+", block)
        items.append(ProjectItem(name=name, description=lines[1:], links=links))
    return items


def parse_resume_code_first(text: str) -> Tuple[ParsedResume, ParseDiagnostics]:
    sections = split_sections(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    candidate_name = lines[0] if lines else None
    contact = extract_contact(text)
    summary = sections.get("summary")
    skills = parse_skills(sections.get("skills", ""))
    if not skills:
        skills = _infer_skills_from_text(text)
    experience = parse_experience(sections.get("experience", ""))
    education = parse_education(sections.get("education", ""))
    projects = parse_projects(sections.get("projects", ""))
    certifications = parse_skills(sections.get("certifications", ""))

    resume = ParsedResume(
        candidate_name=candidate_name,
        contact=contact,
        summary=summary[:1200] if summary else None,
        skills=skills,
        experience=experience,
        education=education,
        projects=projects,
        certifications=certifications,
        raw_sections=sections,
    )

    confidence, reasons = parser_confidence(resume, text)
    diagnostics = ParseDiagnostics(
        parser_mode="code_first",
        used_llm_fallback=False,
        confidence=confidence,
        reasons=reasons,
    )
    return resume, diagnostics


def parser_confidence(resume: ParsedResume, raw_text: str) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: list[str] = []

    if len(raw_text) >= 600:
        score += 0.25
    else:
        reasons.append("low_text_density")

    if resume.contact.email:
        score += 0.2
    else:
        reasons.append("missing_email")

    if resume.contact.phone or resume.contact.linkedin or resume.contact.github:
        score += 0.1
    else:
        reasons.append("limited_contact_signals")

    if resume.experience:
        score += 0.2
    else:
        reasons.append("missing_experience")

    if resume.skills:
        score += 0.15
    else:
        reasons.append("missing_skills")

    if resume.education or resume.projects:
        score += 0.2
    else:
        reasons.append("missing_education_projects")

    if resume.summary and len(resume.summary) >= 40:
        score += 0.1
    else:
        reasons.append("weak_summary")

    return min(score, 1.0), reasons


def _split_title_company(header: str) -> Tuple[str, str]:
    if "|" in header:
        left, right = header.split("|", 1)
        return left.strip(), right.strip()
    if " at " in header.lower():
        parts = re.split(r"\s+at\s+", header, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return header.strip(), ""


def _normalize_header(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"[\-:|]+$", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z\s]", "", lowered)
    return lowered.strip()


def _match_section_header(line: str) -> Tuple[str | None, str]:
    stripped = line.strip()
    normalized = _normalize_header(stripped)

    for key, aliases in SECTION_HEADERS.items():
        if normalized in aliases:
            return key, ""

    # Inline header style: "Skills: Python, SQL".
    inline_match = re.match(r"^\s*([A-Za-z\s/&]+?)\s*[:\-]\s*(.+)$", stripped)
    if inline_match:
        header = _normalize_header(inline_match.group(1))
        content = inline_match.group(2).strip()
        for key, aliases in SECTION_HEADERS.items():
            if header in aliases:
                return key, content

    return None, ""


def _normalize_skill_key(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.lower().strip())


def _infer_skills_from_text(text: str) -> List[str]:
    # Lightweight inference for resumes missing a dedicated "Skills" section.
    known = [
        "python",
        "java",
        "javascript",
        "typescript",
        "sql",
        "postgresql",
        "mysql",
        "mongodb",
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "react",
        "angular",
        "node",
        "fastapi",
        "flask",
        "django",
        "kafka",
        "rabbitmq",
        "kinesis",
        "pytorch",
        "tensorflow",
        "nlp",
        "machine learning",
    ]
    lowered = text.lower()
    found: list[str] = []
    for item in known:
        if re.search(rf"(^|[^a-z0-9]){re.escape(item)}([^a-z0-9]|$)", lowered):
            found.append(item)

    # Present skill names with familiar casing.
    casing = {
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "postgresql": "PostgreSQL",
        "aws": "AWS",
        "gcp": "GCP",
        "nlp": "NLP",
        "fastapi": "FastAPI",
        "machine learning": "Machine Learning",
    }
    return [casing.get(item, item.title()) for item in found]
