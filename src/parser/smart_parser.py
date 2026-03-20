from __future__ import annotations

from src.config import settings
from src.parser.extractors import (
    extract_text_from_doc,
    extract_text_from_docx,
    extract_text_from_pdf,
)
from src.parser.heuristics import parse_resume_code_first
from src.parser.llm_fallback import run_llm_fallback
from src.schemas import ParseDiagnostics, ParseResult, ParsedResume


class SmartParser:
    def __init__(self, min_confidence_for_code_first: float = 0.65) -> None:
        self.min_confidence_for_code_first = min_confidence_for_code_first

    def parse(self, file_name: str, content: bytes) -> ParseResult:
        ext = _infer_ext(file_name)
        raw_text = self._extract(ext, content)
        if not raw_text.strip():
            raise ValueError("No readable text extracted from file.")

        if settings.force_llm_only:
            fallback_resume, fallback_diag = run_llm_fallback(raw_text)
            fallback_diag.reasons.append("forced_llm_only_mode")
            return ParseResult(
                file_name=file_name,
                file_type=ext,
                resume=fallback_resume,
                diagnostics=fallback_diag,
                raw_text=raw_text,
            )

        resume, diagnostics = parse_resume_code_first(raw_text)

        if self._needs_fallback(resume, diagnostics):
            try:
                fallback_resume, fallback_diag = run_llm_fallback(raw_text)
                merged_resume = self._merge_resumes(primary=resume, secondary=fallback_resume)

                primary_score = self._resume_completeness_score(resume)
                fallback_score = self._resume_completeness_score(fallback_resume)
                merged_score = self._resume_completeness_score(merged_resume)

                if merged_score >= max(primary_score, fallback_score):
                    resume = merged_resume
                    diagnostics.parser_mode = "code_first_plus_llm_merge"
                    diagnostics.used_llm_fallback = True
                    diagnostics.confidence = max(diagnostics.confidence, fallback_diag.confidence)
                    diagnostics.reasons.append("fallback_merge_applied")
                elif fallback_score > primary_score:
                    resume = fallback_resume
                    diagnostics = fallback_diag
                    diagnostics.reasons.append("fallback_replaced_code_first")
                else:
                    diagnostics.used_llm_fallback = True
                    diagnostics.reasons.append("fallback_kept_code_first")
            except Exception as exc:
                diagnostics.reasons.append(f"llm_fallback_failed:{str(exc)}")

        return ParseResult(
            file_name=file_name,
            file_type=ext,
            resume=resume,
            diagnostics=diagnostics,
            raw_text=raw_text,
        )

    def _extract(self, ext: str, content: bytes) -> str:
        if ext == "pdf":
            return extract_text_from_pdf(content)
        if ext == "docx":
            return extract_text_from_docx(content)
        if ext == "doc":
            return extract_text_from_doc(content)
        raise ValueError(f"Unsupported file type: {ext}")

    def _needs_fallback(self, resume: ParsedResume, diagnostics: ParseDiagnostics) -> bool:
        if diagnostics.confidence < self.min_confidence_for_code_first:
            return True

        # Trigger fallback for sparse yet structurally parseable resumes.
        sparse_signals = 0
        if not resume.skills:
            sparse_signals += 1
        if not resume.experience:
            sparse_signals += 1
        if not resume.summary or len((resume.summary or "").strip()) < 30:
            sparse_signals += 1
        if not (resume.contact.email or resume.contact.phone or resume.contact.linkedin):
            sparse_signals += 1

        return sparse_signals >= 2

    def _resume_completeness_score(self, resume: ParsedResume) -> float:
        score = 0.0
        if resume.candidate_name:
            score += 0.5
        if resume.contact.email:
            score += 0.7
        if resume.contact.phone or resume.contact.linkedin or resume.contact.github:
            score += 0.5
        if resume.summary and len(resume.summary.strip()) >= 30:
            score += 0.8
        score += min(1.5, 0.25 * len(resume.skills))
        score += min(2.0, 0.6 * len(resume.experience))
        score += min(1.0, 0.5 * len(resume.education))
        score += min(0.7, 0.35 * len(resume.projects))
        return score

    def _merge_resumes(self, primary: ParsedResume, secondary: ParsedResume) -> ParsedResume:
        # Keep strong code-first structure but fill missing/weak fields from fallback.
        merged = primary.model_dump()
        alt = secondary.model_dump()

        if not merged.get("candidate_name") and alt.get("candidate_name"):
            merged["candidate_name"] = alt.get("candidate_name")

        if (not merged.get("summary") or len(str(merged.get("summary") or "").strip()) < 30) and alt.get("summary"):
            merged["summary"] = alt.get("summary")

        for key in ["email", "phone", "location", "linkedin", "github", "website"]:
            if not merged["contact"].get(key) and alt.get("contact", {}).get(key):
                merged["contact"][key] = alt["contact"][key]

        merged["skills"] = self._merge_unique_list(merged.get("skills", []), alt.get("skills", []))

        if len(merged.get("experience", [])) < len(alt.get("experience", [])):
            merged["experience"] = alt.get("experience", [])

        if len(merged.get("education", [])) < len(alt.get("education", [])):
            merged["education"] = alt.get("education", [])

        if len(merged.get("projects", [])) < len(alt.get("projects", [])):
            merged["projects"] = alt.get("projects", [])

        merged["certifications"] = self._merge_unique_list(
            merged.get("certifications", []),
            alt.get("certifications", []),
        )
        return ParsedResume.model_validate(merged)

    def _merge_unique_list(self, left: list[str], right: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in [*left, *right]:
            value = str(item).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out


def _infer_ext(file_name: str) -> str:
    ext = file_name.rsplit(".", maxsplit=1)[-1].lower()
    if ext not in {"pdf", "docx", "doc"}:
        raise ValueError("Only PDF and Word files (.pdf, .docx, .doc) are supported.")
    return ext
