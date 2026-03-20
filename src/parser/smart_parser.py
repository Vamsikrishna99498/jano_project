from __future__ import annotations

from src.parser.extractors import (
    extract_text_from_doc,
    extract_text_from_docx,
    extract_text_from_pdf,
)
from src.parser.heuristics import parse_resume_code_first
from src.parser.llm_fallback import run_llm_fallback
from src.schemas import ParseResult


class SmartParser:
    def __init__(self, min_confidence_for_code_first: float = 0.65) -> None:
        self.min_confidence_for_code_first = min_confidence_for_code_first

    def parse(self, file_name: str, content: bytes) -> ParseResult:
        ext = _infer_ext(file_name)
        raw_text = self._extract(ext, content)
        if not raw_text.strip():
            raise ValueError("No readable text extracted from file.")

        resume, diagnostics = parse_resume_code_first(raw_text)

        if diagnostics.confidence < self.min_confidence_for_code_first:
            try:
                fallback_resume, fallback_diag = run_llm_fallback(raw_text)
                resume = fallback_resume
                diagnostics = fallback_diag
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


def _infer_ext(file_name: str) -> str:
    ext = file_name.rsplit(".", maxsplit=1)[-1].lower()
    if ext not in {"pdf", "docx", "doc"}:
        raise ValueError("Only PDF and Word files (.pdf, .docx, .doc) are supported.")
    return ext
