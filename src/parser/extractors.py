from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pdfplumber
from docx import Document


def extract_text_from_pdf(content: bytes) -> str:
    text_chunks: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                text_chunks.append(text)
    return "\n".join(text_chunks).strip()


def extract_text_from_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines).strip()


def extract_text_from_doc(content: bytes) -> str:
    # .doc is best-effort via textract if available.
    try:
        import textract  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "DOC parsing requires optional dependency 'textract' and system tools."
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        out = textract.process(str(tmp_path))
        return out.decode("utf-8", errors="ignore").strip()
    finally:
        tmp_path.unlink(missing_ok=True)
