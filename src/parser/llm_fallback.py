from __future__ import annotations

import json
from typing import Any

import requests

from src.config import settings
from src.schemas import ParseDiagnostics, ParsedResume


PROMPT_TEMPLATE = """
You are a resume parser.
Return valid JSON only, matching this schema keys exactly:
- candidate_name: string or null
- contact: {email, phone, location, linkedin, github, website}
- summary: string or null
- skills: string[]
- experience: [{title, company, start_date, end_date, description:string[]}]
- education: [{degree, institution, start_date, end_date, details:string[]}]
- projects: [{name, description:string[], links:string[]}]
- certifications: string[]
- raw_sections: object

Resume text:
{resume_text}
""".strip()


def run_llm_fallback(raw_text: str) -> tuple[ParsedResume, ParseDiagnostics]:
    mode = settings.llm_mode
    if mode == "none":
        raise RuntimeError("LLM fallback is disabled. Set LLM_MODE to ollama.")
    if mode == "ollama":
        data = _call_ollama(raw_text)
    else:
        raise ValueError(f"Unsupported LLM_MODE: {mode}")

    resume = ParsedResume.model_validate(data)
    diagnostics = ParseDiagnostics(
        parser_mode=f"llm_{mode}",
        used_llm_fallback=True,
        confidence=0.75,
        reasons=["fallback_invoked_due_to_low_confidence"],
    )
    return resume, diagnostics


def _call_ollama(raw_text: str) -> dict[str, Any]:
    payload = {
        "model": settings.llm_model,
        "prompt": PROMPT_TEMPLATE.format(resume_text=raw_text[:12000]),
        "stream": False,
        "format": "json",
    }
    response = requests.post(
        f"{settings.ollama_base_url}/api/generate", json=payload, timeout=90
    )
    response.raise_for_status()
    body = response.json()
    return _to_json(body.get("response", "{}"))


def _to_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)
