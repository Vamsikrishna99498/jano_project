from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI
import requests

from src.config import settings
from src.schemas import ParseDiagnostics, ParsedResume


PROMPT_TEMPLATE = """
You are a resume parser.
Return valid JSON only, matching this schema keys exactly:
- candidate_name: string or null
- contact: {{email, phone, location, linkedin, github, website}}
- summary: string or null
- skills: string[]
- experience: [{{title, company, start_date, end_date, description:string[]}}]
- education: [{{degree, institution, start_date, end_date, details:string[]}}]
- projects: [{{name, description:string[], links:string[]}}]
- certifications: string[]
- raw_sections: object

Resume text:
{resume_text}
""".strip()


def run_llm_fallback(raw_text: str) -> tuple[ParsedResume, ParseDiagnostics]:
    mode = settings.llm_mode
    if mode == "none":
        raise RuntimeError("LLM fallback is disabled. Set LLM_MODE to openai or anthropic.")
    if mode == "openai":
        data = _call_openai(raw_text)
    elif mode == "anthropic":
        data = _call_anthropic(raw_text)
    else:
        raise ValueError(f"Unsupported LLM_MODE: {mode}")

    resume = ParsedResume.model_validate(_normalize_payload(data))
    diagnostics = ParseDiagnostics(
        parser_mode=f"llm_{mode}",
        used_llm_fallback=True,
        confidence=0.75,
        reasons=["fallback_invoked_due_to_low_confidence"],
    )
    return resume, diagnostics


def _call_openai(raw_text: str) -> dict[str, Any]:
    api_key = settings.openai_api_key.strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when LLM_MODE=openai.")

    client = OpenAI(
        api_key=api_key,
        timeout=90.0,
    )
    response = client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a strict resume parser. Return valid JSON only.",
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(resume_text=raw_text[:12000]),
            },
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return _to_json(content)


def _call_anthropic(raw_text: str) -> dict[str, Any]:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_MODE=anthropic.")

    payload = {
        "model": settings.llm_model,
        "max_tokens": 2048,
        "temperature": 0,
        "system": "You are a strict resume parser. Return valid JSON only.",
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(resume_text=raw_text[:12000]),
            }
        ],
    }
    response = requests.post(
        f"{settings.anthropic_base_url.rstrip('/')}/v1/messages",
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": settings.anthropic_api_version,
            "content-type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    body = response.json()
    content_blocks = body.get("content", [])
    text_out = ""
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_out += str(block.get("text", ""))
    return _to_json(text_out or "{}")


def _to_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def _normalize_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)

    payload.setdefault("candidate_name", None)
    payload.setdefault("summary", None)
    payload.setdefault("raw_sections", {})

    contact = payload.get("contact")
    if not isinstance(contact, dict):
        contact = {}
    payload["contact"] = {
        "email": contact.get("email"),
        "phone": contact.get("phone"),
        "location": contact.get("location"),
        "linkedin": contact.get("linkedin"),
        "github": contact.get("github"),
        "website": contact.get("website"),
    }

    payload["skills"] = _as_str_list(payload.get("skills"))
    payload["certifications"] = _as_str_list(payload.get("certifications"))

    payload["experience"] = _normalize_items(
        payload.get("experience"),
        fields=("title", "company", "start_date", "end_date"),
        list_field="description",
    )
    payload["education"] = _normalize_items(
        payload.get("education"),
        fields=("degree", "institution", "start_date", "end_date"),
        list_field="details",
    )
    payload["projects"] = _normalize_items(
        payload.get("projects"),
        fields=("name",),
        list_field="description",
        second_list_field="links",
    )

    return payload


def _normalize_items(
    value: Any,
    fields: tuple[str, ...],
    list_field: str,
    second_list_field: str | None = None,
) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else []
    normalized: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {k: item.get(k, "") for k in fields}
        row[list_field] = _as_str_list(item.get(list_field))
        if second_list_field:
            row[second_list_field] = _as_str_list(item.get(second_list_field))
        normalized.append(row)

    return normalized


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if any(sep in stripped for sep in [",", ";", "|", "\n"]):
            parts = re.split(r"[,;|\n]", stripped)
            return [part.strip() for part in parts if part.strip()]
        return [stripped]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []
