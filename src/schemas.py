from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None


class ExperienceItem(BaseModel):
    title: str = ""
    company: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: List[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    degree: str = ""
    institution: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    details: List[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: str = ""
    description: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    candidate_name: Optional[str] = None
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_sections: dict = Field(default_factory=dict)


class ParseDiagnostics(BaseModel):
    parser_mode: str = "code_first"
    used_llm_fallback: bool = False
    confidence: float = 0.0
    reasons: List[str] = Field(default_factory=list)


class ParseResult(BaseModel):
    file_name: str
    file_type: str
    resume: ParsedResume
    diagnostics: ParseDiagnostics
    raw_text: str
