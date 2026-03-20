# AI Resume Shortlisting - Phase 1 + Phase 2

Phase 1 implements a local-first smart parser pipeline for resumes.
Phase 2 adds a multi-dimensional scoring engine with recruiter-configurable weights.

## What is included

- PDF and Word resume ingestion (`.pdf`, `.docx`, `.doc`)
- Separate Job Description input and storage
- Code-first parsing with deterministic section extraction
- LLM fallback for messy layouts (optional)
- PostgreSQL persistence for parsed outputs
- FAISS vector indexing for resume chunks/embeddings
- Streamlit UI for local operation
- Multi-dimensional scoring engine (Exact Match, Semantic Similarity, Impact, Ownership)
- Strict rejection rules (minimum years, degree constraints, certification constraints)
- Recruiter-facing short explanation for each scored resume

## Local Architecture

1. User uploads resumes and enters a Job Description in Streamlit.
2. Parser runs deterministic extraction first.
3. Quality gates evaluate parser confidence.
4. If enabled and quality is low, LLM fallback attempts strict JSON extraction.
5. Structured record is stored in PostgreSQL.
6. Resume text embedding is stored in a local FAISS index.
7. Recruiter selects per-role weights and strict constraints, then runs Phase 2 scoring.

## Phase 2 Scoring

Dimensions:

- Exact Match: required skill matching and JD skill-term overlap.
- Semantic Similarity: local embedding similarity between resume profile text and JD text.
- Impact: quantified outcomes and impact language signals.
- Ownership: lead/ownership language strength.

Strict rejection rules:

- Minimum years of experience.
- Required degree keywords.
- Required certifications.

Scoring target:

- Designed for local execution with practical latency target under 3 seconds per resume on typical developer hardware.

## Why This Transformer

Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

Why chosen in this project:

- Local and free to run, matching your no-paid-services requirement.
- Fast inference suitable for recruiter workflows and sub-3s scoring targets.
- Strong enough semantic quality for JD-resume alignment without heavier infrastructure.
- Reused across FAISS indexing and semantic scoring to keep behavior consistent.

When to upgrade in future:

- If you need higher semantic precision for niche roles, a stronger local model can be added as an optional profile.

## Chunking Strategy (Future)

- Current implementation does not apply chunking in Phase 2 scoring.
- As requested, no chunking strategy will be implemented until you provide your preferred strategy.
- Before any chunking changes, you will be asked to approve exact rules (chunk size, overlap, section-aware behavior, and weighting).

## Setup

1. Install dependencies:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Ensure PostgreSQL is running and update `DATABASE_URL`.

4. Run Streamlit:

```bash
streamlit run app.py
```

## Quick Start (Recruiter/Tester)

Use this if you want to validate the app quickly on a local machine:

One-command bootstrap:

```bash
./setup_local.sh
```

Then run:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Manual steps (alternative):

1. Copy env template:

```bash
cp .env.example .env
```

2. Set your database password inside `.env`:

```env
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_DB_PASSWORD@localhost:5432/resume_ai
```

3. Start app:

```bash
python -m streamlit run app.py
```

4. Open browser URL shown by Streamlit, add one Job Description, then upload resumes.

Security note:

- `.env` is local-only and is intentionally not committed.
- `.env.example` is the safe template committed to GitHub.

## Troubleshooting

- If you see database auth errors, reset postgres password and update `.env` to match.
- If app still shows old DB settings, stop Streamlit and restart it.

## LLM Fallback Modes

- `none`: no fallback.
- `ollama`: free local model hosting (best for privacy, no API cost).

Set using env vars in `.env`.

## Notes

- `.doc` support is best-effort and requires optional `textract` system dependencies.
- Default embedding model is `all-MiniLM-L6-v2`.
- File size enforcement is done in app: ideal `<1MB`, hard limit `2MB`.
