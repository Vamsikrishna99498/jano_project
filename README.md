# AI Resume Shortlisting - Phase 1

Phase 1 implements a local-first smart parser pipeline for resumes.

## What is included

- PDF and Word resume ingestion (`.pdf`, `.docx`, `.doc`)
- Separate Job Description input and storage
- Code-first parsing with deterministic section extraction
- LLM fallback for messy layouts (optional)
- PostgreSQL persistence for parsed outputs
- FAISS vector indexing for resume chunks/embeddings
- Streamlit UI for local operation

## Local Architecture

1. User uploads resumes and enters a Job Description in Streamlit.
2. Parser runs deterministic extraction first.
3. Quality gates evaluate parser confidence.
4. If enabled and quality is low, LLM fallback attempts strict JSON extraction.
5. Structured record is stored in PostgreSQL.
6. Resume text embedding is stored in a local FAISS index.

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

## LLM Fallback Modes

- `none`: no fallback.
- `ollama`: free local model hosting (best for privacy, no API cost).

Set using env vars in `.env`.

## Notes

- `.doc` support is best-effort and requires optional `textract` system dependencies.
- Default embedding model is `all-MiniLM-L6-v2`.
- File size enforcement is done in app: ideal `<1MB`, hard limit `2MB`.
