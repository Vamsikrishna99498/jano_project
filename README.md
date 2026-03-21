# AI Resume Assistant

Local-first resume parsing and scoring system built with Streamlit, PostgreSQL, and FAISS.

Implemented focus is Option A: parser + explainable scoring engine.

## Demo Video

Watch the product walkthrough below.

<p align="center">
  <a href="https://github.com/user-attachments/assets/25c44cd5-7a73-47fd-869a-812f30bd9132">
    <img alt="Watch Demo Video" src="https://img.shields.io/badge/Watch-Demo%20Video-2ea44f?style=for-the-badge&logo=github" />
  </a>
</p>

<p align="center">
  <a href="https://github.com/user-attachments/assets/25c44cd5-7a73-47fd-869a-812f30bd9132">Open video on GitHub</a>
</p>

## Introduction

AI Resume Assistant helps recruiters and hiring teams quickly evaluate resumes against a job description.

It takes uploaded resumes, converts them into structured candidate profiles, and scores each candidate across four dimensions: Exact Match, Semantic Similarity, Achievement, and Ownership. The system also applies strict rejection checks (experience and degree constraints) and produces recruiter-friendly explainability output for every score.

The project is designed to run locally with low infrastructure complexity. You can use deterministic code-first parsing by default, and optionally enable LLM fallback only for difficult resume layouts.

## What This Project Does

- Ingest resumes in `.pdf`, `.docx`, and `.doc` formats.
- Ingest job descriptions as text or uploaded `.pdf`/`.docx`/`.doc`/`.txt`.
- Parse resumes with a deterministic code-first parser.
- Optionally use LLM fallback (OpenAI or Anthropic) when parse quality is low.
- Persist parsed data and scoring outputs in PostgreSQL.
- Store local vector index in FAISS for semantic workflows.
- Rank candidates with weighted multi-dimensional scoring.
- Generate recruiter-facing explainability output for each score.

## Tech Stack

- UI: Streamlit
- Parsing: pdfplumber, python-docx, optional textract for `.doc`
- Data validation: Pydantic v2
- Storage: PostgreSQL (SQLAlchemy)
- Vector index: FAISS (local file index)
- Embeddings: sentence-transformers (`all-MiniLM-L6-v2` by default)
- Optional LLM fallback: OpenAI SDK and Anthropic HTTP API

## Repository Layout

- `app.py`: Streamlit app
- `src/pipeline.py`: orchestration of parse, persistence, vector sync, and scoring
- `src/parser/`: extractors, heuristic parser, JD parser, LLM fallback
- `src/scoring/engine.py`: score computation + explainability
- `src/storage/`: PostgreSQL store and FAISS vector store
- `scripts/`: benchmarks and validation scripts
- `data/testsets/`: synthetic QA/eval datasets and sample reports

## Quick Start

1. Bootstrap locally:

```bash
./setup_local.sh
```

2. Create/update env:

```bash
cp .env.example .env
```

3. Set DB connection in `.env`:

```env
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_DB_PASSWORD@localhost:5432/resume_ai
```

4. Run app:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

## Runtime Flow

1. Add a job description.
2. Upload resumes and parse.
3. Parsed result is stored in `resumes`, and a vector sync job is queued.
4. Pipeline attempts immediate FAISS upsert; failed upserts stay queued for retry.
5. Run scoring for selected JD with recruiter-configurable weights.
6. Score rows are persisted in `resume_scores` with dimensions and explanation.

## Parser Modes

Parser controls are configured in `.env`.

- `LLM_MODE=none` and `FORCE_LLM_ONLY=false`:
  - Pure code-first parser (recommended default local mode).
- `LLM_MODE=openai` and `FORCE_LLM_ONLY=false`:
  - Code-first parser, OpenAI fallback only when needed.
- `LLM_MODE=anthropic` and `FORCE_LLM_ONLY=false`:
  - Code-first parser, Anthropic fallback only when needed.
- `FORCE_LLM_ONLY=true`:
  - Skip code parser and use configured LLM path directly.

The parser fallback decision is based on confidence and sparse-content signals.

## Scoring Model

Dimensions:

- Exact Match
- Semantic Similarity
- Achievement
- Ownership

Final score is weighted using normalized recruiter-configured weights. Strict-rule failures mark candidates as rejected and cap final score at 40.

Strict rules:

- minimum years of experience (when inferrable)
- required degree keywords

Constraints are auto-derived from JD text by the JD parser.

## Explainability Output

Each scored row includes:

- per-dimension score + note
- strict-rule rejection reasons
- recruiter explanation payload (`recruiter_explanation`) stored as JSON string

UI displays this explanation directly in scoring results.

## Environment Variables

See `.env.example` for all options and use-case presets.

Core variables:

- `DATABASE_URL`
- `VECTOR_INDEX_PATH`
- `VECTOR_META_PATH`
- `EMBEDDING_MODEL`
- `LLM_MODE`
- `LLM_MODEL`
- `FORCE_LLM_ONLY`

Provider variables (optional):

- OpenAI: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_PROJECT`, `OPENAI_ORGANIZATION`
- Anthropic: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_VERSION`

## Benchmark and Validation Scripts

List JDs:

```bash
python scripts/benchmark_option_a.py --list-jds
```

Run throughput benchmark:

```bash
python scripts/benchmark_option_a.py --jd-id 1 --warmups 2 --runs 20
```

Include DB write cost during benchmark:

```bash
python scripts/benchmark_option_a.py --jd-id 1 --warmups 2 --runs 20 --persist-scores
```

Seed synthetic resumes for benchmarking:

```bash
python scripts/seed_benchmark_resumes.py --jd-id 1 --count 5000 --batch-size 500
```

Run small scoring eval:

```bash
python scripts/run_small_scoring_eval.py
```

Run eval with parser-layer e2e mode:

```bash
python scripts/run_small_scoring_eval.py --e2e-parse
```

Run scoring stability checks:

```bash
python scripts/run_small_scoring_stability.py --iterations 20 --weight-jitter 0.15
```

Run parser QA fixtures:

```bash
python scripts/run_parser_qa.py
```

Run all scripts at once (recommended order):

```bash
set -e && \
python scripts/seed_benchmark_resumes.py --jd-id 1 --count 1000 --batch-size 500 && \
python scripts/benchmark_option_a.py --jd-id 1 --runs 5 --warmups 1 && \
python scripts/run_parser_qa.py && \
python scripts/run_small_scoring_eval.py --e2e-parse && \
python scripts/run_small_scoring_stability.py --perturb-text --iterations 20
```

## Validation Notes

Detailed validation rationale and latest run outcomes are documented in `SYSTEM_DESIGN.md` under the validation section.

## Database Reset (Local)

Reset all project tables by recreating `public` schema:

```bash
set -e
DATABASE_URL=$(grep -E '^DATABASE_URL=' .env | head -n1 | cut -d'=' -f2-)
PSQL_URL="${DATABASE_URL/+psycopg2/}"
psql -w "$PSQL_URL" -v ON_ERROR_STOP=1 -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
```

Start the app again to recreate tables.

## Notes and Limitations

- `.doc` parsing requires optional `textract` plus system-level dependencies.
- FAISS index and metadata are stored as local files under `data/` by default.
- LLM fallback is optional; full workflow runs without paid APIs in code-first mode.
- `.env` is local-only and should not be committed.
