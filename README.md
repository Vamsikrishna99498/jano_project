# AI Resume Shortlisting - Phase 1 + Phase 2

Phase 1 implements a local-first smart parser pipeline for resumes.
Phase 2 adds a multi-dimensional scoring engine with recruiter-configurable weights.

## Assignment Deliverables Mapping

- System Design Document: see `SYSTEM_DESIGN.md`
- Implementation: Option A (Evaluation and Scoring Engine) implemented in this repository
- README: this file includes setup, architecture summary, and execution flow

Current scope statement:

- Deep implementation focus is Option A.
- Option B (Claim Verification) and Option C (Tiering + Question Generator) are included as future-ready architecture plans in `SYSTEM_DESIGN.md`.

## What is included

- PDF and Word resume ingestion (`.pdf`, `.docx`, `.doc`)
- Separate Job Description input and storage
- Dedicated Job Description parser (text/PDF/Word/TXT) for auto extraction
- Code-first parsing with deterministic section extraction
- LLM fallback for messy layouts (optional)
- PostgreSQL persistence for parsed outputs
- FAISS vector indexing for resume chunks/embeddings
- Streamlit UI for local operation
- Multi-dimensional scoring engine (Exact Match, Semantic Similarity, Achievement, Ownership)
- Strict rejection rules (minimum years and degree constraints)
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
- Achievement: quantified outcomes and achievement language signals.
- Ownership: lead/ownership language strength.

Strict rejection rules:

- Minimum years of experience.
- Required degree keywords.

Constraint source:

- Constraints are auto-extracted from the selected Job Description using the JD parser.
- Manual skill/year/degree entry is not required in scoring flow.

Scoring target:

- Designed for local execution with practical latency target under 3 seconds per resume on typical developer hardware.

Scale upgrades implemented for Option A:

- Incremental FAISS upserts by resume id (no full index rebuild per new resume).
- Chunked scoring pipeline for large JD-attached resume sets.
- Bulk score writes to PostgreSQL to reduce per-row DB overhead.
- Queue-friendly vector sync retry path retained for resilient ingestion.

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

## Throughput Benchmark (Option A)

Use this script to estimate scoring throughput and validate readiness for 10,000+ resumes/day:

```bash
python scripts/benchmark_option_a.py --list-jds
python scripts/benchmark_option_a.py --jd-id 1 --warmups 2 --runs 20
```

Optional: include DB score write overhead in benchmark numbers:

```bash
python scripts/benchmark_option_a.py --jd-id 1 --warmups 2 --runs 20 --persist-scores
```

If your machine is slower, run a fast sample benchmark first:

```bash
python scripts/benchmark_option_a.py --jd-id 1 --warmups 0 --runs 3 --max-resumes 300
```

If your JD has too few resumes, seed synthetic benchmark data first:

```bash
python scripts/seed_benchmark_resumes.py --jd-id 1 --count 5000 --batch-size 500
```

Notes:

- Default benchmark mode computes scoring without writing score rows to DB.
- Benchmark output includes p50/p95/p99 runtime and projected rows/day.
- For credible capacity claims, benchmark with at least 1000+ resumes per JD.

Latest measured evidence (local run):

- Dataset: 5000 resumes linked to one JD.
- Compute-only benchmark (`persist_scores=false`, `max_resumes=300`):
	- `throughput_rows_per_second=127.68`
	- `projected_rows_per_day=11,031,865`
	- `meets_target=true`
- DB-inclusive benchmark (`persist_scores=true`, `max_resumes=300`):
	- `throughput_rows_per_second=146.9`
	- `projected_rows_per_day=12,692,106`
	- `meets_target=true`

Practical reading:

- The first run is usually slower due to model warm-up.
- Warm runs better represent steady-state throughput.

## Small Synthetic Testset (Option A)

Inspired by JD-plus-ground-truth style evaluation from resume screening RAG projects,
this repo includes a small synthetic scoring testset for quick local validation.

Included assets:

- Testset: `data/testsets/scoring_small_testset.json`
- Evaluator script: `scripts/run_small_scoring_eval.py`

Run evaluation:

```bash
python scripts/run_small_scoring_eval.py
```

Run with stricter checks and parser-layer e2e validation:

```bash
python scripts/run_small_scoring_eval.py --e2e-parse
```

Custom paths:

```bash
python scripts/run_small_scoring_eval.py \
	--testset data/testsets/scoring_small_testset.json \
	--output data/testsets/results/scoring_small_eval_report.json
```

What is validated:

- Top-1 expected candidate per JD test case
- Top-K expected candidate set inclusion
- Strict rejection behavior for configured constraints
- Strict Top-K order check (when expected order is provided)
- False-positive rejection check via `must_not_reject_ids`
- Optional parser-layer e2e top-1 check (`--e2e-parse`)

## Stability Check (Randomized, Small Scale)

You can also run repeated randomized checks to see if ranking decisions stay stable
when scoring weights are slightly perturbed.

Script:

- `scripts/run_small_scoring_stability.py`

Example run:

```bash
python scripts/run_small_scoring_stability.py --iterations 20 --weight-jitter 0.15
```

Stricter mode (weight jitter + candidate text perturbation):

```bash
python scripts/run_small_scoring_stability.py \
	--iterations 20 \
	--weight-jitter 0.15 \
	--perturb-text \
	--text-jitter 0.20
```

Output report:

- `data/testsets/results/scoring_small_stability_report.json`

Key outputs:

- top1_stability per case
- topk_stability per case
- reject_stability per case
- average stability metrics across all cases

## Parser QA (Mildly Unstructured Samples)

To prevent regressions in parser behavior for slightly messy resume text,
run parser QA fixtures:

- Input fixtures: `data/testsets/parser_qa_samples.json`
- Runner: `scripts/run_parser_qa.py`

Command:

```bash
python scripts/run_parser_qa.py
```

Report output:

- `data/testsets/results/parser_qa_report.json`

Checks include:

- required skill extraction
- minimum experience entries
- contact email capture
- summary capture

## LLM Fallback Modes

- `none`: no fallback.
- `openai`: hosted OpenAI models (set `OPENAI_API_KEY`).
- `anthropic`: hosted Anthropic models (set `ANTHROPIC_API_KEY`).

Set using env vars in `.env`.

Example:

```bash
# OpenAI
LLM_MODE=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here

# Anthropic
LLM_MODE=anthropic
LLM_MODEL=claude-3-5-haiku-latest
ANTHROPIC_API_KEY=your_key_here
```

## Notes

- `.doc` support is best-effort and requires optional `textract` system dependencies.
- Default embedding model is `all-MiniLM-L6-v2`.
- File size enforcement is done in app: ideal `<1MB`, hard limit `2MB`.
