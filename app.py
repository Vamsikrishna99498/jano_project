from __future__ import annotations

import json

import streamlit as st

from src.pipeline import ResumeIngestionPipeline
from src.schemas import ScoringConstraints, ScoringWeights


IDEAL_FILE_SIZE_BYTES = 1 * 1024 * 1024
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024


@st.cache_resource
def get_pipeline() -> ResumeIngestionPipeline:
    return ResumeIngestionPipeline()


def main() -> None:
    st.set_page_config(page_title="Resume AI Assistant", page_icon="📄", layout="wide")
    st.title("Resume AI Assistant")
    st.caption("Phase 1 + Phase 2: parsing and multi-dimensional scoring")

    try:
        pipeline = get_pipeline()
    except Exception as exc:
        st.error("Database connection failed. Please update DATABASE_URL in .env and restart Streamlit.")
        st.code(
            "\n".join(
                [
                    "Example PostgreSQL URL:",
                    "DATABASE_URL=postgresql+psycopg2://postgres:<YOUR_PASSWORD>@localhost:5432/resume_ai",
                    "",
                    "If database does not exist yet:",
                    "createdb resume_ai",
                ]
            )
        )
        st.caption(f"Startup error: {str(exc)}")
        return

    st.subheader("1) Add Job Description")
    with st.form("jd_form"):
        jd_title = st.text_input("Job Title", value="Software Engineer")
        jd_text = st.text_area("Job Description", height=180)
        jd_submit = st.form_submit_button("Save Job Description")

    if jd_submit:
        if not jd_text.strip():
            st.warning("Job Description is required.")
        else:
            jd_id = pipeline.create_job_description(jd_title, jd_text)
            st.success(f"Saved Job Description with ID: {jd_id}")

    jd_rows = pipeline.get_job_descriptions()
    jd_options = {f"{row['id']} - {row['title']}": row["id"] for row in jd_rows}

    st.subheader("2) Upload Resumes")
    selected_jd = st.selectbox(
        "Attach resumes to Job Description",
        options=["None"] + list(jd_options.keys()),
        index=0,
    )
    selected_jd_id = jd_options[selected_jd] if selected_jd != "None" else None

    uploads = st.file_uploader(
        "Upload PDF/Word files",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        help="Ideal size < 1MB, hard limit 2MB per file.",
    )

    if st.button("Parse Uploaded Resumes", type="primary"):
        if not uploads:
            st.warning("Upload at least one resume file.")
            return

        for file in uploads:
            size = file.size or 0
            if size > MAX_FILE_SIZE_BYTES:
                st.error(f"{file.name}: exceeds hard limit (2MB).")
                continue
            if size > IDEAL_FILE_SIZE_BYTES:
                st.info(f"{file.name}: above ideal size (1MB), processing anyway.")

            content = file.read()
            try:
                resume_id, result = pipeline.process_resume(
                    file_name=file.name,
                    content=content,
                    job_description_id=selected_jd_id,
                )
                st.success(
                    f"Parsed {file.name} -> resume_id={resume_id}, confidence={result.diagnostics.confidence:.2f}"
                )
                st.json(json.loads(result.resume.model_dump_json()))
                st.caption(
                    f"Parser mode: {result.diagnostics.parser_mode} | reasons: {', '.join(result.diagnostics.reasons) or 'none'}"
                )
            except Exception as exc:
                st.error(f"{file.name}: {str(exc)}")

    st.subheader("3) Score Resumes (Phase 2)")
    if not jd_rows:
        st.info("Create at least one Job Description to run scoring.")
        return

    with st.form("scoring_form"):
        score_target = st.selectbox(
            "Select Job Description for scoring",
            options=list(jd_options.keys()),
            index=0,
        )

        st.markdown("Recruiter Weight Controls (per role)")
        w_exact = st.slider("Exact Match Weight", min_value=0, max_value=100, value=35)
        w_semantic = st.slider("Semantic Similarity Weight", min_value=0, max_value=100, value=30)
        w_impact = st.slider("Impact Weight", min_value=0, max_value=100, value=20)
        w_ownership = st.slider("Ownership Weight", min_value=0, max_value=100, value=15)

        st.markdown("Strict Rejection Rules")
        min_years = st.number_input("Minimum years of experience", min_value=0.0, max_value=40.0, value=2.0, step=0.5)
        degree_keywords = st.text_input(
            "Required degree keywords (comma-separated)",
            value="",
            help="Example: B.Tech, Computer Science",
        )
        required_certs = st.text_input(
            "Required certifications (comma-separated)",
            value="",
            help="Example: AWS Certified Developer, CKA",
        )
        required_skills = st.text_input(
            "Required skills for exact match (comma-separated)",
            value="",
            help="Example: Python, FastAPI, PostgreSQL",
        )

        score_submit = st.form_submit_button("Run Scoring")

    if score_submit:
        jd_id = jd_options[score_target]
        weights = ScoringWeights(
            exact_match=float(w_exact),
            semantic_similarity=float(w_semantic),
            impact=float(w_impact),
            ownership=float(w_ownership),
        )
        constraints = ScoringConstraints(
            min_years_experience=float(min_years),
            required_degree_keywords=_split_csv(degree_keywords),
            required_certifications=_split_csv(required_certs),
            required_skills=_split_csv(required_skills),
        )

        try:
            score_rows = pipeline.score_resumes_for_job(
                job_description_id=jd_id,
                weights=weights,
                constraints=constraints,
            )
        except Exception as exc:
            st.error(str(exc))
            return

        if not score_rows:
            st.warning("No resumes attached to this Job Description yet.")
            return

        st.success(f"Scored {len(score_rows)} resumes.")
        for rank, row in enumerate(score_rows, start=1):
            title = f"#{rank} | {row.file_name} | Score: {row.total_score:.1f}"
            if row.rejected:
                title += " | REJECTED"
            with st.expander(title, expanded=(rank <= 3)):
                st.write(row.recruiter_explanation)
                if row.rejected and row.rejection_reasons:
                    st.caption("Rejection reasons: " + "; ".join(row.rejection_reasons))
                st.table(
                    [
                        {
                            "Dimension": d.name,
                            "Score": f"{d.score:.1f}",
                            "Note": d.note,
                        }
                        for d in row.dimension_scores
                    ]
                )


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


if __name__ == "__main__":
    main()
