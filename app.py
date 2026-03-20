from __future__ import annotations

import json

import streamlit as st

from src.pipeline import ResumeIngestionPipeline


IDEAL_FILE_SIZE_BYTES = 1 * 1024 * 1024
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024


@st.cache_resource
def get_pipeline() -> ResumeIngestionPipeline:
    return ResumeIngestionPipeline()


def main() -> None:
    st.set_page_config(page_title="Resume Parser - Phase 1", page_icon="📄", layout="wide")
    st.title("Phase 1: Smart Resume Parser")
    st.caption("Local-first parsing with PostgreSQL + FAISS")

    pipeline = get_pipeline()

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


if __name__ == "__main__":
    main()
