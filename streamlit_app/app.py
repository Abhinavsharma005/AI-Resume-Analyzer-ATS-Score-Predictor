"""
app.py
------
Main Streamlit entry point for ResumeIQ.

Run with:
    streamlit run streamlit_app/app.py

(Run from the project root so the `models`, `ats_engine`, `data_preprocessing`,
and `utils` packages import correctly.)
"""

import os
import sys
import tempfile

import streamlit as st

# Ensure project root is on sys.path when Streamlit runs this file directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_preprocessing.pdf_extractor import ResumeTextExtractor
from utils.pipeline import ResumeAnalysisPipeline
from utils.config import SAMPLE_JD_DIR
from streamlit_app import dashboard
from streamlit_app.report_generator import generate_pdf_report, generate_csv_report


@st.cache_resource(show_spinner="Loading trained GRU model + ATS engine ...")
def load_pipeline():
    return ResumeAnalysisPipeline()


def get_sample_jds():
    samples = {}
    if os.path.isdir(SAMPLE_JD_DIR):
        for fname in sorted(os.listdir(SAMPLE_JD_DIR)):
            if fname.endswith(".txt"):
                with open(os.path.join(SAMPLE_JD_DIR, fname)) as f:
                    samples[fname] = f.read()
    return samples


def main():
    dashboard.render_header()

    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Train the model first: `python -m models.train` (run from the project root).")
        return

    extractor = ResumeTextExtractor()

    with st.sidebar:
        st.header("1. Upload Resume")
        resume_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

        st.header("2. Job Description")
        jd_mode = st.radio("Provide JD via:", ["Paste text", "Use sample", "Upload file"])

        jd_text = ""
        if jd_mode == "Paste text":
            jd_text = st.text_area("Paste job description", height=200)
        elif jd_mode == "Use sample":
            samples = get_sample_jds()
            if samples:
                choice = st.selectbox("Choose a sample JD", list(samples.keys()))
                jd_text = samples[choice]
                st.text_area("Preview", jd_text, height=150, disabled=True)
            else:
                st.warning("No sample job descriptions found.")
        else:
            jd_file = st.file_uploader("Upload JD (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"],
                                        key="jd_upload")
            if jd_file is not None:
                if jd_file.name.endswith(".txt"):
                    jd_text = jd_file.read().decode("utf-8")
                else:
                    with tempfile.NamedTemporaryFile(delete=False,
                                                       suffix=os.path.splitext(jd_file.name)[1]) as tmp:
                        tmp.write(jd_file.read())
                        tmp_path = tmp.name
                    jd_text = extractor.extract(tmp_path)
                    os.unlink(tmp_path)

        analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

    if not analyze_clicked:
        st.info("👈 Upload a resume and provide a job description, then click **Analyze Resume**.")
        return

    if resume_file is None:
        st.error("Please upload a resume file (PDF or DOCX).")
        return
    if not jd_text.strip():
        st.error("Please provide a job description.")
        return

    with st.spinner("Extracting resume text ..."):
        with tempfile.NamedTemporaryFile(delete=False,
                                          suffix=os.path.splitext(resume_file.name)[1]) as tmp:
            tmp.write(resume_file.read())
            tmp_path = tmp.name
        resume_text = extractor.extract(tmp_path)
        os.unlink(tmp_path)

    if not resume_text.strip():
        st.error("Could not extract any text from the uploaded resume. Try a different file.")
        return

    with st.spinner("Running GRU prediction + ATS analysis ..."):
        result = pipeline.analyze(resume_text, jd_text)

    st.success("Analysis complete!")

    dashboard.render_prediction_cards(result["gru_prediction"])
    st.divider()
    dashboard.render_scores(result["ats_result"], result["resume_quality"])
    st.divider()
    dashboard.render_skill_match(result["skill_match"])
    st.divider()
    dashboard.render_section_status(result["resume_quality"])
    st.divider()
    dashboard.render_strengths_and_suggestions(result["resume_quality"], result["recommendations"])
    st.divider()

    st.subheader("📥 Download Report")
    col1, col2 = st.columns(2)
    with col1:
        pdf_bytes = generate_pdf_report(result)
        st.download_button("Download PDF Report", data=pdf_bytes,
                            file_name="resumeiq_report.pdf", mime="application/pdf")
    with col2:
        csv_bytes = generate_csv_report(result)
        st.download_button("Download CSV Report", data=csv_bytes,
                            file_name="resumeiq_report.csv", mime="text/csv")

    with st.expander("🔬 Raw analysis JSON (debug)"):
        st.json(result)


if __name__ == "__main__":
    main()
