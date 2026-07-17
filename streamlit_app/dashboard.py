"""
dashboard.py
------------
Streamlit rendering components for the ResumeIQ dashboard. Kept separate
from app.py (which handles file upload / orchestration) so the visual
layer is easy to modify independently.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict


def render_header():
    st.set_page_config(page_title="ResumeIQ — AI Resume Analyzer", page_icon="🧠", layout="wide")
    st.title("🧠 ResumeIQ — AI Resume Analyzer & ATS Score Predictor")
    st.caption("Bidirectional GRU (TensorFlow/Keras) + Rule-Based ATS Engine — no BERT/LLM APIs.")


def render_gauge(score: float, title: str) -> go.Figure:
    color = "#22c55e" if score >= 70 else ("#eab308" if score >= 50 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#3f1f1f"},
                {"range": [50, 70], "color": "#3f3a1f"},
                {"range": [70, 100], "color": "#1f3f28"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_prediction_cards(gru_prediction: Dict):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Resume Category", gru_prediction["category"]["label"],
                   f"{gru_prediction['category']['confidence']*100:.1f}% confidence")
    with col2:
        st.metric("Experience Level", gru_prediction["experience_level"]["label"],
                   f"{gru_prediction['experience_level']['confidence']*100:.1f}% confidence")
    with col3:
        st.metric("Job Domain", gru_prediction["job_domain"]["label"],
                   f"{gru_prediction['job_domain']['confidence']*100:.1f}% confidence")


def render_scores(ats_result: Dict, quality_report: Dict):
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(render_gauge(ats_result["ats_score"], "ATS Score"),
                         use_container_width=True)
        st.caption(f"Rating: **{ats_result['rating']}**")
    with col2:
        st.plotly_chart(render_gauge(quality_report["quality_score"], "Resume Quality Score"),
                         use_container_width=True)


def render_skill_match(match_report: Dict):
    st.subheader("🎯 Skill Match Visualization")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Matched Skills", "Missing Skills"],
        y=[len(match_report["matched_skills"]), len(match_report["missing_skills"])],
        marker_color=["#22c55e", "#ef4444"],
        text=[len(match_report["matched_skills"]), len(match_report["missing_skills"])],
        textposition="auto",
    ))
    fig.update_layout(height=320, showlegend=False, yaxis_title="Skill count")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Matched Skills**")
        if match_report["matched_skills"]:
            st.write(", ".join(match_report["matched_skills"]))
        else:
            st.write("_None found_")
    with col2:
        st.markdown("**❌ Missing Skills**")
        if match_report["missing_skills"]:
            st.write(", ".join(match_report["missing_skills"]))
        else:
            st.write("_None — great match!_")


def render_strengths_and_suggestions(quality_report: Dict, recommendations: Dict):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💪 Resume Strengths")
        for s in quality_report["strengths"]:
            st.markdown(f"- {s}")
        if not quality_report["strengths"]:
            st.write("_No notable strengths detected yet._")

    with col2:
        st.subheader("🛠️ Improvement Suggestions")
        for s in recommendations["all_suggestions"]:
            st.markdown(f"- {s}")
        if not recommendations["all_suggestions"]:
            st.write("_No suggestions — resume looks strong!_")


def render_section_status(quality_report: Dict):
    st.subheader("📄 Resume Section Check")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Present**")
        for s in quality_report["present_sections"]:
            st.markdown(f"✅ {s.title()}")
    with col2:
        st.markdown("**Missing**")
        for s in quality_report["missing_sections"]:
            st.markdown(f"⚠️ {s.title()}")
        if not quality_report["missing_sections"]:
            st.markdown("_All required sections present!_")
