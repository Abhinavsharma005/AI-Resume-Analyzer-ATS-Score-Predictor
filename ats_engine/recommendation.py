"""
recommendation.py
------------------
Generates the final, human-readable list of improvement suggestions by
combining outputs from: rule_engine (quality/formatting), job_matcher
(missing skills), and the GRU model's prediction (category/domain), so
suggestions can be tailored (e.g. project recommendations per domain).
"""

from typing import Dict, List


PROJECT_SUGGESTIONS_BY_CATEGORY = {
    # Original tech-focused categories (kept for backward compatibility if this
    # code is reused with the earlier synthetic/tech-only dataset)
    "AI/ML": [
        "Add a project showcasing an end-to-end ML pipeline (data -> model -> deployment).",
        "Include a project that demonstrates model evaluation (accuracy/precision/recall/F1).",
    ],
    "Web Development": [
        "Add a full-stack project with a live deployed link (e.g. Vercel/Render).",
        "Highlight a project using REST/GraphQL APIs with authentication.",
    ],
    "Android Development": [
        "Add a Play-Store-ready or APK-linked Android project.",
        "Highlight a project using Firebase for real-time data or auth.",
    ],
    "Data Analytics": [
        "Add a project with a dashboard (Power BI/Tableau) built from real data.",
        "Include an A/B testing or statistical-analysis project.",
    ],
    "Cloud/DevOps": [
        "Add a project demonstrating a CI/CD pipeline (e.g. GitHub Actions + Docker).",
        "Highlight infrastructure-as-code experience (Terraform/CloudFormation).",
    ],
    "Cybersecurity": [
        "Add a project demonstrating a vulnerability assessment or CTF writeup.",
        "Highlight any security certification or hands-on lab work.",
    ],
    # Categories matching the real Kaggle Resume Dataset (see datasets/prepare_kaggle_dataset.py)
    "Information Technology": [
        "Add a project or initiative showing measurable infrastructure/system impact "
        "(uptime improved, cost reduced, migration completed).",
        "List specific platforms/tools by name (cloud provider, ticketing system, OS).",
    ],
    "Hr": [
        "Quantify recruitment/retention impact (e.g. reduced time-to-hire by X%).",
        "Mention HRIS/ATS platforms used by name.",
    ],
    "Sales": [
        "Quantify sales performance (quota attainment %, revenue generated, deals closed).",
        "Name the CRM/sales tools used.",
    ],
    "Finance": [
        "Quantify financial impact (cost savings, budget size managed, accuracy improvements).",
        "Mention specific financial software/tools used (Excel, QuickBooks, SAP, etc.).",
    ],
    "Business Development": [
        "Quantify partnerships or revenue growth driven.",
        "Highlight a specific market or client segment expanded.",
    ],
}

EXPERIENCE_SUGGESTIONS = {
    "Fresher": "Emphasize internships, academic projects, and hackathons since formal "
               "experience is limited.",
    "Junior": "Quantify the impact of your work more explicitly (%, users, time saved).",
    "Mid-Level": "Highlight ownership -- projects you led end-to-end, not just contributed to.",
    "Senior": "Emphasize leadership, mentorship, and cross-team/strategic impact.",
}


GENERIC_PROJECT_SUGGESTIONS = [
    "Add a specific, measurable accomplishment from your most recent role "
    "(e.g. a process you improved, a target you exceeded).",
    "Include any certifications, training, or ongoing professional development "
    "relevant to this field.",
]


class RecommendationEngine:
    """Aggregates rule-engine + job-matcher + GRU predictions into final suggestions."""

    def generate(self, rule_report: Dict, match_report: Dict, gru_prediction: Dict) -> Dict:
        """
        Args:
            rule_report: output of ResumeRuleEngine.run_all_checks()
            match_report: output of JobMatcher.match()
            gru_prediction: output of ResumePredictor.predict()

        Returns:
            {
                "missing_technical_skills": [...],
                "missing_sections": [...],
                "weak_summary_flag": bool,
                "project_recommendations": [...],
                "experience_suggestions": str,
                "formatting_improvements": [...],
                "all_suggestions": [...]   # flattened, ready to render/export
            }
        """
        category = gru_prediction["category"]["label"]
        experience_level = gru_prediction["experience_level"]["label"]

        missing_skills = match_report["missing_skills"]
        missing_sections = rule_report["missing_sections"]

        weak_summary = (
            rule_report["word_count_flag"] == "too_short"
            or rule_report["quantified_impact_count"] == 0
        )

        # Fall back to generic suggestions for categories outside the curated
        # tech-focused mapping (e.g. this project's default skill dictionary
        # doesn't cover every category in a general-purpose resume dataset).
        project_recs = PROJECT_SUGGESTIONS_BY_CATEGORY.get(category, GENERIC_PROJECT_SUGGESTIONS)
        experience_rec = EXPERIENCE_SUGGESTIONS.get(
            experience_level, "Tailor achievements to the target role's seniority expectations."
        )

        formatting_improvements = list(rule_report["issues"])  # already human-readable

        all_suggestions: List[str] = []

        if missing_skills:
            all_suggestions.append(
                f"Add these missing skills relevant to the target role: {', '.join(missing_skills)}."
            )
        if missing_sections:
            all_suggestions.append(
                f"Add missing resume section(s): {', '.join(missing_sections)}."
            )
        if weak_summary:
            all_suggestions.append(
                "Strengthen your summary/experience with quantified, specific achievements."
            )
        all_suggestions.extend(project_recs)
        all_suggestions.append(experience_rec)
        all_suggestions.extend(formatting_improvements)

        # De-duplicate while preserving order
        seen = set()
        deduped = []
        for s in all_suggestions:
            if s not in seen:
                deduped.append(s)
                seen.add(s)

        return {
            "missing_technical_skills": missing_skills,
            "missing_sections": missing_sections,
            "weak_summary_flag": weak_summary,
            "project_recommendations": project_recs,
            "experience_suggestions": experience_rec,
            "formatting_improvements": formatting_improvements,
            "all_suggestions": deduped,
        }
