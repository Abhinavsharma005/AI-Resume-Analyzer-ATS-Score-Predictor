"""
report_generator.py
--------------------
Generates a downloadable PDF (and CSV) report summarizing the analysis
result produced by utils.pipeline.ResumeAnalysisPipeline.analyze().
"""

import csv
import io
from typing import Dict

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)


def generate_pdf_report(result: Dict) -> bytes:
    """Builds a PDF report from the analysis result dict and returns raw PDF bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20)
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], spaceBefore=12)
    body_style = styles["BodyText"]

    story = []
    story.append(Paragraph("ResumeIQ — AI Resume Analyzer &amp; ATS Score Report", title_style))
    story.append(Spacer(1, 12))

    gru = result["gru_prediction"]
    ats = result["ats_result"]
    quality = result["resume_quality"]
    match = result["skill_match"]
    recs = result["recommendations"]

    # Summary table
    summary_data = [
        ["Metric", "Value"],
        ["Resume Category", f"{gru['category']['label']} ({gru['category']['confidence']*100:.1f}%)"],
        ["Experience Level", f"{gru['experience_level']['label']} "
                              f"({gru['experience_level']['confidence']*100:.1f}%)"],
        ["Job Domain", f"{gru['job_domain']['label']} ({gru['job_domain']['confidence']*100:.1f}%)"],
        ["ATS Score", f"{ats['ats_score']} / 100 ({ats['rating']})"],
        ["Resume Quality Score", f"{quality['quality_score']} / 100"],
        ["Skill Match %", f"{match['match_percentage']}%"],
    ]
    table = Table(summary_data, colWidths=[180, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    def bullet_section(title, items):
        story.append(Paragraph(title, heading_style))
        if items:
            story.append(ListFlowable(
                [ListItem(Paragraph(str(i), body_style)) for i in items],
                bulletType="bullet",
            ))
        else:
            story.append(Paragraph("None", body_style))

    bullet_section("Matched Skills", match["matched_skills"])
    bullet_section("Missing Skills", match["missing_skills"])
    bullet_section("Resume Strengths", quality["strengths"])
    bullet_section("Improvement Suggestions", recs["all_suggestions"])

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def generate_csv_report(result: Dict) -> bytes:
    """Builds a flat CSV summary of the analysis result and returns raw bytes."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    gru = result["gru_prediction"]
    ats = result["ats_result"]
    quality = result["resume_quality"]
    match = result["skill_match"]

    writer.writerow(["Metric", "Value"])
    writer.writerow(["Resume Category", gru["category"]["label"]])
    writer.writerow(["Experience Level", gru["experience_level"]["label"]])
    writer.writerow(["Job Domain", gru["job_domain"]["label"]])
    writer.writerow(["ATS Score", ats["ats_score"]])
    writer.writerow(["Rating", ats["rating"]])
    writer.writerow(["Resume Quality Score", quality["quality_score"]])
    writer.writerow(["Skill Match %", match["match_percentage"]])
    writer.writerow(["Matched Skills", "; ".join(match["matched_skills"])])
    writer.writerow(["Missing Skills", "; ".join(match["missing_skills"])])
    writer.writerow(["Suggestions", "; ".join(result["recommendations"]["all_suggestions"])])

    return buffer.getvalue().encode("utf-8")
