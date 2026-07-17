"""
generate_synthetic_dataset.py
------------------------------
Generates a synthetic resume dataset for demonstration / training purposes.

NOTE: This is a SYNTHETIC dataset built from templates + skill vocabulary.
It exists so the training pipeline (models/train.py) is runnable end-to-end
without requiring you to source a real, licensed resume corpus.

For a production-quality model, replace `resume_dataset.csv` with a real
labeled dataset (e.g. a Kaggle resume dataset) that has the same column
schema:
    resume_text, category, experience_level, job_domain
"""

import json
import random
import csv
import os

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "skills_database.json")) as f:
    SKILLS_DB = json.load(f)

CATEGORIES = list(SKILLS_DB.keys())  # used as both "category" and "job_domain" pool
EXPERIENCE_LEVELS = ["Fresher", "Junior", "Mid-Level", "Senior"]

STRONG_VERBS = ["Led", "Built", "Designed", "Implemented", "Optimized", "Developed",
                "Deployed", "Engineered", "Automated", "Delivered"]

PROJECT_NOUNS = ["a scalable web application", "a machine learning pipeline",
                 "an internal dashboard", "a mobile application", "a REST API",
                 "a data processing system", "a recommendation engine",
                 "an automated testing suite", "a cloud deployment pipeline",
                 "a real-time analytics tool"]

EDU_LEVELS = ["B.Tech in Computer Science", "B.Tech in Information Technology",
              "M.Tech in Data Science", "BCA", "MCA", "B.Sc in Computer Science"]

COMPANIES = ["Infosys", "TCS", "a fintech startup", "an e-commerce company",
             "a healthcare tech firm", "a freelance client", "a college research lab",
             "Wipro", "a SaaS startup", "an ed-tech company"]


def make_resume(category, experience_level):
    skills = random.sample(SKILLS_DB[category], k=min(8, len(SKILLS_DB[category])))
    n_exp_lines = {"Fresher": 1, "Junior": 2, "Mid-Level": 3, "Senior": 4}[experience_level]

    lines = []
    lines.append(f"Summary: {experience_level} professional skilled in {', '.join(skills[:4])}.")
    lines.append(f"Education: {random.choice(EDU_LEVELS)}.")
    lines.append("Skills: " + ", ".join(skills) + ".")

    lines.append("Experience:")
    for _ in range(n_exp_lines):
        verb = random.choice(STRONG_VERBS)
        proj = random.choice(PROJECT_NOUNS)
        company = random.choice(COMPANIES)
        metric = random.choice(["improving efficiency by 20%", "reducing load time by 35%",
                                 "serving over 10000 users", "cutting costs by 15%",
                                 "increasing accuracy by 12%"])
        lines.append(f"{verb} {proj} at {company}, {metric}.")

    lines.append("Projects:")
    for _ in range(random.randint(1, 3)):
        proj = random.choice(PROJECT_NOUNS)
        skill_mention = random.choice(skills)
        lines.append(f"Developed {proj} using {skill_mention}.")

    if random.random() > 0.3:
        lines.append("Certifications: Completed an online certification relevant to " +
                      random.choice(skills) + ".")

    return " ".join(lines)


def main():
    rows = []
    for category in CATEGORIES:
        for experience_level in EXPERIENCE_LEVELS:
            # job_domain mirrors category here for simplicity but pulled
            # independently so the model genuinely learns 3 separate heads
            for _ in range(15):
                domain = category if random.random() > 0.15 else random.choice(CATEGORIES)
                text = make_resume(category, experience_level)
                rows.append([text, category, experience_level, domain])

    random.shuffle(rows)

    out_path = os.path.join(BASE_DIR, "resume_dataset.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resume_text", "category", "experience_level", "job_domain"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} synthetic resumes -> {out_path}")


if __name__ == "__main__":
    main()
