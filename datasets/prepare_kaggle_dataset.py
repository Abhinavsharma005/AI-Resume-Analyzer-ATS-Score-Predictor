"""
prepare_kaggle_dataset.py
--------------------------
Converts the real Kaggle "Resume Dataset" (columns: ID, Resume_str,
Resume_html, Category) into the schema required by models/train.py:

    resume_text, category, experience_level, job_domain

This dataset only provides a ground-truth label for one thing: broad job
category (24 classes: HR, FINANCE, CHEF, ENGINEERING, TEACHER, ...). It has
NO ground-truth labels for seniority or a separate "domain" field, so:

  - `category`         <- taken directly from the dataset's `Category` column
                           (real, human-labeled ground truth)
  - `experience_level` <- DERIVED with a rule-based heuristic that scans the
                           resume text for explicit years-of-experience
                           mentions ("15+ years", "5 years of experience", ...)
                           and seniority keywords in job titles (Senior/Lead/
                           Manager/Director vs. Intern/Entry-level). This is a
                           PROXY label, not verified ground truth.
  - `job_domain`        <- mirrors `category` (no independent domain signal
                           exists in the source data). The 3rd GRU head is
                           kept for architectural symmetry with the original
                           design, but in practice `category` and
                           `job_domain` will always agree for this dataset.

IMPORTANT: Because `experience_level` and `job_domain` are heuristic/derived
rather than human-labeled, evaluation accuracy on those two heads should be
read as "how well the GRU learned to reproduce the heuristic", not as a
measure of true seniority-prediction accuracy. This is documented in the
README as a known limitation.
"""

import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "raw_kaggle_resume_dataset.csv")
OUT_PATH = os.path.join(BASE_DIR, "resume_dataset.csv")

SENIOR_KEYWORDS = [
    "senior", "sr.", "lead", "principal", "director", "head of", "manager",
    "vp ", "vice president", "chief", "architect"
]
FRESHER_KEYWORDS = [
    "intern", "internship", "entry level", "entry-level", "trainee",
    "recent graduate", "fresher"
]

YEARS_PATTERN = re.compile(
    r"(\d{1,2})\+?\s*(?:years|yrs)\.?\s*(?:of)?\s*(?:experience|exp)?", re.IGNORECASE
)


def infer_experience_level(text: str) -> str:
    """
    Heuristic seniority classifier used ONLY to create training labels for
    this dataset (which has no seniority ground truth). See module docstring.
    """
    text_lower = str(text).lower()

    # 1. Explicit years-of-experience mentions -> take the max number found
    years_found = [int(y) for y in YEARS_PATTERN.findall(text_lower)]
    max_years = max(years_found) if years_found else None

    if max_years is not None:
        if max_years >= 8:
            return "Senior"
        if max_years >= 4:
            return "Mid-Level"
        if max_years >= 1:
            return "Junior"
        return "Fresher"

    # 2. Fall back to title/keyword cues if no explicit years mentioned
    if any(k in text_lower for k in SENIOR_KEYWORDS):
        return "Senior"
    if any(k in text_lower for k in FRESHER_KEYWORDS):
        return "Fresher"

    # 3. Fall back to resume length as a weak proxy (longer often = more experience)
    word_count = len(text_lower.split())
    if word_count > 900:
        return "Mid-Level"
    return "Junior"


def clean_category_label(raw_category: str) -> str:
    """'INFORMATION-TECHNOLOGY' -> 'Information Technology' (cosmetic only)."""
    return raw_category.replace("-", " ").title()


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Place the raw Kaggle CSV there first "
            "(columns: ID, Resume_str, Resume_html, Category)."
        )

    print(f"Loading raw dataset from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)

    required_cols = {"Resume_str", "Category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing expected column(s): {missing}")

    # Drop empty/near-empty resumes
    df["Resume_str"] = df["Resume_str"].astype(str)
    df = df[df["Resume_str"].str.split().str.len() > 20].copy()

    print("Deriving experience_level (rule-based heuristic) ...")
    df["experience_level"] = df["Resume_str"].apply(infer_experience_level)

    df["category"] = df["Category"].astype(str).apply(clean_category_label)
    df["job_domain"] = df["category"]  # no independent domain signal available
    df["resume_text"] = df["Resume_str"]

    out_df = df[["resume_text", "category", "experience_level", "job_domain"]]

    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out_df)} resumes -> {OUT_PATH}")
    print("\nCategory distribution:\n", out_df["category"].value_counts())
    print("\nExperience level distribution (heuristic-derived):\n",
          out_df["experience_level"].value_counts())


if __name__ == "__main__":
    main()
