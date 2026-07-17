"""
job_matcher.py
--------------
Compares extracted resume skills against a job description's extracted
skills to compute matched/missing skills. Pure set-based matching --
deliberately simple and explainable (no embedding similarity, per the
project's "no Transformer/BERT" constraint).
"""

from typing import Dict, List

from ats_engine.skill_extractor import SkillExtractor


class JobMatcher:
    """Matches a resume's extracted skills against a job description's required skills."""

    def __init__(self, skill_extractor: SkillExtractor):
        self.skill_extractor = skill_extractor

    def match(self, resume_text: str, job_description_text: str) -> Dict:
        """
        Args:
            resume_text: raw resume text.
            job_description_text: raw job description text.

        Returns:
            {
                "resume_skills": [...],
                "jd_skills": [...],
                "matched_skills": [...],
                "missing_skills": [...],
                "match_percentage": float
            }
        """
        resume_skills = set(self.skill_extractor.extract(resume_text))
        jd_skills = set(self.skill_extractor.extract(job_description_text))

        matched = sorted(resume_skills & jd_skills)
        missing = sorted(jd_skills - resume_skills)

        match_pct = round(100.0 * len(matched) / len(jd_skills), 1) if jd_skills else 0.0

        return {
            "resume_skills": sorted(resume_skills),
            "jd_skills": sorted(jd_skills),
            "matched_skills": matched,
            "missing_skills": missing,
            "match_percentage": match_pct,
        }
