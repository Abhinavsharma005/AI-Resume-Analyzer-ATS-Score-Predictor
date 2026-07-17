"""
pipeline.py
-----------
Orchestrates the full analysis flow used by the Streamlit dashboard:

    resume text + job description text
        -> GRU prediction (category, experience_level, job_domain)
        -> ATS engine (skill extraction, job matching, rule-based quality,
           ATS score, recommendations)
        -> single combined result dict

Kept separate from streamlit_app/ so the pipeline can also be run/tested
from the command line or a notebook without Streamlit installed.
"""

from typing import Dict

from ats_engine.skill_extractor import SkillExtractor
from ats_engine.job_matcher import JobMatcher
from ats_engine.rule_engine import ResumeRuleEngine
from ats_engine.ats_score import ATSScoreCalculator
from ats_engine.recommendation import RecommendationEngine
from models.predict import ResumePredictor
from utils.config import SKILLS_DB_PATH, RESUME_RULES_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


class ResumeAnalysisPipeline:
    """High-level facade tying the GRU model + ATS engine together."""

    def __init__(self):
        logger.info("Loading GRU model and ATS engine components ...")
        self.predictor = ResumePredictor()
        self.skill_extractor = SkillExtractor(SKILLS_DB_PATH)
        self.job_matcher = JobMatcher(self.skill_extractor)
        self.rule_engine = ResumeRuleEngine(RESUME_RULES_PATH)
        self.ats_calculator = ATSScoreCalculator()
        self.recommender = RecommendationEngine()

    def analyze(self, resume_text: str, job_description_text: str) -> Dict:
        """
        Runs the full analysis and returns everything the dashboard needs.
        """
        gru_prediction = self.predictor.predict(resume_text)
        match_report = self.job_matcher.match(resume_text, job_description_text)
        quality_report = self.rule_engine.run_all_checks(resume_text)
        ats_result = self.ats_calculator.compute(
            match_percentage=match_report["match_percentage"],
            quality_score=quality_report["quality_score"],
        )
        recommendations = self.recommender.generate(quality_report, match_report, gru_prediction)

        return {
            "gru_prediction": gru_prediction,
            "skill_match": match_report,
            "resume_quality": quality_report,
            "ats_result": ats_result,
            "recommendations": recommendations,
        }


if __name__ == "__main__":
    import json

    pipeline = ResumeAnalysisPipeline()
    resume_sample = (
        "INFORMATION TECHNOLOGY Summary Dedicated Information Assurance Professional "
        "with 12 years of Enterprise design and engineering methodology. Led a "
        "machine learning pipeline at a fintech startup, improving efficiency by 20%. "
        "Skills: python, tensorflow, keras, pandas, numpy, docker, aws. "
        "Built a recommendation engine serving 10000 users. Contact: a@b.com, 9876543210"
    )
    jd_sample = (
        "Looking for a Machine Learning Engineer with Python, TensorFlow, PyTorch, "
        "NLP, computer vision, and Docker experience."
    )
    result = pipeline.analyze(resume_sample, jd_sample)
    print(json.dumps(result, indent=2))
