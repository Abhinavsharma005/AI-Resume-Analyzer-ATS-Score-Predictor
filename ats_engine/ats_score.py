"""
ats_score.py
------------
Combines the job-description skill-match percentage and the rule-based
resume quality score into a single, transparent overall ATS Score.
"""

from typing import Dict


class ATSScoreCalculator:
    """Computes a weighted overall ATS score from skill-match % and quality score."""

    def __init__(self, skill_match_weight: float = 0.6, quality_weight: float = 0.4):
        if abs((skill_match_weight + quality_weight) - 1.0) > 1e-6:
            raise ValueError("skill_match_weight + quality_weight must sum to 1.0")
        self.skill_match_weight = skill_match_weight
        self.quality_weight = quality_weight

    def compute(self, match_percentage: float, quality_score: float) -> Dict:
        """
        Args:
            match_percentage: 0-100, from JobMatcher.
            quality_score: 0-100, from ResumeRuleEngine.

        Returns:
            {
                "ats_score": float (0-100),
                "skill_match_component": float,
                "quality_component": float,
                "rating": "Excellent" | "Good" | "Fair" | "Needs Improvement"
            }
        """
        skill_component = round(match_percentage * self.skill_match_weight, 1)
        quality_component = round(quality_score * self.quality_weight, 1)
        ats_score = round(skill_component + quality_component, 1)

        rating = self._rate(ats_score)

        return {
            "ats_score": ats_score,
            "skill_match_component": skill_component,
            "quality_component": quality_component,
            "rating": rating,
        }

    @staticmethod
    def _rate(score: float) -> str:
        if score >= 85:
            return "Excellent"
        if score >= 70:
            return "Good"
        if score >= 50:
            return "Fair"
        return "Needs Improvement"
