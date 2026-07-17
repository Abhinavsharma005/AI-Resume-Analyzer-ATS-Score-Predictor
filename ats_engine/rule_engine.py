"""
rule_engine.py
--------------
Rule-based, fully-explainable resume quality checks: missing sections,
weak/passive language, word count, quantified-impact detection, contact
info presence, etc. Everything here is deterministic (regex/keyword based)
-- deliberately NOT a black-box model, so every flagged issue can be traced
back to an exact rule.
"""

import json
import re
from typing import Dict, List


class ResumeRuleEngine:
    """Runs rule-based quality/formatting checks against raw resume text."""

    def __init__(self, rules_path: str):
        with open(rules_path, "r") as f:
            self.rules = json.load(f)

    def run_all_checks(self, resume_text: str) -> Dict:
        """
        Runs every rule-based check and returns a structured report.

        Returns:
            {
                "missing_sections": [...],
                "present_sections": [...],
                "weak_verb_hits": [...],
                "passive_voice_hits": [...],
                "quantified_impact_count": int,
                "word_count": int,
                "word_count_flag": "ok" | "too_short" | "too_long",
                "has_email": bool,
                "has_phone": bool,
                "quality_score": float (0-100),
                "strengths": [...],
                "issues": [...],
            }
        """
        text_lower = resume_text.lower()

        section_report = self._check_sections(text_lower)
        weak_verbs = self._find_weak_verbs(text_lower)
        passive_hits = self._find_passive_voice(text_lower)
        quantified_count = self._count_quantified_impact(text_lower)
        word_count = len(resume_text.split())
        word_count_flag = self._check_word_count(word_count)
        has_email = bool(re.search(r"\S+@\S+\.\S+", resume_text))
        has_phone = bool(re.search(r"(\+?\d[\d\-\s]{8,}\d)", resume_text))
        strong_verb_count = self._count_strong_verbs(text_lower)

        quality_score, strengths, issues = self._score(
            section_report, weak_verbs, passive_hits, quantified_count,
            word_count_flag, has_email, has_phone, strong_verb_count
        )

        return {
            "missing_sections": section_report["missing"],
            "present_sections": section_report["present"],
            "weak_verb_hits": weak_verbs,
            "passive_voice_hits": passive_hits,
            "quantified_impact_count": quantified_count,
            "strong_verb_count": strong_verb_count,
            "word_count": word_count,
            "word_count_flag": word_count_flag,
            "has_email": has_email,
            "has_phone": has_phone,
            "quality_score": quality_score,
            "strengths": strengths,
            "issues": issues,
        }

    def _check_sections(self, text_lower: str) -> Dict[str, List[str]]:
        required = self.rules["required_sections"]
        present, missing = [], []
        for section in required:
            # Match section as a heading-ish word (e.g. "Experience:" or "EXPERIENCE")
            if re.search(r"\b" + re.escape(section) + r"\b", text_lower):
                present.append(section)
            else:
                missing.append(section)
        return {"present": present, "missing": missing}

    def _find_weak_verbs(self, text_lower: str) -> List[str]:
        found = []
        for verb in self.rules["weak_verbs"]:
            if re.search(r"\b" + re.escape(verb) + r"\b", text_lower):
                found.append(verb)
        return found

    def _count_strong_verbs(self, text_lower: str) -> int:
        count = 0
        for verb in self.rules["strong_verbs"]:
            count += len(re.findall(r"\b" + re.escape(verb) + r"\b", text_lower))
        return count

    def _find_passive_voice(self, text_lower: str) -> List[str]:
        found = []
        for pattern in self.rules["passive_voice_patterns"]:
            if pattern in text_lower:
                found.append(pattern)
        return found

    def _count_quantified_impact(self, text_lower: str) -> int:
        count = 0
        for indicator in self.rules["quantification_indicators"]:
            count += len(re.findall(re.escape(indicator.lower()), text_lower))
        return count

    def _check_word_count(self, word_count: int) -> str:
        checks = self.rules["formatting_checks"]
        if word_count < checks["min_recommended_words"]:
            return "too_short"
        if word_count > checks["max_recommended_words"]:
            return "too_long"
        return "ok"

    def _score(self, section_report, weak_verbs, passive_hits, quantified_count,
               word_count_flag, has_email, has_phone, strong_verb_count):
        """
        Simple, transparent additive scoring rubric (out of 100).
        Every deduction/addition is traceable -- no black box.
        """
        score = 100.0
        strengths, issues = [], []

        # Sections (40 pts total)
        missing = section_report["missing"]
        deduction_per_section = 40 / max(len(self.rules["required_sections"]), 1)
        score -= deduction_per_section * len(missing)
        if missing:
            issues.append(f"Missing required section(s): {', '.join(missing)}.")
        else:
            strengths.append("All required resume sections are present.")

        # Contact info (10 pts)
        if not has_email:
            score -= 5
            issues.append("No email address detected.")
        else:
            strengths.append("Email address found.")
        if not has_phone:
            score -= 5
            issues.append("No phone number detected.")
        else:
            strengths.append("Phone number found.")

        # Weak verbs / passive voice (20 pts)
        if weak_verbs:
            penalty = min(10, 2 * len(weak_verbs))
            score -= penalty
            issues.append(
                f"Uses weak phrasing ({', '.join(sorted(set(weak_verbs)))}); "
                "prefer strong action verbs."
            )
        if passive_hits:
            penalty = min(10, 3 * len(passive_hits))
            score -= penalty
            issues.append("Passive voice detected in places; rewrite in active voice.")
        if strong_verb_count >= 3:
            strengths.append(f"Good use of strong action verbs ({strong_verb_count} found).")

        # Quantified impact (20 pts)
        if quantified_count == 0:
            score -= 20
            issues.append("No quantified achievements found (numbers, %, metrics).")
        elif quantified_count < 3:
            score -= 10
            issues.append("Few quantified achievements; add more measurable results.")
        else:
            strengths.append(f"Good use of quantified achievements ({quantified_count} instances).")

        # Word count (10 pts)
        if word_count_flag == "too_short":
            score -= 10
            issues.append("Resume content seems too short; add more detail.")
        elif word_count_flag == "too_long":
            score -= 5
            issues.append("Resume content seems too long; consider trimming to 1-2 pages.")
        else:
            strengths.append("Resume length is within the recommended range.")

        score = max(0.0, min(100.0, round(score, 1)))
        return score, strengths, issues


if __name__ == "__main__":
    import os
    rules_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "datasets", "resume_rules.json")
    engine = ResumeRuleEngine(rules_path)
    sample = (
        "Education: B.Tech. Skills: Python, TensorFlow. "
        "I was responsible for helping the team. Contact: test@email.com, +91 9876543210"
    )
    print(json.dumps(engine.run_all_checks(sample), indent=2))
