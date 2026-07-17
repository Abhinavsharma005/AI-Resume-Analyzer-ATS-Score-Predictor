"""
skill_extractor.py
-------------------
Extracts skills mentioned in a resume (or job description) by matching
against the predefined skills_database.json dictionary. Pure string/regex
matching -- no ML/DL model involved, fully deterministic and explainable.
"""

import json
import re
from typing import Dict, List, Set


class SkillExtractor:
    """Matches resume/JD text against a curated skill dictionary."""

    def __init__(self, skills_db_path: str):
        with open(skills_db_path, "r") as f:
            self.skills_by_category: Dict[str, List[str]] = json.load(f)

        # Flat set of all known skills across all categories, for quick lookup
        self.all_skills: Set[str] = {
            skill.lower() for skills in self.skills_by_category.values() for skill in skills
        }

    def extract(self, text: str) -> List[str]:
        """
        Extract all known skills mentioned in the given text.

        Uses word-boundary regex matching so e.g. "java" doesn't falsely
        match inside "javascript", and multi-word skills like "machine
        learning" are matched as phrases.

        Args:
            text: raw resume or job description text (any case).

        Returns:
            Sorted list of unique matched skills (lowercase, as they appear
            in the skills database).
        """
        text_lower = text.lower()
        matched = set()

        for skill in self.all_skills:
            pattern = r"(?<![a-zA-Z0-9+#.])" + re.escape(skill) + r"(?![a-zA-Z0-9+#])"
            if re.search(pattern, text_lower):
                matched.add(skill)

        return sorted(matched)

    def extract_by_category(self, text: str) -> Dict[str, List[str]]:
        """Same as extract(), but grouped by skill category."""
        matched_flat = set(self.extract(text))
        result = {}
        for category, skills in self.skills_by_category.items():
            found = sorted({s.lower() for s in skills} & matched_flat)
            if found:
                result[category] = found
        return result

    def best_matching_category(self, text: str) -> str:
        """Returns the skill category with the most matches in the text (heuristic)."""
        grouped = self.extract_by_category(text)
        if not grouped:
            return "Unknown"
        return max(grouped, key=lambda cat: len(grouped[cat]))


if __name__ == "__main__":
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "datasets", "skills_database.json")
    extractor = SkillExtractor(db_path)
    sample = "Experienced with Python, TensorFlow, Keras, React and REST APIs."
    print(extractor.extract(sample))
    print(extractor.extract_by_category(sample))
