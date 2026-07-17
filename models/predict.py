"""
predict.py
----------
Inference-only script: loads the trained GRU model + tokenizer + label
encoders and predicts (category, experience_level, job_domain) for new,
unseen resume text. Kept separate from train.py so the Streamlit app can
do fast inference without re-importing training code/dependencies.
"""

import os
from typing import Dict

import numpy as np

from data_preprocessing.clean_text import TextCleaner
from data_preprocessing.tokenizer import ResumeTokenizer, LabelEncoderSet
from models.gru_model import ResumeGRUModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")


class ResumePredictor:
    """Loads trained artifacts once and exposes a simple `.predict(text)` API."""

    def __init__(self, model_path: str = MODEL_PATH, tokenizer_path: str = TOKENIZER_PATH,
                 label_encoders_path: str = LABEL_ENCODERS_PATH):
        for p in (model_path, tokenizer_path, label_encoders_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found. Train the model first with: python -m models.train"
                )

        self.model = ResumeGRUModel.load(model_path)
        self.tokenizer = ResumeTokenizer.load(tokenizer_path)
        self.label_encoders = LabelEncoderSet.load(label_encoders_path)
        self.cleaner = TextCleaner()

    def predict(self, resume_text: str) -> Dict:
        """
        Predict category, experience level, and job domain for a raw resume text.

        Returns a dict with predicted labels and their confidence scores, e.g.:
        {
            "category": {"label": "AI/ML", "confidence": 0.94},
            "experience_level": {"label": "Mid-Level", "confidence": 0.88},
            "job_domain": {"label": "AI/ML", "confidence": 0.91},
        }
        """
        cleaned = self.cleaner.clean(resume_text)
        X = self.tokenizer.transform([cleaned])

        category_probs, experience_probs, domain_probs = self.model.predict(X, verbose=0)

        return {
            "category": self._decode("category", category_probs[0]),
            "experience_level": self._decode("experience_level", experience_probs[0]),
            "job_domain": self._decode("job_domain", domain_probs[0]),
        }

    def _decode(self, head_name: str, probs: np.ndarray) -> Dict:
        idx = int(np.argmax(probs))
        label = self.label_encoders.inverse_transform(head_name, idx)
        confidence = float(probs[idx])
        # Include the full probability distribution for transparency/debugging
        class_names = self.label_encoders.classes(head_name)
        distribution = {cls: round(float(p), 4) for cls, p in zip(class_names, probs)}
        return {"label": label, "confidence": round(confidence, 4), "distribution": distribution}


if __name__ == "__main__":
    predictor = ResumePredictor()
    # Demo text is a real excerpt from the trained dataset's "Information Technology"
    # category (see datasets/resume_dataset.csv), since the model now predicts the
    # 24 broad job categories from the real Kaggle Resume Dataset, not the earlier
    # tech-skill-only category set.
    sample_resume = (
        "INFORMATION TECHNOLOGY Summary Dedicated Information Assurance Professional "
        "well-versed in analyzing and mitigating risk and finding cost-effective "
        "solutions. Excels at boosting performance and productivity by establishing "
        "realistic goals and enforcing deadlines. Versatile IT professional with 12 "
        "years of Enterprise design and engineering methodology. Skills: Enterprise "
        "platforms, Knowledge of Product Lifecycle Management (PLM), Project "
        "tracking, Hardware and software upgrade planning, Product requirements "
        "documentation."
    )
    result = predictor.predict(sample_resume)
    import json
    print(json.dumps(result, indent=2))
