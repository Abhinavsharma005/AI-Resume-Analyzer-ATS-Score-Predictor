"""
tokenizer.py
------------
Wraps Keras's Tokenizer + pad_sequences so training and inference always use
the exact same vocabulary and padding scheme. Also provides save/load so the
fitted tokenizer can be persisted alongside the trained model.
"""

import json
import pickle
from typing import List

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ResumeTokenizer:
    """Fits a Keras Tokenizer on cleaned resume text and converts text <-> padded sequences."""

    def __init__(self, vocab_size: int = 10000, max_len: int = 300, oov_token: str = "<OOV>"):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        self._is_fitted = False

    def fit(self, texts: List[str]) -> "ResumeTokenizer":
        """Fit the tokenizer's vocabulary on a list of cleaned texts."""
        self.tokenizer.fit_on_texts(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: List[str]):
        """Convert a list of cleaned texts into a padded integer-sequence array."""
        if not self._is_fitted:
            raise RuntimeError("ResumeTokenizer must be fit() before transform().")
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

    def fit_transform(self, texts: List[str]):
        self.fit(texts)
        return self.transform(texts)

    @property
    def vocabulary_size(self) -> int:
        """Actual vocabulary size learned (capped at self.vocab_size)."""
        return min(self.vocab_size, len(self.tokenizer.word_index) + 1)

    def save(self, path: str) -> None:
        """Persist the fitted tokenizer + config to disk (pickle)."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "tokenizer": self.tokenizer,
                    "vocab_size": self.vocab_size,
                    "max_len": self.max_len,
                    "oov_token": self.oov_token,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "ResumeTokenizer":
        """Load a previously saved tokenizer from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            vocab_size=data["vocab_size"],
            max_len=data["max_len"],
            oov_token=data["oov_token"],
        )
        instance.tokenizer = data["tokenizer"]
        instance._is_fitted = True
        return instance


class LabelEncoderSet:
    """
    Holds and persists the 3 label encoders needed for the multi-output GRU:
    category, experience_level, job_domain.
    """

    def __init__(self):
        from sklearn.preprocessing import LabelEncoder
        self.encoders = {
            "category": LabelEncoder(),
            "experience_level": LabelEncoder(),
            "job_domain": LabelEncoder(),
        }

    def fit_transform(self, df):
        """Fit each encoder on its column and return a dict of encoded label arrays."""
        return {
            name: enc.fit_transform(df[name])
            for name, enc in self.encoders.items()
        }

    def inverse_transform(self, name: str, encoded_value):
        return self.encoders[name].inverse_transform([encoded_value])[0]

    def classes(self, name: str):
        return list(self.encoders[name].classes_)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.encoders, f)

    @classmethod
    def load(cls, path: str) -> "LabelEncoderSet":
        instance = cls()
        with open(path, "rb") as f:
            instance.encoders = pickle.load(f)
        return instance
