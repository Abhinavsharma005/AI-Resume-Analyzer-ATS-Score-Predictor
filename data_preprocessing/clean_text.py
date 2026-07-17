"""
clean_text.py
-------------
NLP text-cleaning pipeline shared by both training and inference so that
the exact same preprocessing is applied at both stages.

Pipeline: lowercase -> remove URLs/emails -> remove punctuation/symbols
          -> remove stopwords -> lemmatize
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data():
    """Download required NLTK corpora if not already present (idempotent)."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_data()


class TextCleaner:
    """Cleans and normalizes raw resume/job-description text for the DL pipeline."""

    def __init__(self, language: str = "english"):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        # Keep a few domain-relevant tokens that default stopword lists might drop
        self._keep_tokens = {"c", "r"}  # e.g. "C" language, "R" language
        self.stop_words -= self._keep_tokens

    def clean(self, text: str) -> str:
        """
        Run the full cleaning pipeline on a raw text string.

        Args:
            text: raw resume or job description text.

        Returns:
            Cleaned, lemmatized, whitespace-joined string ready for tokenization.
        """
        if not text:
            return ""

        text = text.lower()
        text = self._remove_urls_and_emails(text)
        text = self._remove_punctuation_and_symbols(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)
        return " ".join(tokens)

    @staticmethod
    def _remove_urls_and_emails(text: str) -> str:
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        return text

    @staticmethod
    def _remove_punctuation_and_symbols(text: str) -> str:
        # Keep '%' and digits since they matter for quantified-impact detection
        # (that check happens elsewhere on RAW text; here we prep for the DL model)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str):
        """
        Lightweight whitespace tokenizer.

        NOTE: We deliberately avoid nltk.word_tokenize() here -- it depends on
        the 'punkt_tab' resource which may be unavailable in offline/sandboxed
        environments. Since punctuation and digits are already stripped by
        _remove_punctuation_and_symbols(), a simple whitespace split is
        equivalent in practice for this pipeline.
        """
        return text.split()

    def _remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def _lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]


if __name__ == "__main__":
    cleaner = TextCleaner()
    sample = "Led a team of 5 engineers, responsible for building REST APIs using Node.js!!"
    print(cleaner.clean(sample))
