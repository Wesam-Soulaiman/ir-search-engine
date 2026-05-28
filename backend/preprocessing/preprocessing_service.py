import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextPreprocessor:
    def __init__(self, language: str = "english", use_stemming: bool = True):
        self.language = language
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer()

        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words(language))

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        if not self.use_stemming:
            return tokens

        return [self.stemmer.stem(token) for token in tokens]

    def preprocess(self, text: str) -> str:
        text = self.normalize_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.apply_stemming(tokens)

        return " ".join(tokens)

    def preprocess_tokens(self, text: str) -> List[str]:
        processed_text = self.preprocess(text)
        return processed_text.split()