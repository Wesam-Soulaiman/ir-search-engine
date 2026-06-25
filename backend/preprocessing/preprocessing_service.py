import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ftfy import fix_text
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Tokens that carry important semantic meaning and must not be removed
# even if they are present in a standard English stop-word list.
NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "without",
}


URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

HTML_TAG_PATTERN = re.compile(
    r"<[^>]+>",
)

WHITESPACE_PATTERN = re.compile(
    r"\s+",
)

# This token pattern preserves medical and alphanumeric expressions such as:
#
# BRAF
# V600E
# HER2-positive
# 64-year-old
# 20
# mg/kg/day
# 150/100
# stage-iii
#
# [^\W_] means a Unicode letter or digit, excluding underscores.
TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:[./:+\-][^\W_]+)*",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class PreprocessingProfile:
    """
    Configuration used to preprocess a particular dataset.
    """

    remove_stopwords: bool
    use_stemming: bool
    minimum_token_length: int
    preserve_negations: bool = True


DEFAULT_PROFILE = PreprocessingProfile(
    remove_stopwords=True,
    use_stemming=True,
    minimum_token_length=2,
    preserve_negations=True,
)


DATASET_PROFILES: Dict[str, PreprocessingProfile] = {
    "sample_dataset": PreprocessingProfile(
        remove_stopwords=True,
        use_stemming=True,
        minimum_token_length=2,
        preserve_negations=True,
    ),

    # Quora contains natural-language questions.
    # Stemming and English stop-word removal are useful for matching
    # alternative grammatical forms of similar questions.
    "quora": PreprocessingProfile(
        remove_stopwords=True,
        use_stemming=True,
        minimum_token_length=2,
        preserve_negations=True,
    ),

    # Clinical Trials contains gene names, mutations, ages, dosages,
    # trial stages, and medical terms. Stemming can damage those tokens,
    # so it is disabled. One-character numeric tokens are retained because
    # values such as stage 1 or type 2 may be meaningful.
    "clinical_trials": PreprocessingProfile(
        remove_stopwords=True,
        use_stemming=False,
        minimum_token_length=1,
        preserve_negations=True,
    ),
}


DATASET_KEY_ALIASES = {
    "clinicaltrials": "clinical_trials",
    "clinical-trials": "clinical_trials",
    "clinical_trials": "clinical_trials",
    "quora": "quora",
    "sample": "sample_dataset",
    "sample_dataset": "sample_dataset",
}


class TextPreprocessor:
    """
    Dataset-aware text preprocessing service.

    The same service must be used when preprocessing documents and
    processing search queries. This ensures that query tokens and index
    tokens remain compatible.

    Parameters can be overridden explicitly for experiments, while the
    default behavior is selected from the dataset profile.
    """

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        language: str = "english",
        use_stemming: Optional[bool] = None,
        remove_stopwords: Optional[bool] = None,
        minimum_token_length: Optional[int] = None,
        preserve_negations: Optional[bool] = None,
    ):
        if language.lower() != "english":
            raise ValueError(
                "Only English preprocessing is currently supported."
            )

        normalized_dataset_key = self._normalize_dataset_key(
            dataset_key
        )

        profile = DATASET_PROFILES.get(
            normalized_dataset_key,
            DEFAULT_PROFILE,
        )

        self.dataset_key = normalized_dataset_key
        self.language = language.lower()

        self.use_stemming = (
            profile.use_stemming
            if use_stemming is None
            else bool(use_stemming)
        )

        self.remove_stopwords_enabled = (
            profile.remove_stopwords
            if remove_stopwords is None
            else bool(remove_stopwords)
        )

        self.minimum_token_length = (
            profile.minimum_token_length
            if minimum_token_length is None
            else int(minimum_token_length)
        )

        self.preserve_negations = (
            profile.preserve_negations
            if preserve_negations is None
            else bool(preserve_negations)
        )

        if self.minimum_token_length < 1:
            raise ValueError(
                "minimum_token_length must be at least 1."
            )

        # PorterStemmer does not require an external NLTK corpus.
        self.stemmer = PorterStemmer()

        # sklearn's stop-word list is bundled with the installed package,
        # so preprocessing does not attempt an internet download.
        self.stop_words = self._build_stop_words()

    @staticmethod
    def _normalize_dataset_key(dataset_key: Any) -> str:
        if dataset_key is None:
            return "sample_dataset"

        key = str(dataset_key).strip().lower()

        if not key:
            return "sample_dataset"

        return DATASET_KEY_ALIASES.get(key, key)

    def _build_stop_words(self) -> set[str]:
        if not self.remove_stopwords_enabled:
            return set()

        stop_words = set(ENGLISH_STOP_WORDS)

        if self.preserve_negations:
            stop_words.difference_update(
                NEGATION_WORDS
            )

        return stop_words

    @staticmethod
    def _expand_common_negations(text: str) -> str:
        """
        Expand common English negative contractions.

        This is particularly important for Quora because questions such as
        "Why doesn't this work?" should retain the semantic token "not".
        """
        text = re.sub(
            r"\bwon['’]t\b",
            "will not",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\bcan['’]t\b",
            "can not",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"n['’]t\b",
            " not",
            text,
            flags=re.IGNORECASE,
        )

        return text

    def normalize_text(self, text: Any) -> str:
        """
        Normalize Unicode and whitespace without deleting medical numbers
        or alphanumeric identifiers.
        """
        if text is None:
            return ""

        text = fix_text(str(text))
        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        text = text.replace("’", "'")
        text = text.replace("‘", "'")
        text = text.replace("–", "-")
        text = text.replace("—", "-")
        text = text.replace("−", "-")

        text = self._expand_common_negations(
            text
        )

        text = text.lower()

        # URLs and HTML markup are not useful retrieval terms.
        text = URL_PATTERN.sub(
            " ",
            text,
        )

        text = HTML_TAG_PATTERN.sub(
            " ",
            text,
        )

        # Underscores are treated as separators. Hyphens, slashes,
        # decimal points, colons, and plus signs are handled by the
        # tokenizer and may remain inside meaningful tokens.
        text = text.replace(
            "_",
            " ",
        )

        text = WHITESPACE_PATTERN.sub(
            " ",
            text,
        ).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize normalized text while preserving medical expressions.
        """
        if not text:
            return []

        tokens = TOKEN_PATTERN.findall(text)

        return [
            token
            for token in tokens
            if token
        ]

    @staticmethod
    def _token_content_length(token: str) -> int:
        """
        Count letters and digits in a token while ignoring separators.
        """
        return sum(
            1
            for character in token
            if character.isalnum()
        )

    def remove_stopwords(
        self,
        tokens: List[str],
    ) -> List[str]:
        """
        Remove stop words and tokens that are too short for the selected
        dataset profile.
        """
        filtered_tokens = []

        for token in tokens:
            if (
                self._token_content_length(token)
                < self.minimum_token_length
            ):
                continue

            if (
                self.remove_stopwords_enabled
                and token in self.stop_words
            ):
                continue

            filtered_tokens.append(token)

        return filtered_tokens

    def apply_stemming(
        self,
        tokens: List[str],
    ) -> List[str]:
        """
        Stem ordinary alphabetic words only.

        Tokens containing digits or medical separators are never stemmed.
        Examples that remain unchanged:

        V600E
        HER2-positive
        64-year-old
        mg/kg/day
        """
        if not self.use_stemming:
            return tokens

        stemmed_tokens = []

        for token in tokens:
            if token.isalpha():
                stemmed_tokens.append(
                    self.stemmer.stem(token)
                )
            else:
                stemmed_tokens.append(token)

        return stemmed_tokens

    def preprocess_tokens(
        self,
        text: Any,
    ) -> List[str]:
        """
        Normalize and tokenize text using the selected dataset profile.
        """
        normalized_text = self.normalize_text(
            text
        )

        tokens = self.tokenize(
            normalized_text
        )

        tokens = self.remove_stopwords(
            tokens
        )

        tokens = self.apply_stemming(
            tokens
        )

        return tokens

    def preprocess(
        self,
        text: Any,
    ) -> str:
        """
        Return preprocessed text as a space-separated string.
        """
        return " ".join(
            self.preprocess_tokens(text)
        )

    def get_configuration(self) -> Dict[str, Any]:
        """
        Return the active configuration for index manifests, debugging,
        tests, and reproducibility reports.
        """
        return {
            "dataset_key": self.dataset_key,
            "language": self.language,
            "remove_stopwords": (
                self.remove_stopwords_enabled
            ),
            "use_stemming": self.use_stemming,
            "minimum_token_length": (
                self.minimum_token_length
            ),
            "preserve_negations": (
                self.preserve_negations
            ),
            "token_pattern": TOKEN_PATTERN.pattern,
        }