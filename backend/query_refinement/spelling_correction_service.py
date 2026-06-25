import csv
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets.dataset_registry import (
    get_dataset_config,
)
from preprocessing.preprocessing_service import (
    TOKEN_PATTERN,
    TextPreprocessor,
)


SYMBOL_PATTERN = re.compile(r"[./:+\-]")
ALPHA_WORD_PATTERN = re.compile(r"[A-Za-z]+")


class SpellingCorrectionService:
    """
    Conservative offline spelling correction using raw surface words.
    """

    _SURFACE_VOCABULARY_CACHE: Dict[
        str,
        Counter,
    ] = {}

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        vocabulary: Optional[Iterable[str]] = None,
        max_corrections: int = 5,
        similarity_threshold: float = 0.82,
        min_frequency: int | None = None,
        max_documents: int = 250_000,
        max_vocabulary_size: int = 80_000,
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()

        self.max_corrections = int(
            max_corrections
        )

        self.similarity_threshold = float(
            similarity_threshold
        )

        self.min_frequency = (
            int(min_frequency)
            if min_frequency is not None
            else self._default_min_frequency(
                self.dataset_key
            )
        )

        self.max_documents = int(
            max_documents
        )

        self.max_vocabulary_size = int(
            max_vocabulary_size
        )

        if self.max_corrections <= 0:
            raise ValueError(
                "max_corrections must be greater than zero."
            )

        if self.min_frequency <= 0:
            raise ValueError(
                "min_frequency must be greater than zero."
            )

        self.preprocessor = TextPreprocessor(
            dataset_key=self.dataset_key
        )

        self.surface_counts = (
            self._load_surface_counts(
                vocabulary
            )
        )

        self.vocabulary = {
            term
            for term, frequency
            in self.surface_counts.items()
            if (
                frequency >= self.min_frequency
                and len(term) >= 3
            )
        }

        self.vocabulary_by_first_letter = (
            self._group_vocabulary(
                self.vocabulary
            )
        )

    @staticmethod
    def _default_min_frequency(
        dataset_key: str,
    ) -> int:
        if dataset_key == "sample_dataset":
            return 1

        return 20

    def _load_surface_counts(
        self,
        vocabulary: Optional[Iterable[str]],
    ) -> Counter:
        if vocabulary is not None:
            counts = Counter()

            for term in vocabulary:
                normalized_term = (
                    self._normalize_surface_word(
                        term
                    )
                )

                if normalized_term:
                    counts[normalized_term] = max(
                        counts[normalized_term],
                        self.min_frequency,
                    )

            return counts

        cached_counts = (
            self._SURFACE_VOCABULARY_CACHE.get(
                self.dataset_key
            )
        )

        if cached_counts is not None:
            return cached_counts

        counts = self._build_surface_counts()

        self._SURFACE_VOCABULARY_CACHE[
            self.dataset_key
        ] = counts

        return counts

    @staticmethod
    def _normalize_surface_word(
        value: object,
    ) -> str:
        word = str(value or "").strip().lower()

        if (
            len(word) < 3
            or not word.isalpha()
        ):
            return ""

        return word

    def _build_surface_counts(self) -> Counter:
        config = get_dataset_config(
            self.dataset_key
        )

        counts = Counter()

        self._add_document_terms(
            config=config,
            counts=counts,
        )

        self._add_query_terms(
            queries_path=Path(
                config["queries_path"]
            ),
            counts=counts,
        )

        if (
            self.max_vocabulary_size > 0
            and len(counts) > self.max_vocabulary_size
        ):
            counts = Counter(dict(
                counts.most_common(
                    self.max_vocabulary_size
                )
            ))

        return counts

    def _add_document_terms(
        self,
        config: Dict[str, str],
        counts: Counter,
    ):
        documents_path = Path(
            config["documents_path"]
        )

        dataset_format = config["format"]

        if not documents_path.is_file():
            return

        if dataset_format == "json":
            with documents_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                documents = json.load(file)

            for document in documents[
                :self.max_documents
            ]:
                self._add_text_terms(
                    counts=counts,
                    text=self._document_text(
                        document
                    ),
                )

            return

        if dataset_format == "jsonl":
            with documents_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                for document_index, line in enumerate(
                    file
                ):
                    if document_index >= self.max_documents:
                        break

                    line = line.strip()

                    if not line:
                        continue

                    document = json.loads(
                        line
                    )

                    self._add_text_terms(
                        counts=counts,
                        text=self._document_text(
                            document
                        ),
                    )

    @staticmethod
    def _document_text(
        document: Dict[str, object],
    ) -> str:
        return (
            f"{document.get('title', '')} "
            f"{document.get('text', '')} "
            f"{document.get('contents', '')} "
            f"{document.get('body', '')}"
        )

    def _add_query_terms(
        self,
        queries_path: Path,
        counts: Counter,
    ):
        if not queries_path.is_file():
            return

        with queries_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            reader = csv.reader(
                file,
                delimiter="\t",
            )

            for row in reader:
                if len(row) < 2:
                    continue

                self._add_text_terms(
                    counts=counts,
                    text=row[1],
                )

    def _add_text_terms(
        self,
        counts: Counter,
        text: object,
    ):
        for match in ALPHA_WORD_PATTERN.finditer(
            str(text or "")
        ):
            word = self._normalize_surface_word(
                match.group(0)
            )

            if word:
                counts[word] += 1

    @staticmethod
    def _group_vocabulary(
        vocabulary: set[str],
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}

        for term in sorted(vocabulary):
            grouped.setdefault(
                term[0],
                [],
            ).append(term)

        return grouped

    def correct(
        self,
        query: str,
    ) -> Dict[str, object]:
        original_query = str(query or "").strip()

        if not original_query:
            return {
                "corrected_query": original_query,
                "spelling_corrections": [],
                "spelling_correction_used": False,
            }

        spelling_corrections = []
        corrected_count = 0

        def replace_token(match):
            nonlocal corrected_count

            token = match.group(0)

            if corrected_count >= self.max_corrections:
                return token

            correction = self._correct_token(
                token
            )

            if correction is None:
                return token

            corrected_count += 1

            spelling_corrections.append({
                "original": token,
                "corrected": correction,
            })

            return correction

        corrected_query = TOKEN_PATTERN.sub(
            replace_token,
            original_query,
        )

        return {
            "corrected_query": corrected_query,
            "spelling_corrections": spelling_corrections,
            "spelling_correction_used": bool(
                spelling_corrections
            ),
        }

    def _correct_token(
        self,
        token: str,
    ) -> Optional[str]:
        if self._should_preserve_token(token):
            return None

        normalized_token = token.lower()

        if self._is_frequent_surface_word(
            normalized_token
        ):
            return None

        candidates = self._candidate_terms(
            normalized_token
        )

        best_candidate = None
        best_score = 0.0
        best_frequency = 0

        for candidate in candidates:
            similarity = SequenceMatcher(
                None,
                normalized_token,
                candidate,
            ).ratio()

            edit_distance = self._edit_distance(
                normalized_token,
                candidate,
                max_distance=2,
            )

            if edit_distance is None:
                continue

            if not self._passes_confidence(
                token=normalized_token,
                candidate=candidate,
                similarity=similarity,
                edit_distance=edit_distance,
            ):
                continue

            frequency = int(
                self.surface_counts[candidate]
            )

            if self._is_better_candidate(
                similarity=similarity,
                frequency=frequency,
                candidate=candidate,
                best_score=best_score,
                best_frequency=best_frequency,
                best_candidate=best_candidate,
            ):
                best_candidate = candidate
                best_score = similarity
                best_frequency = frequency

        return best_candidate

    def _is_frequent_surface_word(
        self,
        token: str,
    ) -> bool:
        return (
            self.surface_counts.get(token, 0)
            >= self.min_frequency
        )

    def _passes_confidence(
        self,
        token: str,
        candidate: str,
        similarity: float,
        edit_distance: int,
    ) -> bool:
        if similarity < self.similarity_threshold:
            return False

        token_length = len(token)

        if token_length <= 5:
            return edit_distance <= 1

        return edit_distance <= 2

    @staticmethod
    def _is_better_candidate(
        similarity: float,
        frequency: int,
        candidate: str,
        best_score: float,
        best_frequency: int,
        best_candidate: Optional[str],
    ) -> bool:
        if best_candidate is None:
            return True

        if similarity > best_score + 0.03:
            return True

        if similarity < best_score - 0.03:
            return False

        frequency_score = math.log1p(
            frequency
        )

        best_frequency_score = math.log1p(
            best_frequency
        )

        if frequency_score > best_frequency_score + 0.2:
            return True

        if frequency_score < best_frequency_score - 0.2:
            return False

        return candidate < best_candidate

    def _should_preserve_token(
        self,
        token: str,
    ) -> bool:
        if len(token) < 4:
            return True

        if not token.isalpha():
            return True

        if any(character.isdigit() for character in token):
            return True

        if SYMBOL_PATTERN.search(token):
            return True

        if token.isupper():
            return True

        if token != token.lower() and self._looks_like_abbreviation(token):
            return True

        return False

    @staticmethod
    def _looks_like_abbreviation(
        token: str,
    ) -> bool:
        uppercase_count = sum(
            1
            for character in token
            if character.isupper()
        )

        return (
            uppercase_count >= 2
            and len(token) <= 10
        )

    def _candidate_terms(
        self,
        token: str,
    ) -> List[str]:
        candidates = self.vocabulary_by_first_letter.get(
            token[0],
            [],
        )

        token_length = len(token)

        return [
            candidate
            for candidate in candidates
            if abs(len(candidate) - token_length) <= 2
        ]

    @staticmethod
    def _edit_distance(
        left: str,
        right: str,
        max_distance: int,
    ) -> Optional[int]:
        if abs(len(left) - len(right)) > max_distance:
            return None

        previous_row = list(
            range(len(right) + 1)
        )

        for left_index, left_character in enumerate(
            left,
            start=1,
        ):
            current_row = [
                left_index
            ]

            row_minimum = current_row[0]

            for right_index, right_character in enumerate(
                right,
                start=1,
            ):
                insertion_cost = (
                    current_row[right_index - 1]
                    + 1
                )

                deletion_cost = (
                    previous_row[right_index]
                    + 1
                )

                substitution_cost = (
                    previous_row[right_index - 1]
                    + (
                        0
                        if left_character == right_character
                        else 1
                    )
                )

                current_cost = min(
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )

                current_row.append(
                    current_cost
                )

                row_minimum = min(
                    row_minimum,
                    current_cost,
                )

            if row_minimum > max_distance:
                return None

            previous_row = current_row

        distance = previous_row[-1]

        if distance > max_distance:
            return None

        return distance
