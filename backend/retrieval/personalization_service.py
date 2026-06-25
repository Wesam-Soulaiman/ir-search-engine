from collections import Counter
from typing import Dict, List

from preprocessing.preprocessing_service import (
    TextPreprocessor,
)


class PersonalizedQueryService:
    """
    Lightweight anonymous query personalization from local history.
    """

    def __init__(
        self,
        dataset_key: str,
        max_personalization_terms: int = 3,
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()

        self.max_personalization_terms = int(
            max_personalization_terms
        )

        if self.max_personalization_terms <= 0:
            raise ValueError(
                "max_personalization_terms must be greater than zero."
            )

        self.preprocessor = TextPreprocessor(
            dataset_key=self.dataset_key,
        )

    def personalize(
        self,
        query: str,
        previous_queries: List[str],
    ) -> Dict[str, object]:
        normalized_query = str(query or "").strip()

        if not normalized_query or not previous_queries:
            return {
                "personalized_query": normalized_query,
                "personalization_terms": [],
            }

        current_terms = set(
            self.preprocessor.preprocess_tokens(
                normalized_query
            )
        )

        term_counter = Counter()

        for previous_query in previous_queries:
            tokens = (
                self.preprocessor.preprocess_tokens(
                    previous_query
                )
            )

            seen_tokens = set()

            for token in tokens:
                if token in seen_tokens:
                    continue

                seen_tokens.add(token)

                if token in current_terms:
                    continue

                if len(token) <= 2:
                    continue

                term_counter[token] += 1

        personalization_terms = [
            term
            for term, count in term_counter.most_common(
                self.max_personalization_terms
            )
        ]

        if not personalization_terms:
            return {
                "personalized_query": normalized_query,
                "personalization_terms": [],
            }

        personalized_query = (
            f"{normalized_query} "
            f"{' '.join(personalization_terms)}"
        )

        return {
            "personalized_query": personalized_query,
            "personalization_terms": personalization_terms,
        }
