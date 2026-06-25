from collections import Counter
from typing import Dict, List, Set

from preprocessing.preprocessing_service import TextPreprocessor
from retrieval.bm25_service import BM25RetrievalService


class PseudoRelevanceFeedbackService:
    """
    Query refinement using pseudo-relevance feedback.

    Steps:
    1. Run BM25 to retrieve the top feedback documents.
    2. Extract feedback text from the retrieved documents.
    3. Process feedback text using the same dataset profile.
    4. Select frequent terms that are not already in the query.
    5. Append the selected expansion terms to the original query.

    Notes:
    - This is an unsupervised method and does not use qrels.
    - It can improve recall, but it may also cause query drift.
    """

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        feedback_docs: int = 3,
        expansion_terms: int = 5,
    ):
        self.dataset_key = dataset_key
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.feedback_docs = int(feedback_docs)
        self.expansion_terms = int(expansion_terms)

        self._validate_parameters()

        self.preprocessor = TextPreprocessor(
            dataset_key=dataset_key,
        )

        self.bm25_service = BM25RetrievalService(
            dataset_key=dataset_key,
            k1=self.bm25_k1,
            b=self.bm25_b,
        )

    def _validate_parameters(self):
        if self.feedback_docs <= 0:
            raise ValueError(
                "feedback_docs must be greater than zero."
            )

        if self.expansion_terms <= 0:
            raise ValueError(
                "expansion_terms must be greater than zero."
            )

        if self.bm25_k1 <= 0:
            raise ValueError(
                "BM25 k1 must be greater than zero."
            )

        if not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError(
                "BM25 b must be between 0 and 1."
            )

    def refine(self, query: str) -> str:
        """
        Expand the original query using pseudo-relevance feedback.

        The returned query keeps the original user query and appends
        selected expansion terms. If no useful terms are found, the
        original query is returned unchanged.
        """
        if not isinstance(query, str):
            raise ValueError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            return query

        feedback_results = self.bm25_service.search(
            query=query,
            top_k=self.feedback_docs,
        )

        if not feedback_results:
            return query

        original_query_terms = set(
            self.preprocessor.preprocess_tokens(query)
        )

        term_counter = Counter()

        for result in feedback_results:
            feedback_text = self._build_feedback_text(
                result=result,
            )

            if not feedback_text:
                continue

            tokens = self.preprocessor.preprocess_tokens(
                feedback_text,
            )

            unique_terms_in_document = (
                self._extract_candidate_terms(
                    tokens=tokens,
                    original_query_terms=original_query_terms,
                )
            )

            for token in unique_terms_in_document:
                term_counter[token] += 1

        selected_expansion_terms = (
            self._select_expansion_terms(
                term_counter=term_counter,
            )
        )

        if not selected_expansion_terms:
            return query

        return (
            f"{query} "
            f"{' '.join(selected_expansion_terms)}"
        )

    def _build_feedback_text(
        self,
        result: Dict,
    ) -> str:
        """
        Build feedback text from all text fields available in a
        retrieved result.

        The service prefers full document text when available, but
        falls back to snippets so the method remains compatible with
        older retrieval services.
        """
        text_parts = [
            result.get("title") or "",
            result.get("raw_text") or "",
            result.get("text") or "",
            result.get("content") or "",
            result.get("snippet") or "",
        ]

        return " ".join(
            part.strip()
            for part in text_parts
            if isinstance(part, str)
            and part.strip()
        )

    def _extract_candidate_terms(
        self,
        tokens: List[str],
        original_query_terms: Set[str],
    ) -> Set[str]:
        """
        Extract unique candidate expansion terms from one feedback
        document.

        Counting a term only once per feedback document prevents one
        repeated word inside a single document from dominating the
        expansion list.
        """
        candidate_terms: Set[str] = set()

        for token in tokens:
            if token in original_query_terms:
                continue

            if len(token) <= 2:
                continue

            candidate_terms.add(token)

        return candidate_terms

    def _select_expansion_terms(
        self,
        term_counter: Counter,
    ) -> List[str]:
        """
        Select the most frequent feedback terms across the feedback
        documents.
        """
        return [
            term
            for term, _ in term_counter.most_common(
                self.expansion_terms,
            )
        ]