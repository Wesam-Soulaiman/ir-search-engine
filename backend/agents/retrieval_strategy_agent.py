from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RetrievalDecision:
    """
    A small decision object returned by the retrieval strategy agent.
    """

    selected_model: str
    reason: str
    features: Dict[str, object]


class RetrievalStrategyAgent:
    """
    Rule-based retrieval strategy agent.

    This is an additional feature that automatically selects the most
    appropriate retrieval model according to the dataset and query
    characteristics.

    It does not replace the core retrieval models. It only decides which
    already-implemented model should be used for a specific request.
    """

    SUPPORTED_DATASETS = {"quora", "clinical_trials"}

    ARABIC_CHAR_START = "\u0600"
    ARABIC_CHAR_END = "\u06ff"

    def decide(
        self,
        dataset_key: str,
        query: str,
        requested_features: Dict[str, object] | None = None,
    ) -> RetrievalDecision:
        dataset_key = str(dataset_key).strip()
        query = str(query or "").strip()
        requested_features = requested_features or {}

        if dataset_key not in self.SUPPORTED_DATASETS:
            return RetrievalDecision(
                selected_model="bm25",
                reason=(
                    "Unknown dataset. Falling back to BM25 because it is "
                    "a robust lexical retrieval baseline."
                ),
                features=self._extract_features(
                    dataset_key=dataset_key,
                    query=query,
                    requested_features=requested_features,
                ),
            )

        features = self._extract_features(
            dataset_key=dataset_key,
            query=query,
            requested_features=requested_features,
        )

        if features["is_arabic_query"]:
            return RetrievalDecision(
                selected_model="multilingual",
                reason=(
                    "Arabic query detected. The agent selected the "
                    "multilingual retrieval mode."
                ),
                features=features,
            )

        if dataset_key == "clinical_trials":
            return RetrievalDecision(
                selected_model="bm25",
                reason=(
                    "Clinical Trials contains specialized biomedical terms. "
                    "BM25 performed best on this dataset and is less likely "
                    "to drift than the general embedding model."
                ),
                features=features,
            )

        if dataset_key == "quora":
            if features["query_token_count"] <= 3:
                return RetrievalDecision(
                    selected_model="hybrid_serial",
                    reason=(
                        "Short Quora query detected. Hybrid Serial combines "
                        "BM25 candidate retrieval with embedding reranking "
                        "and achieved the best ranking quality on Quora."
                    ),
                    features=features,
                )

            return RetrievalDecision(
                selected_model="embedding",
                reason=(
                    "General natural-language Quora query detected. "
                    "Embedding retrieval captures semantic similarity well "
                    "and provides strong effectiveness with better speed "
                    "than Hybrid Serial."
                ),
                features=features,
            )

        return RetrievalDecision(
            selected_model="bm25",
            reason="Default fallback model.",
            features=features,
        )

    def _extract_features(
        self,
        dataset_key: str,
        query: str,
        requested_features: Dict[str, object],
    ) -> Dict[str, object]:
        tokens = self._tokenize(query)

        return {
            "dataset_key": dataset_key,
            "query": query,
            "query_token_count": len(tokens),
            "query_character_count": len(query),
            "is_arabic_query": self._contains_arabic(query),
            "requested_features": requested_features,
        }

    @staticmethod
    def _tokenize(query: str) -> List[str]:
        return [
            token
            for token in query.split()
            if token.strip()
        ]

    def _contains_arabic(self, text: str) -> bool:
        return any(
            self.ARABIC_CHAR_START <= character <= self.ARABIC_CHAR_END
            for character in text
        )