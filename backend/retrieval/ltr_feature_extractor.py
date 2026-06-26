from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from preprocessing.preprocessing_service import (
    TextPreprocessor,
)


LTR_BASE_MODELS = (
    "bm25",
    "tfidf",
    "embedding",
    "biomedical",
)

DEFAULT_LTR_CANDIDATE_MODELS = [
    "bm25",
    "tfidf",
    "embedding",
]

LTR_RRF_K = 60

LTR_FEATURE_NAMES = [
    "bm25_score",
    "bm25_rank",
    "tfidf_score",
    "tfidf_rank",
    "embedding_score",
    "embedding_rank",
    "biomedical_score",
    "biomedical_rank",
    "in_bm25",
    "in_tfidf",
    "in_embedding",
    "in_biomedical",
    "query_token_count",
    "title_token_count",
    "snippet_token_count",
    "document_length",
    "query_term_overlap_count",
    "query_term_overlap_ratio",
    "title_query_overlap_count",
    "title_query_overlap_ratio",
    "exact_query_in_title",
    "exact_query_in_snippet",
    "reciprocal_rank_bm25",
    "reciprocal_rank_tfidf",
    "reciprocal_rank_embedding",
    "reciprocal_rank_biomedical",
    "rrf_sum",
]


def normalize_ltr_candidate_models(
    candidate_models: Iterable[str] | None,
    include_biomedical: bool = False,
    dataset_key: str | None = None,
) -> List[str]:
    if candidate_models is None:
        normalized = list(
            DEFAULT_LTR_CANDIDATE_MODELS
        )
    else:
        normalized = []

        for model_name in candidate_models:
            model_name = str(
                model_name
            ).strip().lower()

            if not model_name:
                continue

            if model_name not in LTR_BASE_MODELS:
                raise ValueError(
                    "Unsupported LTR candidate model "
                    f"'{model_name}'. Available models: "
                    + ", ".join(LTR_BASE_MODELS)
                )

            if model_name not in normalized:
                normalized.append(model_name)

    if include_biomedical:
        if dataset_key and dataset_key != "clinical_trials":
            raise ValueError(
                "include_biomedical is only supported for "
                "the clinical_trials dataset."
            )

        if "biomedical" not in normalized:
            normalized.append("biomedical")

    if (
        dataset_key
        and dataset_key != "clinical_trials"
        and "biomedical" in normalized
    ):
        raise ValueError(
            "biomedical LTR candidates are only supported "
            "for the clinical_trials dataset."
        )

    if not normalized:
        raise ValueError(
            "At least one LTR candidate model is required."
        )

    return normalized


def merge_ltr_candidate_results(
    ranked_lists: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    candidates: OrderedDict[
        str,
        Dict[str, Any],
    ] = OrderedDict()

    for model_name in LTR_BASE_MODELS:
        results = ranked_lists.get(
            model_name,
            [],
        )

        for position, result in enumerate(
            results,
            start=1,
        ):
            doc_id = str(
                result.get("doc_id", "")
                or ""
            ).strip()

            if not doc_id:
                continue

            rank = int(
                result.get(
                    "rank",
                    position,
                )
                or position
            )

            score = _safe_float(
                result.get("score"),
                default=0.0,
            )

            if doc_id not in candidates:
                candidates[doc_id] = {
                    "doc_id": doc_id,
                    "title": result.get(
                        "title",
                        "",
                    ),
                    "snippet": result.get(
                        "snippet",
                        "",
                    ),
                    "source_details": {},
                    "candidate_sources": [],
                }

            candidate = candidates[doc_id]

            if (
                not candidate.get("title")
                and result.get("title")
            ):
                candidate["title"] = result.get(
                    "title",
                    "",
                )

            if (
                not candidate.get("snippet")
                and result.get("snippet")
            ):
                candidate["snippet"] = result.get(
                    "snippet",
                    "",
                )

            candidate["source_details"][
                model_name
            ] = {
                "rank": rank,
                "score": score,
            }

            if (
                model_name
                not in candidate[
                    "candidate_sources"
                ]
            ):
                candidate[
                    "candidate_sources"
                ].append(model_name)

    return list(candidates.values())


def build_ltr_labels(
    candidates: List[Dict[str, Any]],
    qrels_for_query: Dict[str, int],
) -> List[float]:
    labels = []

    for candidate in candidates:
        doc_id = str(
            candidate.get("doc_id", "")
            or ""
        )
        labels.append(
            float(
                qrels_for_query.get(
                    doc_id,
                    0,
                )
            )
        )

    return labels


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_rank(
    value: Any,
) -> int:
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return 0

    if rank <= 0:
        return 0

    return rank


class LTRFeatureExtractor:
    """
    Extract stable numeric LTR features for query-document candidates.
    """

    feature_names = list(LTR_FEATURE_NAMES)

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()
        self.preprocessor = TextPreprocessor(
            dataset_key=self.dataset_key
        )

    def extract_feature_matrix(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        documents_by_id: Dict[str, Dict[str, Any]] | None = None,
        feature_names: List[str] | None = None,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        selected_feature_names = (
            feature_names
            if feature_names is not None
            else self.feature_names
        )

        rows = [
            self.extract_features(
                query=query,
                candidate=candidate,
                document=(
                    documents_by_id or {}
                ).get(
                    str(
                        candidate.get(
                            "doc_id",
                            "",
                        )
                    )
                ),
            )
            for candidate in candidates
        ]

        matrix = np.asarray(
            [
                [
                    float(row.get(name, 0.0))
                    for name in selected_feature_names
                ]
                for row in rows
            ],
            dtype=np.float32,
        )

        return (
            matrix,
            rows,
        )

    def extract_features(
        self,
        query: str,
        candidate: Dict[str, Any],
        document: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        source_details = dict(
            candidate.get(
                "source_details",
                {},
            )
            or {}
        )

        feature_values: Dict[str, float] = {}
        rrf_sum = 0.0

        for model_name in LTR_BASE_MODELS:
            detail = dict(
                source_details.get(
                    model_name,
                    {},
                )
                or {}
            )

            score = _safe_float(
                detail.get("score"),
                default=0.0,
            )
            rank = _safe_rank(
                detail.get("rank")
            )
            in_model = 1.0 if rank > 0 else 0.0
            reciprocal_rank = (
                1.0 / (LTR_RRF_K + rank)
                if rank > 0
                else 0.0
            )

            feature_values[
                f"{model_name}_score"
            ] = score if in_model else 0.0
            feature_values[
                f"{model_name}_rank"
            ] = float(rank)
            feature_values[
                f"in_{model_name}"
            ] = in_model
            feature_values[
                f"reciprocal_rank_{model_name}"
            ] = reciprocal_rank

            rrf_sum += reciprocal_rank

        title, snippet = self._resolve_text(
            candidate=candidate,
            document=document,
        )

        query_tokens = self._tokens(query)
        title_tokens = self._tokens(title)
        snippet_tokens = self._tokens(snippet)
        document_tokens = title_tokens + snippet_tokens

        query_token_set = set(query_tokens)
        title_token_set = set(title_tokens)
        document_token_set = set(
            document_tokens
        )

        query_overlap = (
            query_token_set
            & document_token_set
        )
        title_overlap = (
            query_token_set
            & title_token_set
        )

        feature_values.update({
            "query_token_count": float(
                len(query_tokens)
            ),
            "title_token_count": float(
                len(title_tokens)
            ),
            "snippet_token_count": float(
                len(snippet_tokens)
            ),
            "document_length": float(
                len(document_tokens)
            ),
            "query_term_overlap_count": float(
                len(query_overlap)
            ),
            "query_term_overlap_ratio": (
                float(len(query_overlap))
                / float(len(query_token_set))
                if query_token_set
                else 0.0
            ),
            "title_query_overlap_count": float(
                len(title_overlap)
            ),
            "title_query_overlap_ratio": (
                float(len(title_overlap))
                / float(len(query_token_set))
                if query_token_set
                else 0.0
            ),
            "exact_query_in_title": (
                1.0
                if self._contains_exact_query(
                    query,
                    title,
                )
                else 0.0
            ),
            "exact_query_in_snippet": (
                1.0
                if self._contains_exact_query(
                    query,
                    snippet,
                )
                else 0.0
            ),
            "rrf_sum": rrf_sum,
        })

        return {
            name: float(
                feature_values.get(
                    name,
                    0.0,
                )
            )
            for name in self.feature_names
        }

    def _tokens(
        self,
        text: str,
    ) -> List[str]:
        if not text:
            return []

        return list(
            self.preprocessor.preprocess_tokens(
                str(text)
            )
        )

    @staticmethod
    def _contains_exact_query(
        query: str,
        text: str,
    ) -> bool:
        normalized_query = " ".join(
            str(query)
            .strip()
            .lower()
            .split()
        )
        normalized_text = " ".join(
            str(text)
            .strip()
            .lower()
            .split()
        )

        return bool(
            normalized_query
            and normalized_query
            in normalized_text
        )

    @staticmethod
    def _resolve_text(
        candidate: Dict[str, Any],
        document: Dict[str, Any] | None,
    ) -> Tuple[str, str]:
        document = document or {}

        title = str(
            document.get(
                "title",
                candidate.get(
                    "title",
                    "",
                ),
            )
            or ""
        )

        snippet = str(
            document.get(
                "raw_text",
                document.get(
                    "text",
                    candidate.get(
                        "snippet",
                        "",
                    ),
                ),
            )
            or ""
        )

        return (
            title,
            snippet,
        )
