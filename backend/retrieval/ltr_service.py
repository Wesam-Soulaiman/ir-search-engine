import json
import gc
import heapq
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from django.conf import settings

from datasets.dataset_loader import DatasetLoader
from document_store.repository import (
    DocumentStoreRepository,
)
from retrieval.biomedical_embedding_service import (
    BiomedicalEmbeddingService,
)
from retrieval.bm25_service import (
    BM25RetrievalService,
)
from retrieval.embedding_service import (
    EmbeddingRetrievalService,
)
from retrieval.ltr_feature_extractor import (
    DEFAULT_LTR_CANDIDATE_MODELS,
    LTRFeatureExtractor,
    merge_ltr_candidate_results,
    normalize_ltr_candidate_models,
)
from retrieval.result_enrichment import (
    document_store_is_required,
)
from retrieval.tfidf_service import (
    TfidfRetrievalService,
)


DEFAULT_LTR_CANDIDATE_COUNT = 1000
MAX_LTR_CANDIDATE_COUNT = 10_000


class LTRModelNotTrainedError(FileNotFoundError):
    """
    Raised when the requested LTR model artifact is missing.
    """


class LTRCandidateGenerator:
    """
    Build an LTR candidate pool from existing retrieval models.
    """

    def __init__(
        self,
        dataset_key: str,
        candidate_models: List[str] | None = None,
        include_biomedical: bool = False,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()
        self.candidate_models = (
            normalize_ltr_candidate_models(
                candidate_models,
                include_biomedical=(
                    include_biomedical
                ),
                dataset_key=self.dataset_key,
            )
        )
        self.include_biomedical = (
            "biomedical" in self.candidate_models
        )
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self._services: Dict[str, Any] = {}
        self._repository: (
            DocumentStoreRepository | None
        ) = None
        self._fallback_documents_by_id: (
            Dict[str, Dict[str, Any]] | None
        ) = None

    def _build_service(
        self,
        model_name: str,
    ):
        if model_name == "bm25":
            return BM25RetrievalService(
                dataset_key=self.dataset_key,
                k1=self.bm25_k1,
                b=self.bm25_b,
            )

        if model_name == "tfidf":
            return TfidfRetrievalService(
                dataset_key=self.dataset_key
            )

        if model_name == "embedding":
            return EmbeddingRetrievalService(
                dataset_key=self.dataset_key
            )

        if model_name == "biomedical":
            return BiomedicalEmbeddingService(
                dataset_key=self.dataset_key
            )

        raise ValueError(
            "Unsupported LTR candidate model "
            f"'{model_name}'."
        )

    def _get_service(
        self,
        model_name: str,
    ):
        if model_name not in self._services:
            self._services[
                model_name
            ] = self._build_service(
                model_name
            )

        return self._services[model_name]

    def generate(
        self,
        query: str,
        candidate_count: int = DEFAULT_LTR_CANDIDATE_COUNT,
    ) -> List[Dict[str, Any]]:
        ranked_lists = self.generate_ranked_lists(
            query=query,
            candidate_count=candidate_count,
        )

        return merge_ltr_candidate_results(
            ranked_lists
        )

    def generate_ranked_lists(
        self,
        query: str,
        candidate_count: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        candidate_count = validate_ltr_candidate_count(
            candidate_count
        )

        if not query.strip():
            return {
                model_name: []
                for model_name in self.candidate_models
            }

        with ThreadPoolExecutor(
            max_workers=len(self.candidate_models)
        ) as executor:
            futures = {
                model_name: executor.submit(
                    self._search_model,
                    model_name,
                    query,
                    candidate_count,
                )
                for model_name in self.candidate_models
            }

            return {
                model_name: futures[
                    model_name
                ].result()
                for model_name in self.candidate_models
            }

    def _search_model(
        self,
        model_name: str,
        query: str,
        candidate_count: int,
    ) -> List[Dict[str, Any]]:
        service = self._get_service(
            model_name
        )

        search_batch = getattr(
            service,
            "search_batch",
            None,
        )

        if callable(search_batch):
            try:
                return search_batch(
                    queries=[query],
                    top_k=candidate_count,
                    query_batch_size=1,
                    hydrate=False,
                )[0]
            except TypeError:
                pass

        try:
            return service.search(
                query=query,
                top_k=candidate_count,
                hydrate=False,
            )
        except TypeError:
            return service.search(
                query=query,
                top_k=candidate_count,
            )

    def fetch_documents_for_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        doc_ids = [
            str(candidate.get("doc_id", ""))
            for candidate in candidates
            if candidate.get("doc_id")
        ]

        if not doc_ids:
            return {}

        if document_store_is_required(
            self.dataset_key
        ):
            repository = self._get_repository()
            documents = repository.get_documents(
                dataset_key=self.dataset_key,
                doc_ids=doc_ids,
            )

            return {
                document["doc_id"]: document
                for document in documents
            }

        fallback_documents = (
            self._get_fallback_documents()
        )

        return {
            doc_id: fallback_documents[doc_id]
            for doc_id in doc_ids
            if doc_id in fallback_documents
        }

    def _get_repository(
        self,
    ) -> DocumentStoreRepository:
        if self._repository is None:
            self._repository = (
                DocumentStoreRepository(
                    settings.CORPUS_DATABASE_PATH
                )
            )
            self._repository.initialize()

        return self._repository

    def _get_fallback_documents(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        if self._fallback_documents_by_id is None:
            documents = DatasetLoader.load_documents(
                self.dataset_key
            )
            self._fallback_documents_by_id = {
                str(document["doc_id"]): document
                for document in documents
            }

        return self._fallback_documents_by_id


class LTRRetrievalService:
    """
    Learning-to-Rank retrieval as an optional reranking model.
    """

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        candidate_count: int = DEFAULT_LTR_CANDIDATE_COUNT,
        candidate_models: List[str] | None = None,
        include_biomedical: bool = False,
        model_path: str | Path | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()
        self.candidate_count = (
            validate_ltr_candidate_count(
                candidate_count
            )
        )
        self.candidate_models = (
            normalize_ltr_candidate_models(
                candidate_models,
                include_biomedical=(
                    include_biomedical
                ),
                dataset_key=self.dataset_key,
            )
        )
        self.include_biomedical = (
            "biomedical" in self.candidate_models
        )
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.model_path = self._resolve_model_path(
            model_path
        )
        self.metadata_path = (
            get_ltr_metadata_path(
                self.model_path
            )
        )

        self.model = self._load_model()
        self.training_metadata = (
            self._load_metadata()
        )
        self.feature_names = list(
            self.training_metadata.get(
                "feature_names",
                LTRFeatureExtractor.feature_names,
            )
        )
        self.extractor = LTRFeatureExtractor(
            dataset_key=self.dataset_key
        )
        self.candidate_generator = (
            LTRCandidateGenerator(
                dataset_key=self.dataset_key,
                candidate_models=(
                    self.candidate_models
                ),
                include_biomedical=False,
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b,
            )
        )
        self.last_search_metadata: Dict[
            str,
            Any,
        ] = {}

    def _resolve_model_path(
        self,
        model_path: str | Path | None,
    ) -> Path:
        if model_path is not None:
            path = Path(model_path).expanduser()
        else:
            artifacts_dir = Path(
                getattr(
                    settings,
                    "ARTIFACTS_DIR",
                    settings.BASE_DIR.parent
                    / "artifacts",
                )
            )
            path = (
                artifacts_dir
                / "models"
                / "ltr"
                / f"{self.dataset_key}_ltr.joblib"
            )

        if not path.is_absolute():
            path = (
                settings.BASE_DIR.parent
                / path
            )

        return path.resolve()

    def _load_model(self):
        if not self.model_path.is_file():
            raise LTRModelNotTrainedError(
                "LTR model is not trained for dataset "
                f"{self.dataset_key}. Run "
                "python backend/scripts/train_ltr_model.py "
                f"--dataset {self.dataset_key} "
                "--candidate-count "
                f"{self.candidate_count} "
                f"--output {self.model_path}"
            )

        return joblib.load(self.model_path)

    def _load_metadata(
        self,
    ) -> Dict[str, Any]:
        if not self.metadata_path.is_file():
            return {
                "dataset": self.dataset_key,
                "model_type": type(
                    self.model
                ).__name__,
                "feature_names": (
                    LTRFeatureExtractor.feature_names
                ),
            }

        metadata = json.loads(
            self.metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if metadata.get("dataset") not in {
            None,
            self.dataset_key,
        }:
            raise ValueError(
                "LTR model metadata dataset mismatch. "
                f"Expected {self.dataset_key}, found "
                f"{metadata.get('dataset')}."
            )

        return metadata

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_count: int | None = None,
        hydrate: bool = True,
    ) -> List[Dict[str, Any]]:
        if not isinstance(query, str):
            raise ValueError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            self.last_search_metadata = (
                self._build_metadata(
                    candidate_count=0,
                    result_count=0,
                )
            )
            return []

        try:
            parsed_top_k = int(top_k)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "top_k must be an integer."
            ) from error

        if parsed_top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        effective_candidate_count = (
            validate_ltr_candidate_count(
                candidate_count
                if candidate_count is not None
                else self.candidate_count
            )
        )

        candidates = (
            self.candidate_generator.generate(
                query=query,
                candidate_count=(
                    effective_candidate_count
                ),
            )
        )

        documents_by_id = (
            self.candidate_generator
            .fetch_documents_for_candidates(
                candidates
            )
        )

        feature_matrix, feature_rows = (
            self.extractor.extract_feature_matrix(
                query=query,
                candidates=candidates,
                documents_by_id=documents_by_id,
                feature_names=self.feature_names,
            )
        )

        if feature_matrix.size == 0:
            self.last_search_metadata = (
                self._build_metadata(
                    candidate_count=(
                        effective_candidate_count
                    ),
                    result_count=0,
                )
            )
            return []

        scores = np.asarray(
            self.model.predict(feature_matrix),
            dtype=np.float64,
        )

        def rank_key(position):
            return (
                -float(scores[position]),
                -float(
                    feature_rows[position].get(
                        "rrf_sum",
                        0.0,
                    )
                ),
                str(
                    candidates[position].get(
                        "doc_id",
                        "",
                    )
                ),
            )

        if parsed_top_k < len(candidates):
            ranked_positions = heapq.nsmallest(
                parsed_top_k,
                range(len(candidates)),
                key=rank_key,
            )
        else:
            ranked_positions = sorted(
                range(len(candidates)),
                key=rank_key,
            )

        results = []

        for rank, position in enumerate(
            ranked_positions,
            start=1,
        ):
            candidate = candidates[position]
            doc_id = str(candidate["doc_id"])
            score = round(
                float(scores[position]),
                8,
            )

            if not hydrate:
                results.append({
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": score,
                    "model": "ltr",
                    "ltr_score": score,
                })
                continue

            document = documents_by_id.get(
                doc_id,
                {},
            )
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
            source_details = dict(
                candidate.get(
                    "source_details",
                    {},
                )
                or {}
            )
            feature_values = (
                feature_rows[position]
            )

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": title,
                "snippet": snippet[:250],
                "score": score,
                "model": "ltr",
                "ltr_score": score,
                "model_details": {
                    "candidate_sources": list(
                        candidate.get(
                            "candidate_sources",
                            [],
                        )
                    ),
                    "base_model_scores": {
                        model_name: (
                            source_details.get(
                                model_name,
                                {},
                            )
                            .get("score", 0.0)
                        )
                        for model_name
                        in self.candidate_models
                    },
                    "base_model_ranks": {
                        model_name: (
                            source_details.get(
                                model_name,
                                {},
                            )
                            .get("rank", 0)
                        )
                        for model_name
                        in self.candidate_models
                    },
                    "features": (
                        self._important_features(
                            feature_values
                        )
                    ),
                },
            })

        self.last_search_metadata = (
            self._build_metadata(
                candidate_count=(
                    effective_candidate_count
                ),
                result_count=len(results),
            )
        )

        del (
            candidates,
            documents_by_id,
            feature_matrix,
            feature_rows,
            scores,
            ranked_positions,
        )

        return results

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        query_batch_size: int = 64,
        hydrate: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        if not isinstance(queries, list):
            raise ValueError(
                "queries must be a list of strings."
            )

        if query_batch_size <= 0:
            raise ValueError(
                "query_batch_size must be greater than zero."
            )

        batch_results: List[
            List[Dict[str, Any]]
        ] = []

        for batch_start in range(
            0,
            len(queries),
            int(query_batch_size),
        ):
            query_batch = queries[
                batch_start:batch_start
                + int(query_batch_size)
            ]

            for query in query_batch:
                batch_results.append(
                    self.search(
                        query=query,
                        top_k=top_k,
                        hydrate=hydrate,
                    )
                    if str(query).strip()
                    else []
                )

            del query_batch
            gc.collect()

        return batch_results

    def _important_features(
        self,
        feature_values: Dict[str, float],
    ) -> Dict[str, float]:
        names = [
            "bm25_score",
            "bm25_rank",
            "tfidf_score",
            "embedding_score",
            "query_term_overlap_ratio",
            "title_query_overlap_ratio",
            "rrf_sum",
        ]

        if self.include_biomedical:
            names.extend([
                "biomedical_score",
                "biomedical_rank",
            ])

        return {
            name: round(
                float(
                    feature_values.get(
                        name,
                        0.0,
                    )
                ),
                6,
            )
            for name in names
        }

    def _build_metadata(
        self,
        candidate_count: int,
        result_count: int,
    ) -> Dict[str, Any]:
        return {
            "ltr": True,
            "ltr_model_path": str(
                self.model_path
            ),
            "candidate_count": candidate_count,
            "candidate_models": list(
                self.candidate_models
            ),
            "include_biomedical": (
                self.include_biomedical
            ),
            "feature_count": len(
                self.feature_names
            ),
            "training_metadata": (
                self._metadata_summary()
            ),
            "ltr_result_count": result_count,
        }

    def _metadata_summary(
        self,
    ) -> Dict[str, Any]:
        keys = [
            "dataset",
            "model_type",
            "candidate_models",
            "include_biomedical",
            "candidate_count",
            "train_query_count",
            "validation_query_count",
            "training_rows",
            "validation_rows",
            "validation_metrics",
            "created_at",
            "random_seed",
        ]

        return {
            key: self.training_metadata.get(key)
            for key in keys
            if key in self.training_metadata
        }

    def get_last_search_metadata(
        self,
    ) -> Dict[str, Any]:
        return dict(
            self.last_search_metadata
        )


def validate_ltr_candidate_count(
    candidate_count: int,
) -> int:
    try:
        parsed = int(candidate_count)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "candidate_count must be an integer."
        ) from error

    if parsed <= 0:
        raise ValueError(
            "candidate_count must be greater than zero."
        )

    if parsed > MAX_LTR_CANDIDATE_COUNT:
        raise ValueError(
            "candidate_count must not exceed "
            f"{MAX_LTR_CANDIDATE_COUNT}."
        )

    return parsed


def get_ltr_metadata_path(
    model_path: Path,
) -> Path:
    return model_path.with_name(
        f"{model_path.stem}_metadata.json"
    )
