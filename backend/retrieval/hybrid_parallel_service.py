from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from retrieval.biomedical_embedding_service import (
    BiomedicalEmbeddingService,
)
from retrieval.bm25_service import BM25RetrievalService
from retrieval.embedding_service import EmbeddingRetrievalService
from retrieval.tfidf_service import TfidfRetrievalService


class HybridParallelRetrievalService:
    """
    Hybrid Parallel Retrieval.

    The service runs TF-IDF, BM25, and Embedding retrieval independently,
    then fuses the ranked lists using Weighted Reciprocal Rank Fusion.

    Weighted RRF formula:
        final_score(doc) += model_weight / (rrf_k + rank_in_model)
    """

    MODEL_ORDER = (
        "tfidf",
        "bm25",
        "embedding",
        "biomedical",
    )

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        candidate_count: int = 1000,
        tfidf_weight: float = 1.0,
        bm25_weight: float = 1.0,
        embedding_weight: float = 1.0,
        biomedical_weight: float = 0.0,
        device: str = "auto",
    ):
        self.dataset_key = dataset_key
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.rrf_k = int(rrf_k)
        self.candidate_count = int(candidate_count)
        self.device = device

        self.model_weights = {
            "tfidf": float(tfidf_weight),
            "bm25": float(bm25_weight),
            "embedding": float(embedding_weight),
            "biomedical": float(biomedical_weight),
        }

        self._validate_weights()

        self.tfidf_service = None
        self.bm25_service = None
        self.embedding_service = None
        self.biomedical_service = None

        if self.model_weights["tfidf"] > 0:
            self.tfidf_service = TfidfRetrievalService(
                dataset_key=dataset_key,
            )

        if self.model_weights["bm25"] > 0:
            self.bm25_service = BM25RetrievalService(
                dataset_key=dataset_key,
                k1=self.bm25_k1,
                b=self.bm25_b,
            )

        if self.model_weights["embedding"] > 0:
            self.embedding_service = EmbeddingRetrievalService(
                dataset_key=dataset_key,
                use_saved_index=True,
                device=device,
            )

        if self.model_weights["biomedical"] > 0:
            if self.dataset_key != "clinical_trials":
                raise ValueError(
                    "biomedical_weight can only be used with the "
                    "clinical_trials dataset."
                )

            self.biomedical_service = BiomedicalEmbeddingService(
                dataset_key=dataset_key,
                use_saved_index=True,
                device=device,
            )

    def _validate_weights(self):
        for model_name, weight in self.model_weights.items():
            if weight < 0:
                raise ValueError(
                    f"{model_name}_weight must be greater than or equal to 0."
                )

        if sum(self.model_weights.values()) <= 0:
            raise ValueError(
                "At least one hybrid parallel model weight must be greater than 0."
            )

    def _active_services(self) -> List[Tuple[str, object]]:
        services = {
            "tfidf": self.tfidf_service,
            "bm25": self.bm25_service,
            "embedding": self.embedding_service,
            "biomedical": self.biomedical_service,
        }

        return [
            (
                model_name,
                services[model_name],
            )
            for model_name in self.MODEL_ORDER
            if (
                self.model_weights.get(model_name, 0.0) > 0
                and services[model_name] is not None
            )
        ]

    def _visible_fusion_weights(self) -> Dict[str, float]:
        fusion_weights = {
            "tfidf": self.model_weights["tfidf"],
            "bm25": self.model_weights["bm25"],
            "embedding": self.model_weights["embedding"],
        }

        if self.model_weights.get("biomedical", 0.0) > 0:
            fusion_weights["biomedical"] = self.model_weights[
                "biomedical"
            ]

        return fusion_weights

    @staticmethod
    def _normalize_queries(
        queries: List[str],
    ) -> List[str]:
        if not isinstance(queries, list):
            raise ValueError(
                "queries must be a list of strings."
            )

        normalized_queries = []

        for query_index, query in enumerate(queries):
            if not isinstance(query, str):
                raise ValueError(
                    "Every query must be a string. "
                    f"Invalid query position: {query_index}."
                )

            normalized_queries.append(query.strip())

        return normalized_queries

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        active_services = self._active_services()

        if len(active_services) == 1:
            model_name, service = active_services[0]
            ranked_lists = {
                model_name: service.search(
                    query=query,
                    top_k=self.candidate_count,
                )
            }

        else:
            with ThreadPoolExecutor(
                max_workers=len(active_services)
            ) as executor:
                futures = {
                    model_name: executor.submit(
                        service.search,
                        query=query,
                        top_k=self.candidate_count,
                    )
                    for model_name, service in active_services
                }

                ranked_lists = {
                    model_name: futures[model_name].result()
                    for model_name, service in active_services
                }

        fused_results = self._weighted_reciprocal_rank_fusion(
            ranked_lists
        )

        return fused_results[:top_k]

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        query_batch_size: int = 64,
        hydrate: bool = False,
    ) -> List[List[Dict]]:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if query_batch_size <= 0:
            raise ValueError(
                "query_batch_size must be greater than zero."
            )

        normalized_queries = self._normalize_queries(
            queries
        )

        if not normalized_queries:
            return []

        active_services = self._active_services()

        if len(active_services) == 1:
            model_name, service = active_services[0]
            service_results = {
                model_name: self._search_service_batch(
                    service=service,
                    queries=normalized_queries,
                    query_batch_size=query_batch_size,
                    hydrate=hydrate,
                )
            }

        else:
            with ThreadPoolExecutor(
                max_workers=len(active_services)
            ) as executor:
                futures = {
                    model_name: executor.submit(
                        self._search_service_batch,
                        service=service,
                        queries=normalized_queries,
                        query_batch_size=query_batch_size,
                        hydrate=hydrate,
                    )
                    for model_name, service in active_services
                }

                service_results = {
                    model_name: futures[model_name].result()
                    for model_name, service in active_services
                }

        for model_name, results in service_results.items():
            if len(results) != len(normalized_queries):
                raise ValueError(
                    "Hybrid parallel batch retrieval returned "
                    "an unexpected number of result lists for "
                    f"{model_name}. Expected: "
                    f"{len(normalized_queries)}, received: "
                    f"{len(results)}."
                )

        fused_batches: List[List[Dict]] = []

        for query_index in range(len(normalized_queries)):
            ranked_lists = {
                model_name: service_results[model_name][query_index]
                for model_name, service in active_services
            }

            fused_batches.append(
                self._weighted_reciprocal_rank_fusion(
                    ranked_lists
                )[:top_k]
            )

        return fused_batches

    def _search_service_batch(
        self,
        service: object,
        queries: List[str],
        query_batch_size: int,
        hydrate: bool,
    ) -> List[List[Dict]]:
        search_batch = getattr(
            service,
            "search_batch",
            None,
        )

        if callable(search_batch):
            return search_batch(
                queries=queries,
                top_k=self.candidate_count,
                query_batch_size=query_batch_size,
                hydrate=hydrate,
            )

        return [
            service.search(
                query=query,
                top_k=self.candidate_count,
            )
            if query
            else []
            for query in queries
        ]

    def _weighted_reciprocal_rank_fusion(
        self,
        ranked_lists: Dict[str, List[Dict]],
    ) -> List[Dict]:
        fused_scores: Dict[str, float] = {}
        documents: Dict[str, Dict] = {}
        model_details: Dict[str, Dict] = {}

        for model_name, results in ranked_lists.items():
            model_weight = self.model_weights.get(
                model_name,
                0.0,
            )

            if model_weight <= 0:
                continue

            for result in results:
                doc_id = str(result.get("doc_id", "") or "")

                if not doc_id:
                    continue

                rank = int(result.get("rank", 0) or 0)

                if rank <= 0:
                    continue

                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                    documents[doc_id] = {
                        "doc_id": doc_id,
                        "title": result.get("title"),
                        "snippet": result.get("snippet"),
                    }
                    model_details[doc_id] = {}

                contribution = model_weight / (self.rrf_k + rank)
                fused_scores[doc_id] += contribution

                model_details[doc_id][model_name] = {
                    "rank": rank,
                    "score": result.get("score"),
                    "weight": model_weight,
                    "weighted_rrf_contribution": round(
                        float(contribution),
                        8,
                    ),
                }

        ranked_doc_ids = sorted(
            fused_scores.keys(),
            key=lambda doc_id: fused_scores[doc_id],
            reverse=True,
        )

        fused_results: List[Dict] = []

        for rank, doc_id in enumerate(
            ranked_doc_ids,
            start=1,
        ):
            document = documents[doc_id]

            fused_results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": document.get("title"),
                "snippet": document.get("snippet"),
                "score": round(
                    float(fused_scores[doc_id]),
                    6,
                ),
                "fusion_method": "Weighted RRF",
                "rrf_k": self.rrf_k,
                "fusion_weights": self._visible_fusion_weights(),
                "model_details": model_details.get(
                    doc_id,
                    {},
                ),
            })

        return fused_results
