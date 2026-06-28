import gc
from time import perf_counter
from typing import Dict, Iterable, List, Set

from datasets.dataset_loader import DatasetLoader
from evaluation.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from query_refinement.pseudo_relevance_feedback import (
    PseudoRelevanceFeedbackService,
)
from retrieval.bm25_service import (
    BM25RetrievalService,
)
from retrieval.biomedical_embedding_service import (
    BiomedicalEmbeddingService,
)
from retrieval.distributed_bm25_service import (
    DistributedBM25RetrievalService,
)
from retrieval.embedding_service import (
    EmbeddingRetrievalService,
)
from retrieval.hybrid_parallel_service import (
    HybridParallelRetrievalService,
)
from retrieval.hybrid_serial_service import (
    HybridSerialRetrievalService,
)
from retrieval.ltr_feature_extractor import (
    DEFAULT_LTR_CANDIDATE_MODELS,
    normalize_ltr_candidate_models,
)
from retrieval.ltr_service import (
    LTRRetrievalService,
)
from retrieval.tfidf_service import (
    TfidfRetrievalService,
)


SUPPORTED_MODELS = {
    "tfidf",
    "bm25",
    "distributed_bm25",
    "embedding",
    "biomedical_embedding",
    "hybrid_serial",
    "hybrid_parallel",
    "ltr",
}

DEFAULT_EVALUATION_QUERY_BATCH_SIZE = 64
DEFAULT_LTR_EVALUATION_QUERY_BATCH_SIZE = 1


class EvaluationRunner:
    """
    Evaluate one retrieval model on every query represented in qrels.

    Important distinctions:

    - retrieval_depth:
        Number of documents retrieved per query.

    - precision_k:
        Cutoff used for Precision@K.

    - recall_k:
        Cutoff used for Recall@K.

    - ndcg_k:
        Cutoff used for nDCG@K.

    - query_batch_size:
        Number of queries sent to retrieval services that support
        batch search.

    Average Precision is calculated on the retrieved ranking up to
    retrieval_depth. Therefore, the reported metric is named
    MAP@retrieval_depth rather than unrestricted MAP.
    """

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        model_name: str = "tfidf",
        retrieval_depth: int = 1000,
        precision_k: int = 10,
        recall_k: int | None = None,
        ndcg_k: int = 10,
        candidate_count: int = 1000,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        num_shards: int = 4,
        shard_top_k: int | None = None,
        ltr_candidate_models: List[str] | None = None,
        include_biomedical: bool = False,
        ltr_model_path: str | None = None,
        biomedical_weight: float = 0.0,
        use_query_refinement: bool = False,
        feedback_docs: int = 3,
        expansion_terms: int = 5,
        query_batch_size: int | None = None,
        evaluation_query_ids: Iterable[str] | None = None,
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()

        self.model_name = str(
            model_name
        ).strip().lower()

        self.retrieval_depth = int(
            retrieval_depth
        )

        self.precision_k = int(
            precision_k
        )

        self.recall_k = int(
            recall_k
            if recall_k is not None
            else retrieval_depth
        )

        self.ndcg_k = int(
            ndcg_k
        )

        self.candidate_count = int(
            candidate_count
        )

        self.bm25_k1 = float(
            bm25_k1
        )

        self.bm25_b = float(
            bm25_b
        )

        self.rrf_k = int(
            rrf_k
        )

        self.num_shards = int(
            num_shards
        )

        self.shard_top_k = (
            int(shard_top_k)
            if shard_top_k is not None
            else (
                max(
                    self.retrieval_depth * 10,
                    100,
                )
                if self.model_name
                == "distributed_bm25"
                else None
            )
        )

        self.ltr_candidate_models = (
            normalize_ltr_candidate_models(
                ltr_candidate_models,
                include_biomedical=(
                    include_biomedical
                ),
                dataset_key=self.dataset_key,
            )
            if self.model_name == "ltr"
            else list(DEFAULT_LTR_CANDIDATE_MODELS)
        )

        self.include_biomedical = bool(
            include_biomedical
        )

        self.ltr_model_path = (
            str(ltr_model_path)
            if ltr_model_path
            else None
        )

        self.biomedical_weight = float(
            biomedical_weight
        )

        self.use_query_refinement = bool(
            use_query_refinement
        )

        self.feedback_docs = int(
            feedback_docs
        )

        self.expansion_terms = int(
            expansion_terms
        )

        self.query_batch_size = (
            int(query_batch_size)
            if query_batch_size is not None
            else (
                DEFAULT_LTR_EVALUATION_QUERY_BATCH_SIZE
                if self.model_name == "ltr"
                else DEFAULT_EVALUATION_QUERY_BATCH_SIZE
            )
        )

        self.requested_evaluation_query_ids = (
            [
                str(query_id).strip()
                for query_id in evaluation_query_ids
                if str(query_id).strip()
            ]
            if evaluation_query_ids is not None
            else None
        )

        self._validate_parameters()

        self.queries = (
            DatasetLoader.load_queries(
                self.dataset_key
            )
        )

        self.qrels_raw = (
            DatasetLoader.load_qrels(
                self.dataset_key
            )
        )

        self.query_by_id = (
            self._build_query_map(
                self.queries
            )
        )

        self.evaluation_query_ids = (
            self
            ._validate_and_get_evaluation_query_ids()
        )

        self.qrels_binary = (
            self._build_binary_qrels(
                self.qrels_raw
            )
        )

        self.retrieval_service = (
            self._build_retrieval_service()
        )

        self.query_refinement_service = (
            self
            ._build_query_refinement_service()
        )

    def _validate_parameters(self):
        if not self.dataset_key:
            raise ValueError(
                "dataset_key cannot be empty."
            )

        if self.model_name not in SUPPORTED_MODELS:
            available = ", ".join(
                sorted(SUPPORTED_MODELS)
            )

            raise ValueError(
                "Unsupported evaluation model "
                f"'{self.model_name}'. "
                f"Available models: {available}"
            )

        if self.retrieval_depth <= 0:
            raise ValueError(
                "retrieval_depth must be greater "
                "than zero."
            )

        if self.precision_k <= 0:
            raise ValueError(
                "precision_k must be greater "
                "than zero."
            )

        if self.recall_k <= 0:
            raise ValueError(
                "recall_k must be greater "
                "than zero."
            )

        if self.ndcg_k <= 0:
            raise ValueError(
                "ndcg_k must be greater "
                "than zero."
            )

        largest_metric_cutoff = max(
            self.precision_k,
            self.recall_k,
            self.ndcg_k,
        )

        if (
            largest_metric_cutoff
            > self.retrieval_depth
        ):
            raise ValueError(
                "retrieval_depth must be greater "
                "than or equal to every metric cutoff. "
                f"retrieval_depth="
                f"{self.retrieval_depth}, "
                f"largest cutoff="
                f"{largest_metric_cutoff}."
            )

        if self.candidate_count <= 0:
            raise ValueError(
                "candidate_count must be greater "
                "than zero."
            )

        if (
            self.model_name
            in {
                "hybrid_serial",
                "hybrid_parallel",
            }
            and self.candidate_count
            < self.retrieval_depth
        ):
            raise ValueError(
                "candidate_count must be greater "
                "than or equal to retrieval_depth "
                "for hybrid models."
            )

        if self.bm25_k1 <= 0:
            raise ValueError(
                "BM25 k1 must be greater than zero."
            )

        if not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError(
                "BM25 b must be between 0 and 1."
            )

        if self.rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        if self.num_shards <= 0:
            raise ValueError(
                "num_shards must be greater than zero."
            )

        if (
            self.shard_top_k is not None
            and self.shard_top_k <= 0
        ):
            raise ValueError(
                "shard_top_k must be greater than zero."
            )

        if (
            self.model_name == "ltr"
            and self.include_biomedical
            and self.dataset_key != "clinical_trials"
        ):
            raise ValueError(
                "include_biomedical can only be used with "
                "ltr on the clinical_trials dataset."
            )

        if self.biomedical_weight < 0:
            raise ValueError(
                "biomedical_weight must be greater than or equal to zero."
            )

        if (
            self.model_name == "hybrid_parallel"
            and self.biomedical_weight > 0
            and self.dataset_key != "clinical_trials"
        ):
            raise ValueError(
                "biomedical_weight can only be used with the "
                "clinical_trials dataset."
            )

        if self.query_batch_size <= 0:
            raise ValueError(
                "query_batch_size must be greater "
                "than zero."
            )

        if self.use_query_refinement:
            if self.feedback_docs <= 0:
                raise ValueError(
                    "feedback_docs must be greater "
                    "than zero."
                )

            if self.expansion_terms <= 0:
                raise ValueError(
                    "expansion_terms must be greater "
                    "than zero."
                )

    @staticmethod
    def _build_query_map(
        queries: List[Dict],
    ) -> Dict[str, str]:
        if not queries:
            raise ValueError(
                "No queries were loaded."
            )

        query_by_id: Dict[str, str] = {}

        for query_item in queries:
            query_id = str(
                query_item.get(
                    "query_id",
                    "",
                )
            ).strip()

            query_text = str(
                query_item.get(
                    "query",
                    "",
                )
            ).strip()

            if not query_id:
                raise ValueError(
                    "A loaded query is missing "
                    "its query_id."
                )

            if not query_text:
                raise ValueError(
                    f"Query '{query_id}' has "
                    "empty text."
                )

            if query_id in query_by_id:
                raise ValueError(
                    "Duplicate query ID found: "
                    f"'{query_id}'."
                )

            query_by_id[
                query_id
            ] = query_text

        return query_by_id

    def _validate_and_get_evaluation_query_ids(
        self,
    ) -> List[str]:
        """
        Return every qrels query ID in query-file order.

        Evaluation fails if a query represented in qrels has no matching
        query text. This prevents silently evaluating fewer queries than
        required.
        """
        if not self.qrels_raw:
            raise ValueError(
                f"No qrels found for dataset "
                f"'{self.dataset_key}'."
            )

        qrels_query_ids = set(
            self.qrels_raw.keys()
        )

        loaded_query_ids = set(
            self.query_by_id.keys()
        )

        missing_query_ids = (
            qrels_query_ids
            - loaded_query_ids
        )

        if missing_query_ids:
            examples = sorted(
                missing_query_ids
            )[:10]

            raise ValueError(
                "Some query IDs present in qrels "
                "have no matching query text. "
                f"Missing count: "
                f"{len(missing_query_ids)}. "
                f"Examples: {examples}"
            )

        if self.requested_evaluation_query_ids is not None:
            ordered_query_ids = list(
                dict.fromkeys(
                    self.requested_evaluation_query_ids
                )
            )

            if not ordered_query_ids:
                raise ValueError(
                    "evaluation_query_ids cannot be empty."
                )

            missing_from_qrels = [
                query_id
                for query_id in ordered_query_ids
                if query_id not in qrels_query_ids
            ]

            missing_from_queries = [
                query_id
                for query_id in ordered_query_ids
                if query_id not in loaded_query_ids
            ]

            if missing_from_qrels or missing_from_queries:
                raise ValueError(
                    "Some requested evaluation query IDs are unavailable. "
                    f"Missing from qrels: {missing_from_qrels[:10]}; "
                    f"missing from query file: {missing_from_queries[:10]}."
                )

            return ordered_query_ids

        ordered_query_ids = [
            query_id
            for query_id in self.query_by_id
            if query_id in qrels_query_ids
        ]

        if (
            len(ordered_query_ids)
            != len(qrels_query_ids)
        ):
            raise ValueError(
                "The evaluation query count does "
                "not match the number of unique "
                "query IDs in qrels."
            )

        return ordered_query_ids

    @staticmethod
    def _build_binary_qrels(
        qrels_raw: Dict[
            str,
            Dict[str, int],
        ]
    ) -> Dict[str, Set[str]]:
        binary_qrels: Dict[
            str,
            Set[str],
        ] = {}

        for (
            query_id,
            doc_relevances,
        ) in qrels_raw.items():
            binary_qrels[
                query_id
            ] = {
                doc_id
                for (
                    doc_id,
                    relevance,
                ) in doc_relevances.items()
                if relevance > 0
            }

        return binary_qrels

    def _build_retrieval_service(self):
        if self.model_name == "tfidf":
            return TfidfRetrievalService(
                dataset_key=self.dataset_key
            )

        if self.model_name == "bm25":
            return BM25RetrievalService(
                dataset_key=self.dataset_key,
                k1=self.bm25_k1,
                b=self.bm25_b,
            )

        if self.model_name == "distributed_bm25":
            return DistributedBM25RetrievalService(
                dataset_key=self.dataset_key,
                num_shards=self.num_shards,
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b,
                default_shard_top_k=(
                    self.shard_top_k
                ),
                default_rrf_k=self.rrf_k,
            )

        if self.model_name == "embedding":
            return EmbeddingRetrievalService(
                dataset_key=self.dataset_key
            )

        if self.model_name == "biomedical_embedding":
            return BiomedicalEmbeddingService(
                dataset_key=self.dataset_key
            )

        if self.model_name == "hybrid_serial":
            return HybridSerialRetrievalService(
                dataset_key=self.dataset_key,
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b,
                candidate_count=(
                    self.candidate_count
                ),
            )

        if self.model_name == "hybrid_parallel":
            return HybridParallelRetrievalService(
                dataset_key=self.dataset_key,
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b,
                rrf_k=self.rrf_k,
                candidate_count=(
                    self.candidate_count
                ),
                biomedical_weight=(
                    self.biomedical_weight
                ),
            )

        if self.model_name == "ltr":
            service = LTRRetrievalService(
                dataset_key=self.dataset_key,
                candidate_count=(
                    self.candidate_count
                ),
                candidate_models=(
                    self.ltr_candidate_models
                ),
                include_biomedical=(
                    self.include_biomedical
                ),
                model_path=self.ltr_model_path,
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b,
            )

            self.ltr_model_path = str(
                service.model_path
            )

            return service

        raise ValueError(
            "Unsupported model for evaluation: "
            f"{self.model_name}"
        )

    def _build_query_refinement_service(
        self,
    ):
        if not self.use_query_refinement:
            return None

        return PseudoRelevanceFeedbackService(
            dataset_key=self.dataset_key,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            feedback_docs=self.feedback_docs,
            expansion_terms=self.expansion_terms,
        )

    def _prepare_search_queries(
        self,
    ) -> List[str]:
        """
        Build search queries in evaluation_query_ids order.

        Query refinement is applied before retrieval. The retrieval timing
        therefore measures retrieval only and does not include refinement.
        """
        search_queries: List[str] = []

        for query_id in self.evaluation_query_ids:
            search_queries.append(
                self._prepare_search_query(
                    query_id
                )
            )

        return search_queries

    def _prepare_search_query(
        self,
        query_id: str,
    ) -> str:
        original_query = (
            self.query_by_id[
                query_id
            ]
        )

        if (
            self.use_query_refinement
            and self.query_refinement_service
        ):
            return (
                self.query_refinement_service
                .refine(
                    original_query
                )
            )

        return original_query

    @staticmethod
    def _deduplicate_document_ids(
        results: List[Dict],
    ) -> List[str]:
        """
        Remove duplicate document IDs while preserving ranking order.
        """
        retrieved_doc_ids: List[str] = []
        seen_doc_ids: Set[str] = set()

        for result in results:
            doc_id = result.get(
                "doc_id"
            )

            if doc_id is None:
                continue

            doc_id = str(
                doc_id
            ).strip()

            if not doc_id:
                continue

            if doc_id in seen_doc_ids:
                continue

            seen_doc_ids.add(
                doc_id
            )

            retrieved_doc_ids.append(
                doc_id
            )

        return retrieved_doc_ids

    def _calculate_query_metrics(
        self,
        query_id: str,
        results: List[Dict],
    ) -> Dict[str, float]:
        """
        Calculate all configured metrics for one ranked result list.
        """
        relevant_doc_ids = (
            self.qrels_binary.get(
                query_id,
                set(),
            )
        )

        relevance_scores = (
            self.qrels_raw.get(
                query_id,
                {},
            )
        )

        retrieved_doc_ids = (
            self._deduplicate_document_ids(
                results
            )
        )

        return {
            "average_precision": (
                average_precision(
                    retrieved_doc_ids=(
                        retrieved_doc_ids
                    ),
                    relevant_doc_ids=(
                        relevant_doc_ids
                    ),
                )
            ),
            "precision": (
                precision_at_k(
                    retrieved_doc_ids=(
                        retrieved_doc_ids
                    ),
                    relevant_doc_ids=(
                        relevant_doc_ids
                    ),
                    k=self.precision_k,
                )
            ),
            "recall": (
                recall_at_k(
                    retrieved_doc_ids=(
                        retrieved_doc_ids
                    ),
                    relevant_doc_ids=(
                        relevant_doc_ids
                    ),
                    k=self.recall_k,
                )
            ),
            "ndcg": (
                ndcg_at_k(
                    retrieved_doc_ids=(
                        retrieved_doc_ids
                    ),
                    relevance_scores=(
                        relevance_scores
                    ),
                    k=self.ndcg_k,
                )
            ),
        }

    @staticmethod
    def _mean(
        values: List[float],
    ) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _add_metric_values(
        metric_totals: Dict[str, float],
        metric_values: Dict[str, float],
    ):
        for metric_name, value in (
            metric_values.items()
        ):
            metric_totals[metric_name] = (
                metric_totals.get(
                    metric_name,
                    0.0,
                )
                + float(value)
            )

    @staticmethod
    def _mean_from_total(
        total: float,
        count: int,
    ) -> float:
        if count <= 0:
            return 0.0

        return total / count

    def evaluate(self) -> Dict:
        total_available_qrels_queries = len(
            self.qrels_raw
        )
        total_qrels_queries = len(
            self.evaluation_query_ids
        )

        total_loaded_queries = len(
            self.query_by_id
        )

        total_evaluation_queries = len(
            self.evaluation_query_ids
        )

        queries_with_positive_qrels = sum(
            1
            for query_id
            in self.evaluation_query_ids
            if self.qrels_binary.get(
                query_id,
                set(),
            )
        )

        queries_without_positive_qrels = (
            total_evaluation_queries
            - queries_with_positive_qrels
        )

        print(
            "Loaded queries: "
            f"{total_loaded_queries:,}"
        )

        print(
            "Evaluation query IDs from qrels: "
            f"{total_qrels_queries:,}"
        )
        if total_qrels_queries != total_available_qrels_queries:
            print(
                "Total available qrels query IDs: "
                f"{total_available_qrels_queries:,}"
            )

        print(
            "Queries scheduled for evaluation: "
            f"{total_evaluation_queries:,}"
        )

        print(
            "Queries with positive qrels: "
            f"{queries_with_positive_qrels:,}"
        )

        if queries_without_positive_qrels:
            print(
                "Warning: queries without positive "
                "qrels will receive zero binary "
                "relevance scores: "
                f"{queries_without_positive_qrels:,}"
            )

        metric_totals = {
            "average_precision": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
        }

        evaluated_queries = 0
        retrieval_elapsed_seconds = 0.0
        retrieval_mode = "single_query"
        use_batch_retrieval = False

        if self.model_name == "ltr":
            retrieval_mode = "ltr_streaming"
            use_batch_retrieval = True
            print(
                "Evaluation retrieval mode: "
                "LTR streaming"
            )

            print(
                "Query batch size: "
                f"{self.query_batch_size:,}"
            )

            for batch_start in range(
                0,
                total_evaluation_queries,
                self.query_batch_size,
            ):
                batch_end = min(
                    batch_start
                    + self.query_batch_size,
                    total_evaluation_queries,
                )

                batch_query_ids = (
                    self.evaluation_query_ids[
                        batch_start:batch_end
                    ]
                )

                for query_id in batch_query_ids:
                    search_query = (
                        self._prepare_search_query(
                            query_id
                        )
                    )

                    retrieval_start = (
                        perf_counter()
                    )

                    results = (
                        self.retrieval_service.search(
                            query=search_query,
                            top_k=(
                                self.retrieval_depth
                            ),
                            hydrate=False,
                        )
                    )

                    retrieval_elapsed_seconds += (
                        perf_counter()
                        - retrieval_start
                    )

                    metric_values = (
                        self._calculate_query_metrics(
                            query_id=query_id,
                            results=results,
                        )
                    )

                    self._add_metric_values(
                        metric_totals,
                        metric_values,
                    )

                    evaluated_queries += 1

                    del (
                        results,
                        metric_values,
                        search_query,
                    )

                print(
                    "Evaluated LTR queries: "
                    f"{batch_end:,}/"
                    f"{total_evaluation_queries:,}",
                    flush=True,
                )

                del batch_query_ids
                gc.collect()

        else:
            search_queries = (
                self._prepare_search_queries()
            )

            search_batch_method = getattr(
                self.retrieval_service,
                "search_batch",
                None,
            )

            use_batch_retrieval = callable(
                search_batch_method
            )

            if use_batch_retrieval:
                retrieval_mode = "batch"
                print(
                    "Evaluation retrieval mode: "
                    f"{self.model_name.upper()} batch"
                )

                print(
                    "Query batch size: "
                    f"{self.query_batch_size:,}"
                )

                for batch_start in range(
                    0,
                    total_evaluation_queries,
                    self.query_batch_size,
                ):
                    batch_end = min(
                        batch_start
                        + self.query_batch_size,
                        total_evaluation_queries,
                    )

                    batch_query_ids = (
                        self.evaluation_query_ids[
                            batch_start:batch_end
                        ]
                    )

                    batch_search_queries = (
                        search_queries[
                            batch_start:batch_end
                        ]
                    )

                    retrieval_start = (
                        perf_counter()
                    )

                    batch_results = (
                        search_batch_method(
                            queries=(
                                batch_search_queries
                            ),
                            top_k=(
                                self.retrieval_depth
                            ),
                            query_batch_size=(
                                self.query_batch_size
                            ),
                            hydrate=False,
                        )
                    )

                    retrieval_elapsed_seconds += (
                        perf_counter()
                        - retrieval_start
                    )

                    if (
                        len(batch_results)
                        != len(batch_query_ids)
                    ):
                        raise ValueError(
                            "Batch retrieval "
                            "returned an unexpected number "
                            "of result lists. "
                            f"Expected: "
                            f"{len(batch_query_ids)}, "
                            f"received: "
                            f"{len(batch_results)}."
                        )

                    for (
                        query_id,
                        results,
                    ) in zip(
                        batch_query_ids,
                        batch_results,
                    ):
                        metric_values = (
                            self
                            ._calculate_query_metrics(
                                query_id=query_id,
                                results=results,
                            )
                        )

                        self._add_metric_values(
                            metric_totals,
                            metric_values,
                        )

                        evaluated_queries += 1

                    print(
                        "Evaluated queries: "
                        f"{batch_end:,}/"
                        f"{total_evaluation_queries:,}",
                        flush=True,
                    )

                    del (
                        batch_results,
                        batch_query_ids,
                        batch_search_queries,
                    )

            else:
                retrieval_mode = "single_query"
                print(
                    "Evaluation retrieval mode: "
                    "single-query"
                )

                for query_number, (
                    query_id,
                    search_query,
                ) in enumerate(
                    zip(
                        self.evaluation_query_ids,
                        search_queries,
                    ),
                    start=1,
                ):
                    retrieval_start = (
                        perf_counter()
                    )

                    results = (
                        self.retrieval_service.search(
                            query=search_query,
                            top_k=(
                                self.retrieval_depth
                            ),
                        )
                    )

                    retrieval_elapsed_seconds += (
                        perf_counter()
                        - retrieval_start
                    )

                    metric_values = (
                        self._calculate_query_metrics(
                            query_id=query_id,
                            results=results,
                        )
                    )

                    self._add_metric_values(
                        metric_totals,
                        metric_values,
                    )

                    evaluated_queries += 1

                    if (
                        query_number % 10 == 0
                        or query_number
                        == total_evaluation_queries
                    ):
                        print(
                            "Evaluated queries: "
                            f"{query_number:,}/"
                            f"{total_evaluation_queries:,}",
                            flush=True,
                        )

                    del results, metric_values

            del search_queries


        if (
            evaluated_queries
            != total_qrels_queries
        ):
            raise ValueError(
                "Evaluation did not use every query "
                "present in qrels. "
                f"Expected: "
                f"{total_qrels_queries}, "
                f"evaluated: "
                f"{evaluated_queries}."
            )

        average_query_time_ms = (
            retrieval_elapsed_seconds
            * 1000.0
            / evaluated_queries
            if evaluated_queries
            else 0.0
        )

        queries_per_second = (
            evaluated_queries
            / retrieval_elapsed_seconds
            if retrieval_elapsed_seconds > 0.0
            else 0.0
        )

        map_key = (
            f"MAP@{self.retrieval_depth}"
        )

        precision_key = (
            f"Precision@{self.precision_k}"
        )

        recall_key = (
            f"Recall@{self.recall_k}"
        )

        ndcg_key = (
            f"nDCG@{self.ndcg_k}"
        )

        return {
            "dataset": self.dataset_key,
            "model": self.model_name,
            "retrieval_mode": (
                retrieval_mode
            ),
            "query_batch_size": (
                self.query_batch_size
                if use_batch_retrieval
                else None
            ),
            "retrieval_depth": (
                self.retrieval_depth
            ),
            "precision_k": (
                self.precision_k
            ),
            "recall_k": self.recall_k,
            "ndcg_k": self.ndcg_k,
            "loaded_queries": (
                total_loaded_queries
            ),
            "qrels_queries": (
                total_qrels_queries
            ),
            "evaluated_queries": (
                evaluated_queries
            ),
            "queries_with_positive_qrels": (
                queries_with_positive_qrels
            ),
            "queries_without_positive_qrels": (
                queries_without_positive_qrels
            ),
            "use_query_refinement": (
                self.use_query_refinement
            ),
            "feedback_docs": (
                self.feedback_docs
                if self.use_query_refinement
                else None
            ),
            "expansion_terms": (
                self.expansion_terms
                if self.use_query_refinement
                else None
            ),
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "candidate_count": (
                self.candidate_count
                if self.model_name
                in {
                    "hybrid_serial",
                    "hybrid_parallel",
                    "ltr",
                }
                else None
            ),
            "ltr_candidate_models": (
                " ".join(
                    self.ltr_candidate_models
                )
                if self.model_name == "ltr"
                else None
            ),
            "include_biomedical": (
                self.include_biomedical
                if self.model_name == "ltr"
                else None
            ),
            "ltr_model_path": (
                self.ltr_model_path
                if self.model_name == "ltr"
                else None
            ),
            "rrf_k": (
                self.rrf_k
                if self.model_name
                in {
                    "hybrid_parallel",
                    "distributed_bm25",
                }
                else None
            ),
            "distributed": (
                self.model_name
                == "distributed_bm25"
            ),
            "num_shards": (
                self.num_shards
                if self.model_name
                == "distributed_bm25"
                else None
            ),
            "shard_top_k": (
                self.shard_top_k
                if self.model_name
                == "distributed_bm25"
                else None
            ),
            "biomedical_weight": (
                self.biomedical_weight
                if self.model_name
                == "hybrid_parallel"
                else None
            ),
            map_key: round(
                self._mean_from_total(
                    metric_totals[
                        "average_precision"
                    ],
                    evaluated_queries,
                ),
                6,
            ),
            precision_key: round(
                self._mean_from_total(
                    metric_totals["precision"],
                    evaluated_queries,
                ),
                6,
            ),
            recall_key: round(
                self._mean_from_total(
                    metric_totals["recall"],
                    evaluated_queries,
                ),
                6,
            ),
            ndcg_key: round(
                self._mean_from_total(
                    metric_totals["ndcg"],
                    evaluated_queries,
                ),
                6,
            ),
            "EvaluationWallTimeSeconds": round(
                retrieval_elapsed_seconds,
                3,
            ),
            "AverageQueryTimeMs": round(
                average_query_time_ms,
                3,
            ),
            "QueriesPerSecond": round(
                queries_per_second,
                3,
            ),
        }
