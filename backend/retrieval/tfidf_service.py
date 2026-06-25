import heapq
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from django.conf import settings
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from datasets.dataset_loader import DatasetLoader
from document_store.repository import (
    DocumentStoreRepository,
)
from preprocessing.preprocessing_service import (
    TextPreprocessor,
)


class TfidfIndexError(RuntimeError):
    """
    Raised when a saved TF-IDF index is invalid or incomplete.
    """


class TfidfRetrievalService:
    """
    Dataset-aware TF-IDF retrieval service.

    Supported index modes:

    1. Sharded TF-IDF
       Used by the complete Quora and Clinical Trials collections.

    2. Legacy single-matrix TF-IDF
       Retained for backward compatibility.

    3. In-memory development index
       Used by automated tests and sample_dataset when
       use_saved_index=False.

    The sharded mode supports both single-query search and efficient
    batch search. Batch search loads every sparse shard once for a group
    of queries, which is substantially faster during evaluation.
    """

    SHARDED_REQUIRED_DATASETS = {
        "quora",
        "clinical_trials",
    }

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        use_saved_index: bool = True,
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()

        self.use_saved_index = bool(
            use_saved_index
        )

        if not self.dataset_key:
            raise ValueError(
                "dataset_key cannot be empty."
            )

        self.preprocessor = TextPreprocessor(
            dataset_key=self.dataset_key
        )

        indexes_root = Path(
            getattr(
                settings,
                "INDEXES_DIR",
                settings.BASE_DIR.parent
                / "indexes",
            )
        ).expanduser().resolve()

        self.index_dir = (
            indexes_root
            / self.dataset_key
            / "tfidf"
        )

        # Sharded index artifacts.
        self.manifest_path = (
            self.index_dir
            / "manifest.json"
        )

        self.vocabulary_path = (
            self.index_dir
            / "vocabulary.joblib"
        )

        self.idf_path = (
            self.index_dir
            / "idf.npy"
        )

        self.preprocessing_config_path = (
            self.index_dir
            / "preprocessing_config.json"
        )

        # Legacy index artifacts.
        self.legacy_vectorizer_path = (
            self.index_dir
            / "vectorizer.joblib"
        )

        self.legacy_matrix_path = (
            self.index_dir
            / "matrix.joblib"
        )

        self.legacy_documents_path = (
            self.index_dir
            / "documents.joblib"
        )

        self.mode: Optional[str] = None

        # Sharded attributes.
        self.manifest: Dict[str, Any] = {}
        self.vocabulary: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.shards: List[Dict[str, Any]] = []
        self.document_count = 0
        self.sublinear_tf = True

        # Legacy and in-memory attributes.
        self.documents: List[Dict[str, Any]] = []
        self.processed_texts: List[str] = []
        self.vectorizer: Optional[
            TfidfVectorizer
        ] = None
        self.tfidf_matrix = None

        # Raw-document database, initialized only when needed.
        self._document_repository: Optional[
            DocumentStoreRepository
        ] = None

        self._initialize_index()

    def _initialize_index(self):
        if (
            self.use_saved_index
            and self.manifest_path.is_file()
        ):
            self.load_sharded_index()
            return

        if (
            self.use_saved_index
            and self.legacy_index_exists()
        ):
            self.load_legacy_index()
            return

        if (
            self.dataset_key
            in self.SHARDED_REQUIRED_DATASETS
        ):
            raise FileNotFoundError(
                "The complete dataset requires a sharded "
                f"TF-IDF index, but no manifest was found: "
                f"{self.manifest_path}"
            )

        self.load_documents()
        self.build_in_memory_index()

    def saved_index_exists(self) -> bool:
        return (
            self.manifest_path.is_file()
            or self.legacy_index_exists()
        )

    def legacy_index_exists(self) -> bool:
        return all(
            path.is_file()
            for path in [
                self.legacy_vectorizer_path,
                self.legacy_matrix_path,
                self.legacy_documents_path,
            ]
        )

    def load_sharded_index(self):
        """
        Load lightweight sharded-index metadata.

        Sparse matrix shards themselves are loaded one at a time during
        search, so the complete matrix is never held in memory.
        """
        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        self._validate_sharded_manifest(
            manifest
        )

        self.vocabulary = joblib.load(
            self.vocabulary_path
        )

        self.idf = np.load(
            self.idf_path,
            allow_pickle=False,
        ).astype(
            np.float32,
            copy=False,
        )

        if not isinstance(
            self.vocabulary,
            dict,
        ):
            raise TfidfIndexError(
                "The TF-IDF vocabulary artifact "
                "must contain a dictionary."
            )

        if (
            len(self.vocabulary)
            != len(self.idf)
        ):
            raise TfidfIndexError(
                "The TF-IDF vocabulary and IDF "
                "array have different sizes."
            )

        if (
            len(self.vocabulary)
            != int(
                manifest["vocabulary_size"]
            )
        ):
            raise TfidfIndexError(
                "The loaded vocabulary size does "
                "not match the manifest."
            )

        self.manifest = manifest
        self.shards = list(
            manifest["shards"]
        )

        self.document_count = int(
            manifest["document_count"]
        )

        self.sublinear_tf = bool(
            manifest.get(
                "sublinear_tf",
                True,
            )
        )

        self.mode = "sharded"

    def _validate_sharded_manifest(
        self,
        manifest: Dict[str, Any],
    ):
        if (
            manifest.get("index_type")
            != "sharded_tfidf"
        ):
            raise TfidfIndexError(
                "Unsupported TF-IDF index type: "
                f"{manifest.get('index_type')}"
            )

        if manifest.get("version") != 1:
            raise TfidfIndexError(
                "Unsupported sharded TF-IDF "
                f"version: {manifest.get('version')}"
            )

        if (
            manifest.get("dataset")
            != self.dataset_key
        ):
            raise TfidfIndexError(
                "TF-IDF manifest dataset mismatch. "
                f"Expected '{self.dataset_key}', found "
                f"'{manifest.get('dataset')}'."
            )

        if not self.vocabulary_path.is_file():
            raise TfidfIndexError(
                "TF-IDF vocabulary file is missing: "
                f"{self.vocabulary_path}"
            )

        if not self.idf_path.is_file():
            raise TfidfIndexError(
                "TF-IDF IDF file is missing: "
                f"{self.idf_path}"
            )

        if (
            not self.preprocessing_config_path
            .is_file()
        ):
            raise TfidfIndexError(
                "TF-IDF preprocessing configuration "
                f"is missing: "
                f"{self.preprocessing_config_path}"
            )

        with self.preprocessing_config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            saved_preprocessing = json.load(
                file
            )

        current_preprocessing = (
            self.preprocessor.get_configuration()
        )

        if (
            saved_preprocessing
            != current_preprocessing
        ):
            raise TfidfIndexError(
                "The sharded TF-IDF index was built "
                "with a different preprocessing "
                "configuration. Rebuild the index."
            )

        manifest_preprocessing = (
            manifest.get("preprocessing")
        )

        if (
            manifest_preprocessing
            != current_preprocessing
        ):
            raise TfidfIndexError(
                "The preprocessing configuration in "
                "the manifest does not match the "
                "current service configuration."
            )

        shards = manifest.get("shards")

        if not isinstance(shards, list):
            raise TfidfIndexError(
                "TF-IDF manifest does not contain "
                "a valid shard list."
            )

        if (
            len(shards)
            != int(
                manifest.get(
                    "shard_count",
                    -1,
                )
            )
        ):
            raise TfidfIndexError(
                "TF-IDF shard count does not "
                "match the manifest."
            )

        if not shards:
            raise TfidfIndexError(
                "TF-IDF manifest contains no shards."
            )

        vocabulary_size = int(
            manifest.get(
                "vocabulary_size",
                0,
            )
        )

        if vocabulary_size <= 0:
            raise TfidfIndexError(
                "TF-IDF vocabulary is empty."
            )

        expected_document_count = int(
            manifest.get(
                "document_count",
                0,
            )
        )

        if expected_document_count <= 0:
            raise TfidfIndexError(
                "TF-IDF manifest contains an invalid "
                "document count."
            )

        total_rows = 0

        for expected_shard_id, shard in enumerate(
            shards
        ):
            if (
                int(shard.get("shard_id", -1))
                != expected_shard_id
            ):
                raise TfidfIndexError(
                    "TF-IDF shard IDs are not "
                    "continuous and ordered."
                )

            if (
                int(shard.get("columns", -1))
                != vocabulary_size
            ):
                raise TfidfIndexError(
                    "A TF-IDF shard column count "
                    "does not match the vocabulary."
                )

            shard_rows = int(
                shard.get("rows", 0)
            )

            if shard_rows <= 0:
                raise TfidfIndexError(
                    "A TF-IDF shard contains an "
                    "invalid row count."
                )

            matrix_path = (
                self.index_dir
                / shard["matrix_file"]
            )

            doc_ids_path = (
                self.index_dir
                / shard["doc_ids_file"]
            )

            if not matrix_path.is_file():
                raise TfidfIndexError(
                    "Missing TF-IDF matrix shard: "
                    f"{matrix_path}"
                )

            if not doc_ids_path.is_file():
                raise TfidfIndexError(
                    "Missing TF-IDF document-ID "
                    f"shard: {doc_ids_path}"
                )

            total_rows += shard_rows

        if (
            total_rows
            != expected_document_count
        ):
            raise TfidfIndexError(
                "TF-IDF shard rows do not match "
                "the manifest document count."
            )

        self._validate_document_store_count(
            expected_document_count
        )

    def _validate_document_store_count(
        self,
        expected_document_count: int,
    ):
        repository = (
            self._get_document_repository()
        )

        dataset = repository.get_dataset(
            self.dataset_key
        )

        if dataset is None:
            raise TfidfIndexError(
                f"Dataset '{self.dataset_key}' "
                "is missing from the document store."
            )

        counts = repository.get_dataset_counts(
            self.dataset_key
        )

        actual_count = int(
            counts["documents"]
        )

        if actual_count != expected_document_count:
            raise TfidfIndexError(
                "TF-IDF index and document store "
                "contain different document counts. "
                f"Index: {expected_document_count}, "
                f"database: {actual_count}."
            )

    def _get_document_repository(
        self,
    ) -> DocumentStoreRepository:
        if self._document_repository is None:
            database_path = getattr(
                settings,
                "CORPUS_DATABASE_PATH",
                None,
            )

            if database_path is None:
                raise TfidfIndexError(
                    "CORPUS_DATABASE_PATH is not "
                    "configured."
                )

            self._document_repository = (
                DocumentStoreRepository(
                    database_path
                )
            )

            self._document_repository.initialize()

        return self._document_repository

    def load_legacy_index(self):
        """
        Load the previous single-matrix TF-IDF format.
        """
        if (
            self.preprocessing_config_path
            .is_file()
        ):
            with (
                self.preprocessing_config_path.open(
                    "r",
                    encoding="utf-8",
                )
            ) as file:
                saved_config = json.load(file)

            current_config = (
                self.preprocessor
                .get_configuration()
            )

            if saved_config != current_config:
                raise TfidfIndexError(
                    "The legacy TF-IDF index was "
                    "built with another preprocessing "
                    "configuration."
                )

        self.vectorizer = joblib.load(
            self.legacy_vectorizer_path
        )

        self.tfidf_matrix = joblib.load(
            self.legacy_matrix_path
        )

        self.documents = joblib.load(
            self.legacy_documents_path
        )

        self._validate_legacy_index()

        self.document_count = len(
            self.documents
        )

        self.mode = "legacy"

    def _validate_legacy_index(self):
        if self.vectorizer is None:
            raise TfidfIndexError(
                "Legacy TF-IDF vectorizer "
                "was not loaded."
            )

        if self.tfidf_matrix is None:
            raise TfidfIndexError(
                "Legacy TF-IDF matrix "
                "was not loaded."
            )

        if not isinstance(
            self.documents,
            list,
        ):
            raise TfidfIndexError(
                "Legacy TF-IDF document "
                "metadata must be a list."
            )

        if not self.documents:
            raise TfidfIndexError(
                "Legacy TF-IDF index is empty."
            )

        if (
            self.tfidf_matrix.shape[0]
            != len(self.documents)
        ):
            raise TfidfIndexError(
                "Legacy TF-IDF matrix rows "
                "do not match document count."
            )

    def load_documents(self):
        """
        Load a small development collection directly.

        This path is intentionally unavailable for complete Quora and
        Clinical Trials datasets.
        """
        self.documents = (
            DatasetLoader.load_documents(
                self.dataset_key
            )
        )

        if not self.documents:
            raise ValueError(
                f"No documents found for dataset "
                f"'{self.dataset_key}'."
            )

        self.processed_texts = [
            self.preprocessor.preprocess(
                self._get_document_search_text(
                    document
                )
            )
            for document in self.documents
        ]

    @staticmethod
    def _get_document_search_text(
        document: Dict[str, Any],
    ) -> str:
        return (
            f"{document.get('title', '')} "
            f"{document.get('text', '')}"
        ).strip()

    @staticmethod
    def create_vectorizer(
    ) -> TfidfVectorizer:
        """
        Create the legacy/in-memory vectorizer.

        str.split preserves pre-tokenized medical expressions such as
        v600e, her2-positive, and mg/kg/day.
        """
        return TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            dtype=np.float32,
            sublinear_tf=True,
            norm="l2",
        )

    def build_in_memory_index(self):
        if not self.processed_texts:
            raise ValueError(
                "Cannot build TF-IDF from "
                "an empty corpus."
            )

        self.vectorizer = (
            self.create_vectorizer()
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.processed_texts
            )
        )

        self._validate_legacy_index()

        self.document_count = len(
            self.documents
        )

        self.mode = "in_memory"

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not isinstance(query, str):
            raise ValueError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        result_count = min(
            int(top_k),
            int(self.document_count),
        )

        if result_count == 0:
            return []

        if self.mode == "sharded":
            return self._search_sharded(
                query=query,
                top_k=result_count,
            )

        if self.mode in {
            "legacy",
            "in_memory",
        }:
            return self._search_legacy(
                query=query,
                top_k=result_count,
            )

        raise RuntimeError(
            "TF-IDF service has no initialized "
            "index mode."
        )

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        query_batch_size: int = 64,
        hydrate: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """
        Search multiple queries efficiently.

        For a sharded index, each sparse shard is loaded once for an
        entire query batch. This is substantially faster than invoking
        search() separately for every evaluation query.

        Args:
            queries:
                Query strings in the required output order.

            top_k:
                Number of results returned for every query.

            query_batch_size:
                Number of queries scored together.

            hydrate:
                If True, retrieve titles and raw snippets from SQLite.
                Evaluation should use False because it only needs doc IDs
                and scores.

        Returns:
            One ranked result list for every input query.
        """
        if not isinstance(queries, list):
            raise ValueError(
                "queries must be a list of strings."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if query_batch_size <= 0:
            raise ValueError(
                "query_batch_size must be greater than zero."
            )

        normalized_queries = []

        for query_index, query in enumerate(
            queries
        ):
            if not isinstance(query, str):
                raise ValueError(
                    "Every query must be a string. "
                    f"Invalid query position: {query_index}."
                )

            normalized_queries.append(
                query.strip()
            )

        if not normalized_queries:
            return []

        result_count = min(
            int(top_k),
            int(self.document_count),
        )

        if result_count <= 0:
            return [
                []
                for _ in normalized_queries
            ]

        if self.mode != "sharded":
            return [
                self.search(
                    query=query,
                    top_k=result_count,
                )
                if query
                else []
                for query in normalized_queries
            ]

        all_results: List[
            List[Dict[str, Any]]
        ] = []

        for batch_start in range(
            0,
            len(normalized_queries),
            query_batch_size,
        ):
            batch_queries = normalized_queries[
                batch_start:
                batch_start + query_batch_size
            ]

            batch_results = (
                self._search_sharded_batch(
                    queries=batch_queries,
                    top_k=result_count,
                    hydrate=hydrate,
                )
            )

            all_results.extend(
                batch_results
            )

        return all_results

    def _build_sharded_query_vector(
        self,
        query: str,
    ):
        """
        Build one sparse TF-IDF query vector.

        The implementation uses the same batch-vector construction path
        as search_batch(), ensuring identical preprocessing and weights.
        """
        (
            query_matrix,
            active_queries,
        ) = self._build_sharded_query_matrix(
            [query]
        )

        if not active_queries[0]:
            return None

        return query_matrix

    def _build_sharded_query_matrix(
        self,
        queries: List[str],
    ):
        """
        Build an L2-normalized sparse TF-IDF query matrix.

        The returned matrix has one row for every query, including blank
        or out-of-vocabulary queries. active_queries marks rows that
        contain at least one indexed term.
        """
        if self.idf is None:
            raise TfidfIndexError(
                "TF-IDF IDF values are not loaded."
            )

        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0]
        active_queries: List[bool] = []

        for query in queries:
            if not query:
                active_queries.append(False)
                indptr.append(len(data))
                continue

            tokens = (
                self.preprocessor
                .preprocess_tokens(query)
            )

            term_counts = Counter(tokens)

            weighted_terms: List[
                Tuple[int, float]
            ] = []

            for term, count in term_counts.items():
                term_index = self.vocabulary.get(
                    term
                )

                if term_index is None:
                    continue

                if self.sublinear_tf:
                    tf_value = (
                        1.0
                        + math.log(float(count))
                    )
                else:
                    tf_value = float(count)

                value = (
                    tf_value
                    * float(
                        self.idf[term_index]
                    )
                )

                weighted_terms.append(
                    (
                        int(term_index),
                        float(value),
                    )
                )

            weighted_terms.sort(
                key=lambda item: item[0]
            )

            squared_norm = sum(
                value * value
                for (
                    term_index,
                    value,
                ) in weighted_terms
            )

            norm = math.sqrt(
                squared_norm
            )

            is_active = (
                bool(weighted_terms)
                and norm > 0.0
            )

            active_queries.append(
                is_active
            )

            if is_active:
                for term_index, value in (
                    weighted_terms
                ):
                    indices.append(
                        term_index
                    )

                    data.append(
                        value / norm
                    )

            indptr.append(
                len(data)
            )

        query_matrix = sparse.csr_matrix(
            (
                np.asarray(
                    data,
                    dtype=np.float32,
                ),
                np.asarray(
                    indices,
                    dtype=np.int32,
                ),
                np.asarray(
                    indptr,
                    dtype=np.int64,
                ),
            ),
            shape=(
                len(queries),
                len(self.vocabulary),
            ),
            dtype=np.float32,
        )

        query_matrix.sort_indices()

        return (
            query_matrix,
            active_queries,
        )

    def _search_sharded(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Search one query through the same execution path used by batch
        search. This guarantees identical result ordering and scores.
        """
        return self._search_sharded_batch(
            queries=[query],
            top_k=top_k,
            hydrate=True,
        )[0]

    def _search_sharded_batch(
        self,
        queries: List[str],
        top_k: int,
        hydrate: bool,
    ) -> List[List[Dict[str, Any]]]:
        """
        Score one query batch against all sparse shards.

        Matrix operation per shard:

            shard_documents × query_batch.T

        Each query keeps only its global top-k results in a heap.
        """
        (
            query_matrix,
            active_queries,
        ) = self._build_sharded_query_matrix(
            queries
        )

        query_count = len(queries)

        if query_count == 0:
            return []

        if not any(active_queries):
            return [
                []
                for _ in queries
            ]

        query_heaps: List[
            List[Tuple[float, int, str]]
        ] = [
            []
            for _ in range(query_count)
        ]

        # Used only when an active query has fewer than top_k positive
        # matches. This provides deterministic zero-score results in
        # corpus order.
        fallback_doc_ids: List[str] = []

        global_position_offset = 0

        for shard in self.shards:
            matrix_path = (
                self.index_dir
                / shard["matrix_file"]
            )

            doc_ids_path = (
                self.index_dir
                / shard["doc_ids_file"]
            )

            shard_matrix = sparse.load_npz(
                matrix_path
            ).tocsr()

            doc_ids = joblib.load(
                doc_ids_path
            )

            expected_rows = int(
                shard["rows"]
            )

            if (
                shard_matrix.shape[0]
                != expected_rows
            ):
                raise TfidfIndexError(
                    "Loaded TF-IDF shard row "
                    f"count mismatch: {matrix_path}"
                )

            if (
                shard_matrix.shape[1]
                != len(self.vocabulary)
            ):
                raise TfidfIndexError(
                    "Loaded TF-IDF shard column "
                    f"count mismatch: {matrix_path}"
                )

            if len(doc_ids) != expected_rows:
                raise TfidfIndexError(
                    "TF-IDF document-ID shard "
                    f"count mismatch: {doc_ids_path}"
                )

            if len(fallback_doc_ids) < top_k:
                needed = (
                    top_k
                    - len(fallback_doc_ids)
                )

                fallback_doc_ids.extend(
                    str(doc_id)
                    for doc_id in doc_ids[
                        :needed
                    ]
                )

            # Before transpose:
            # documents_in_shard × queries_in_batch
            #
            # After transpose:
            # queries_in_batch × documents_in_shard
            score_matrix = (
                shard_matrix
                @ query_matrix.transpose()
            ).transpose().tocsr()

            for query_index in range(
                query_count
            ):
                if not active_queries[
                    query_index
                ]:
                    continue

                row_start = int(
                    score_matrix.indptr[
                        query_index
                    ]
                )

                row_end = int(
                    score_matrix.indptr[
                        query_index + 1
                    ]
                )

                local_indices = (
                    score_matrix.indices[
                        row_start:row_end
                    ]
                )

                local_scores = (
                    score_matrix.data[
                        row_start:row_end
                    ]
                )

                if local_scores.size == 0:
                    continue

                positive_mask = (
                    local_scores > 0.0
                )

                local_indices = (
                    local_indices[
                        positive_mask
                    ]
                )

                local_scores = (
                    local_scores[
                        positive_mask
                    ]
                )

                if local_scores.size == 0:
                    continue

                selected_positions = (
                    self._select_shard_candidates(
                        local_indices=(
                            local_indices
                        ),
                        local_scores=(
                            local_scores
                        ),
                        top_k=top_k,
                    )
                )

                query_heap = query_heaps[
                    query_index
                ]

                for selected_position in (
                    selected_positions
                ):
                    selected_position = int(
                        selected_position
                    )

                    local_document_index = int(
                        local_indices[
                            selected_position
                        ]
                    )

                    score = float(
                        local_scores[
                            selected_position
                        ]
                    )

                    global_position = (
                        global_position_offset
                        + local_document_index
                    )

                    doc_id = str(
                        doc_ids[
                            local_document_index
                        ]
                    )

                    heap_item = (
                        score,
                        -global_position,
                        doc_id,
                    )

                    if len(query_heap) < top_k:
                        heapq.heappush(
                            query_heap,
                            heap_item,
                        )

                    elif heap_item > query_heap[0]:
                        heapq.heapreplace(
                            query_heap,
                            heap_item,
                        )

            global_position_offset += (
                expected_rows
            )

            del shard_matrix
            del doc_ids
            del score_matrix

        batch_results: List[
            List[Dict[str, Any]]
        ] = []

        for query_index, query_heap in enumerate(
            query_heaps
        ):
            if not active_queries[
                query_index
            ]:
                batch_results.append([])
                continue

            ranked_items = sorted(
                query_heap,
                reverse=True,
            )

            ranked_documents: List[
                Tuple[str, float]
            ] = [
                (
                    doc_id,
                    float(score),
                )
                for (
                    score,
                    negative_position,
                    doc_id,
                ) in ranked_items
            ]

            selected_doc_ids = {
                doc_id
                for doc_id, score
                in ranked_documents
            }

            if len(ranked_documents) < top_k:
                for doc_id in fallback_doc_ids:
                    if doc_id in selected_doc_ids:
                        continue

                    ranked_documents.append(
                        (
                            doc_id,
                            0.0,
                        )
                    )

                    selected_doc_ids.add(
                        doc_id
                    )

                    if (
                        len(ranked_documents)
                        >= top_k
                    ):
                        break

            ranked_documents = (
                ranked_documents[:top_k]
            )

            if hydrate:
                results = (
                    self._hydrate_sharded_results(
                        ranked_documents
                    )
                )

            else:
                results = [
                    {
                        "rank": rank,
                        "doc_id": doc_id,
                        "score": round(
                            float(score),
                            6,
                        ),
                    }
                    for rank, (
                        doc_id,
                        score,
                    ) in enumerate(
                        ranked_documents,
                        start=1,
                    )
                ]

            batch_results.append(
                results
            )

        return batch_results

    @staticmethod
    def _select_shard_candidates(
        local_indices: np.ndarray,
        local_scores: np.ndarray,
        top_k: int,
    ) -> np.ndarray:
        """
        Select deterministic top-k positions from one shard.

        Results are ordered by descending score, then ascending local
        document position. The returned positions index local_scores and
        local_indices, not the original shard matrix directly.
        """
        candidate_count = int(
            local_scores.size
        )

        if candidate_count <= top_k:
            candidate_positions = np.arange(
                candidate_count,
                dtype=np.int64,
            )

        else:
            threshold = np.partition(
                local_scores,
                -top_k,
            )[-top_k]

            candidate_positions = np.flatnonzero(
                local_scores >= threshold
            )

        candidate_order = np.lexsort(
            (
                local_indices[
                    candidate_positions
                ],
                -local_scores[
                    candidate_positions
                ],
            )
        )

        return candidate_positions[
            candidate_order[:top_k]
        ]

    def _hydrate_sharded_results(
        self,
        ranked_documents: List[
            Tuple[str, float]
        ],
    ) -> List[Dict[str, Any]]:
        ordered_doc_ids = [
            doc_id
            for doc_id, score
            in ranked_documents
        ]

        repository = (
            self._get_document_repository()
        )

        documents = repository.get_documents(
            dataset_key=self.dataset_key,
            doc_ids=ordered_doc_ids,
        )

        documents_by_id = {
            document["doc_id"]: document
            for document in documents
        }

        missing_doc_ids = [
            doc_id
            for doc_id in ordered_doc_ids
            if doc_id not in documents_by_id
        ]

        if missing_doc_ids:
            raise TfidfIndexError(
                "TF-IDF returned document IDs "
                "missing from the document store. "
                f"Examples: {missing_doc_ids[:10]}"
            )

        results = []

        for rank, (
            doc_id,
            score,
        ) in enumerate(
            ranked_documents,
            start=1,
        ):
            document = documents_by_id[
                doc_id
            ]

            raw_text = str(
                document.get(
                    "raw_text",
                    "",
                )
                or ""
            )

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": str(
                    document.get(
                        "title",
                        "",
                    )
                    or ""
                ),
                "snippet": raw_text[:250],
                "score": round(
                    float(score),
                    6,
                ),
            })

        return results

    def _search_legacy(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if (
            self.vectorizer is None
            or self.tfidf_matrix is None
        ):
            raise RuntimeError(
                "Legacy TF-IDF index is "
                "not initialized."
            )

        processed_query = (
            self.preprocessor.preprocess(
                query
            )
        )

        if not processed_query:
            return []

        query_vector = (
            self.vectorizer.transform(
                [processed_query]
            )
        )

        if query_vector.nnz == 0:
            return []

        scores = linear_kernel(
            query_vector,
            self.tfidf_matrix,
        ).ravel()

        if top_k < len(scores):
            candidate_indices = np.argpartition(
                scores,
                -top_k,
            )[-top_k:]

            candidate_scores = scores[
                candidate_indices
            ]

            candidate_order = np.lexsort(
                (
                    candidate_indices,
                    -candidate_scores,
                )
            )

            ranked_indices = candidate_indices[
                candidate_order
            ]

        else:
            all_indices = np.arange(
                len(scores),
                dtype=np.int64,
            )

            ranked_indices = all_indices[
                np.lexsort(
                    (
                        all_indices,
                        -scores,
                    )
                )
            ]

        results = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            document_index = int(index)
            document = self.documents[
                document_index
            ]

            results.append({
                "rank": rank,
                "doc_id": str(
                    document.get(
                        "doc_id",
                        "",
                    )
                ),
                "title": document.get(
                    "title",
                    "",
                ),
                "snippet": document.get(
                    "text",
                    "",
                )[:250],
                "score": round(
                    float(
                        scores[
                            document_index
                        ]
                    ),
                    6,
                ),
            })

        return results

    def get_index_information(
        self,
    ) -> Dict[str, Any]:
        information = {
            "dataset": self.dataset_key,
            "mode": self.mode,
            "document_count": (
                self.document_count
            ),
            "index_directory": str(
                self.index_dir
            ),
            "preprocessing": (
                self.preprocessor
                .get_configuration()
            ),
            "batch_search_supported": True,
        }

        if self.mode == "sharded":
            information.update({
                "index_type": (
                    self.manifest[
                        "index_type"
                    ]
                ),
                "vocabulary_size": len(
                    self.vocabulary
                ),
                "shard_count": len(
                    self.shards
                ),
                "min_df": self.manifest[
                    "min_df"
                ],
                "max_df": self.manifest[
                    "max_df"
                ],
                "max_features": (
                    self.manifest[
                        "max_features"
                    ]
                ),
                "sublinear_tf": (
                    self.sublinear_tf
                ),
            })

        elif self.tfidf_matrix is not None:
            information.update({
                "matrix_shape": tuple(
                    self.tfidf_matrix.shape
                ),
            })

        return information
