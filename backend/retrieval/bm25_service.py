import heapq
import json
import math
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from django.conf import settings
from rank_bm25 import BM25Okapi

from datasets.dataset_loader import DatasetLoader
from document_store.repository import (
    DocumentStoreRepository,
)
from preprocessing.preprocessing_service import (
    TextPreprocessor,
)


class BM25IndexError(RuntimeError):
    """
    Raised when a saved BM25 index is invalid or incomplete.
    """


class BM25RetrievalService:
    """
    Dataset-aware BM25 retrieval service.

    Supported modes:

    1. SQLite inverted BM25
       Used by the complete Quora and Clinical Trials collections.
       Only postings for query terms are read from disk.

    2. Legacy saved BM25
       Loads documents and tokenized_corpus artifacts, then rebuilds
       BM25Okapi in memory. Retained for backward compatibility.

    3. In-memory development BM25
       Used by sample_dataset and automated tests when no saved index
       is available.

    The SQLite index stores raw term frequencies, so k1 and b remain
    configurable at query time without rebuilding the index.
    """

    INVERTED_REQUIRED_DATASETS = {
        "quora",
        "clinical_trials",
    }

    SQLITE_VARIABLE_CHUNK_SIZE = 800
    POSTINGS_FETCH_SIZE = 100_000

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        k1: float = 1.5,
        b: float = 0.75,
        use_saved_index: bool = True,
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()

        self.k1 = float(k1)
        self.b = float(b)
        self.use_saved_index = bool(
            use_saved_index
        )

        self._validate_parameters()

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
            / "bm25"
        )

        # SQLite inverted-index artifacts.
        self.manifest_path = (
            self.index_dir
            / "manifest.json"
        )
        self.vocabulary_path = (
            self.index_dir
            / "vocabulary.joblib"
        )
        self.document_frequencies_path = (
            self.index_dir
            / "document_frequencies.npy"
        )
        self.idf_path = (
            self.index_dir
            / "idf.npy"
        )
        self.document_lengths_path = (
            self.index_dir
            / "document_lengths.npy"
        )
        self.doc_ids_path = (
            self.index_dir
            / "doc_ids.joblib"
        )
        self.postings_database_path = (
            self.index_dir
            / "postings.sqlite3"
        )
        self.preprocessing_config_path = (
            self.index_dir
            / "preprocessing_config.json"
        )

        # Legacy artifacts.
        self.legacy_documents_path = (
            self.index_dir
            / "documents.joblib"
        )
        self.legacy_tokenized_corpus_path = (
            self.index_dir
            / "tokenized_corpus.joblib"
        )
        self.legacy_metadata_path = (
            self.index_dir
            / "metadata.json"
        )

        self.mode: Optional[str] = None

        # SQLite inverted-index state.
        self.manifest: Dict[str, Any] = {}
        self.vocabulary: Dict[str, int] = {}
        self.document_frequencies: Optional[
            np.ndarray
        ] = None
        self.idf: Optional[np.ndarray] = None
        self.document_lengths: Optional[
            np.ndarray
        ] = None
        self.doc_ids: List[str] = []
        self.document_count = 0
        self.average_document_length = 0.0

        # Legacy/in-memory state.
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[
            List[str]
        ] = []
        self.bm25: Optional[BM25Okapi] = None

        self._document_repository: Optional[
            DocumentStoreRepository
        ] = None

        self._initialize_index()

    def _validate_parameters(self):
        if not self.dataset_key:
            raise ValueError(
                "dataset_key cannot be empty."
            )

        if self.k1 <= 0.0:
            raise ValueError(
                "BM25 k1 must be greater than zero."
            )

        if not 0.0 <= self.b <= 1.0:
            raise ValueError(
                "BM25 b must be between 0 and 1."
            )

    def _initialize_index(self):
        if (
            self.use_saved_index
            and self.manifest_path.is_file()
        ):
            self.load_inverted_index()
            return

        if (
            self.use_saved_index
            and self.legacy_index_exists()
        ):
            self.load_legacy_index()
            return

        if (
            self.dataset_key
            in self.INVERTED_REQUIRED_DATASETS
        ):
            raise FileNotFoundError(
                "The complete dataset requires a SQLite "
                "inverted BM25 index, but no manifest was "
                f"found: {self.manifest_path}"
            )

        self.load_documents()
        self.build_index()

    def saved_index_exists(self) -> bool:
        return (
            self.manifest_path.is_file()
            or self.legacy_index_exists()
        )

    def legacy_index_exists(self) -> bool:
        return all(
            path.is_file()
            for path in [
                self.legacy_documents_path,
                self.legacy_tokenized_corpus_path,
                self.preprocessing_config_path,
            ]
        )

    def load_inverted_index(self):
        """
        Load lightweight inverted-index metadata and NumPy arrays.

        The postings database is opened only while a search is running.
        """
        manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        self._validate_inverted_manifest(
            manifest
        )

        vocabulary = joblib.load(
            self.vocabulary_path
        )
        document_frequencies = np.load(
            self.document_frequencies_path,
            allow_pickle=False,
        )
        idf = np.load(
            self.idf_path,
            allow_pickle=False,
        )
        document_lengths = np.load(
            self.document_lengths_path,
            allow_pickle=False,
        )
        doc_ids = joblib.load(
            self.doc_ids_path
        )

        if not isinstance(vocabulary, dict):
            raise BM25IndexError(
                "The BM25 vocabulary artifact must "
                "contain a dictionary."
            )

        if not isinstance(doc_ids, list):
            raise BM25IndexError(
                "The BM25 doc_ids artifact must "
                "contain a list."
            )

        vocabulary_size = int(
            manifest["vocabulary_size"]
        )
        document_count = int(
            manifest["document_count"]
        )

        if not (
            len(vocabulary)
            == len(document_frequencies)
            == len(idf)
            == vocabulary_size
        ):
            raise BM25IndexError(
                "BM25 vocabulary, document-frequency, "
                "IDF, and manifest sizes do not match."
            )

        if not (
            len(document_lengths)
            == len(doc_ids)
            == document_count
        ):
            raise BM25IndexError(
                "BM25 document arrays do not match "
                "the manifest document count."
            )

        self.manifest = manifest
        self.vocabulary = vocabulary
        self.document_frequencies = (
            document_frequencies.astype(
                np.int64,
                copy=False,
            )
        )
        self.idf = idf.astype(
            np.float32,
            copy=False,
        )
        self.document_lengths = (
            document_lengths.astype(
                np.float32,
                copy=False,
            )
        )
        self.doc_ids = [
            str(doc_id)
            for doc_id in doc_ids
        ]
        self.document_count = document_count
        self.average_document_length = float(
            manifest[
                "average_document_length"
            ]
        )

        if self.average_document_length <= 0.0:
            raise BM25IndexError(
                "BM25 average document length "
                "must be greater than zero."
            )

        self.mode = "inverted"

    def _validate_inverted_manifest(
        self,
        manifest: Dict[str, Any],
    ):
        if (
            manifest.get("index_type")
            != "sqlite_inverted_bm25"
        ):
            raise BM25IndexError(
                "Unsupported BM25 index type: "
                f"{manifest.get('index_type')}"
            )

        if manifest.get("version") != 1:
            raise BM25IndexError(
                "Unsupported BM25 index version: "
                f"{manifest.get('version')}"
            )

        if (
            manifest.get("dataset")
            != self.dataset_key
        ):
            raise BM25IndexError(
                "BM25 manifest dataset mismatch. "
                f"Expected '{self.dataset_key}', "
                f"found '{manifest.get('dataset')}'."
            )

        required_paths = [
            self.vocabulary_path,
            self.document_frequencies_path,
            self.idf_path,
            self.document_lengths_path,
            self.doc_ids_path,
            self.postings_database_path,
            self.preprocessing_config_path,
        ]

        missing_paths = [
            path
            for path in required_paths
            if not path.is_file()
        ]

        if missing_paths:
            formatted = "\n".join(
                f"- {path}"
                for path in missing_paths
            )

            raise BM25IndexError(
                "Required BM25 index artifacts are "
                f"missing:\n{formatted}"
            )

        saved_preprocessing = json.loads(
            self.preprocessing_config_path
            .read_text(
                encoding="utf-8"
            )
        )

        current_preprocessing = (
            self.preprocessor
            .get_configuration()
        )

        if saved_preprocessing != current_preprocessing:
            raise BM25IndexError(
                "The BM25 index was built with a "
                "different preprocessing configuration. "
                "Rebuild the index."
            )

        if (
            manifest.get("preprocessing")
            != current_preprocessing
        ):
            raise BM25IndexError(
                "The preprocessing configuration in "
                "the BM25 manifest does not match the "
                "current service configuration."
            )

        vocabulary_size = int(
            manifest.get(
                "vocabulary_size",
                0,
            )
        )
        document_count = int(
            manifest.get(
                "document_count",
                0,
            )
        )
        posting_count = int(
            manifest.get(
                "posting_count",
                -1,
            )
        )

        if vocabulary_size <= 0:
            raise BM25IndexError(
                "The BM25 vocabulary is empty."
            )

        if document_count <= 0:
            raise BM25IndexError(
                "The BM25 document count is invalid."
            )

        if posting_count < 0:
            raise BM25IndexError(
                "The BM25 posting count is invalid."
            )

        self._validate_document_store_count(
            document_count
        )

        self._validate_postings_database(
            expected_posting_count=posting_count,
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
            raise BM25IndexError(
                f"Dataset '{self.dataset_key}' is "
                "missing from the document store."
            )

        counts = repository.get_dataset_counts(
            self.dataset_key
        )

        actual_count = int(
            counts["documents"]
        )

        if actual_count != expected_document_count:
            raise BM25IndexError(
                "BM25 index and document store have "
                "different document counts. "
                f"Index: {expected_document_count}, "
                f"database: {actual_count}."
            )

    def _validate_postings_database(
        self,
        expected_posting_count: int,
    ):
        with self._postings_connection() as connection:
            schema_row = connection.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'postings'
                """
            ).fetchone()

            if schema_row is None:
                raise BM25IndexError(
                    "The BM25 postings table is missing."
                )

            actual_posting_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM postings
                    """
                ).fetchone()[0]
            )

        if (
            actual_posting_count
            != expected_posting_count
        ):
            raise BM25IndexError(
                "BM25 postings count does not match "
                "the manifest. "
                f"Manifest: {expected_posting_count}, "
                f"database: {actual_posting_count}."
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
                raise BM25IndexError(
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

    @contextmanager
    def _postings_connection(self):
        """
        Open the final postings database in read-only mode.

        A short-lived connection keeps Django requests thread-safe and
        avoids Windows file-lock issues.
        """
        database_uri = (
            f"{self.postings_database_path.as_uri()}"
            "?mode=ro"
        )

        connection = sqlite3.connect(
            database_uri,
            uri=True,
            timeout=60.0,
            check_same_thread=False,
        )

        try:
            connection.row_factory = sqlite3.Row
            connection.execute(
                "PRAGMA query_only = ON"
            )
            connection.execute(
                "PRAGMA temp_store = MEMORY"
            )
            connection.execute(
                "PRAGMA cache_size = -131072"
            )
            yield connection

        finally:
            connection.close()

    @staticmethod
    def _chunked(
        values: Sequence[int],
        chunk_size: int,
    ) -> Iterable[Sequence[int]]:
        for start in range(
            0,
            len(values),
            chunk_size,
        ):
            yield values[
                start:start + chunk_size
            ]

    def load_legacy_index(self):
        """
        Load the previous documents/tokenized-corpus BM25 format.
        """
        self._validate_saved_preprocessing_config()

        self.documents = joblib.load(
            self.legacy_documents_path
        )
        self.tokenized_corpus = joblib.load(
            self.legacy_tokenized_corpus_path
        )

        self._validate_loaded_legacy_index()
        self.build_index()

        self.document_count = len(
            self.documents
        )
        self.mode = "legacy"

    def _validate_saved_preprocessing_config(
        self,
    ):
        saved_config = json.loads(
            self.preprocessing_config_path
            .read_text(
                encoding="utf-8"
            )
        )

        current_config = (
            self.preprocessor
            .get_configuration()
        )

        if saved_config != current_config:
            raise BM25IndexError(
                "The saved BM25 index was built "
                "with a different preprocessing "
                "configuration. Rebuild the index."
            )

    def _validate_loaded_legacy_index(self):
        if not isinstance(
            self.documents,
            list,
        ):
            raise BM25IndexError(
                "Invalid BM25 documents artifact: "
                "expected a list."
            )

        if not isinstance(
            self.tokenized_corpus,
            list,
        ):
            raise BM25IndexError(
                "Invalid BM25 tokenized corpus: "
                "expected a list."
            )

        if not self.documents:
            raise BM25IndexError(
                f"BM25 index for dataset "
                f"'{self.dataset_key}' is empty."
            )

        if (
            len(self.documents)
            != len(self.tokenized_corpus)
        ):
            raise BM25IndexError(
                "Invalid BM25 index: document "
                "count does not match tokenized "
                "corpus count."
            )

        for index, tokens in enumerate(
            self.tokenized_corpus
        ):
            if not isinstance(tokens, list):
                raise BM25IndexError(
                    "Invalid BM25 tokenized document "
                    f"at position {index}: "
                    "expected a list."
                )

    def load_documents(self):
        """
        Load a small development collection directly.

        Complete Quora and Clinical Trials collections intentionally do
        not use this path.
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

        self.tokenized_corpus = []

        for document in self.documents:
            search_text = (
                self._get_document_search_text(
                    document
                )
            )

            tokens = (
                self.preprocessor
                .preprocess_tokens(
                    search_text
                )
            )

            self.tokenized_corpus.append(
                tokens
            )

        self._validate_loaded_legacy_index()

    @staticmethod
    def _get_document_search_text(
        document: Dict[str, Any],
    ) -> str:
        title = document.get(
            "title",
            "",
        )
        text = document.get(
            "text",
            "",
        )

        return f"{title} {text}".strip()

    def build_index(self):
        """
        Build the legacy/in-memory BM25Okapi object.
        """
        if not self.tokenized_corpus:
            raise ValueError(
                "Cannot build BM25 from an "
                "empty corpus."
            )

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

        self.document_count = len(
            self.documents
        )

        if self.mode is None:
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

        if result_count <= 0:
            return []

        if self.mode == "inverted":
            return self._search_inverted(
                query=query,
                top_k=result_count,
                hydrate=True,
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
            "BM25 service has no initialized "
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
        Search multiple queries while reusing one SQLite connection.

        The current implementation processes each query independently
        inside a shared read-only database connection. It avoids repeated
        connection setup and supports evaluation without document
        hydration. Query results are identical to search().
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
                "query_batch_size must be greater "
                "than zero."
            )

        normalized_queries = []

        for query_index, query in enumerate(
            queries
        ):
            if not isinstance(query, str):
                raise ValueError(
                    "Every query must be a string. "
                    f"Invalid query position: "
                    f"{query_index}."
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

        if self.mode != "inverted":
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

        with self._postings_connection() as connection:
            for batch_start in range(
                0,
                len(normalized_queries),
                query_batch_size,
            ):
                batch_queries = normalized_queries[
                    batch_start:
                    batch_start + query_batch_size
                ]

                for query in batch_queries:
                    if not query:
                        all_results.append([])
                        continue

                    all_results.append(
                        self._search_inverted(
                            query=query,
                            top_k=result_count,
                            hydrate=hydrate,
                            connection=connection,
                        )
                    )

        return all_results

    def _prepare_inverted_query(
        self,
        query: str,
    ) -> Dict[int, int]:
        """
        Return indexed term IDs and their query frequencies.
        """
        tokens = (
            self.preprocessor
            .preprocess_tokens(query)
        )

        if not tokens:
            return {}

        token_counts = Counter(tokens)
        term_query_frequencies: Dict[
            int,
            int,
        ] = {}

        for term, query_frequency in (
            token_counts.items()
        ):
            term_id = self.vocabulary.get(
                term
            )

            if term_id is None:
                continue

            term_query_frequencies[
                int(term_id)
            ] = int(query_frequency)

        return term_query_frequencies

    def _score_query_terms(
        self,
        term_query_frequencies: Dict[int, int],
        connection: sqlite3.Connection,
    ) -> np.ndarray:
        if (
            self.idf is None
            or self.document_lengths is None
        ):
            raise BM25IndexError(
                "BM25 inverted arrays are not loaded."
            )

        scores = np.zeros(
            self.document_count,
            dtype=np.float32,
        )

        term_ids = list(
            term_query_frequencies
        )

        for term_id_chunk in self._chunked(
            term_ids,
            self.SQLITE_VARIABLE_CHUNK_SIZE,
        ):
            placeholders = ",".join(
                "?"
                for _ in term_id_chunk
            )

            cursor = connection.execute(
                f"""
                SELECT
                    term_id,
                    doc_index,
                    term_frequency
                FROM postings
                WHERE term_id IN ({placeholders})
                ORDER BY term_id, doc_index
                """,
                list(term_id_chunk),
            )

            while True:
                rows = cursor.fetchmany(
                    self.POSTINGS_FETCH_SIZE
                )

                if not rows:
                    break

                term_ids_array = np.fromiter(
                    (
                        int(row["term_id"])
                        for row in rows
                    ),
                    dtype=np.int64,
                    count=len(rows),
                )

                doc_indices = np.fromiter(
                    (
                        int(row["doc_index"])
                        for row in rows
                    ),
                    dtype=np.int64,
                    count=len(rows),
                )

                term_frequencies = np.fromiter(
                    (
                        int(row["term_frequency"])
                        for row in rows
                    ),
                    dtype=np.float32,
                    count=len(rows),
                )

                query_frequencies = np.fromiter(
                    (
                        term_query_frequencies[
                            int(term_id)
                        ]
                        for term_id in term_ids_array
                    ),
                    dtype=np.float32,
                    count=len(rows),
                )

                document_lengths = (
                    self.document_lengths[
                        doc_indices
                    ]
                )

                length_normalization = (
                    1.0
                    - self.b
                    + self.b
                    * (
                        document_lengths
                        / self.average_document_length
                    )
                )

                denominator = (
                    term_frequencies
                    + self.k1
                    * length_normalization
                )

                contributions = (
                    query_frequencies
                    * self.idf[
                        term_ids_array
                    ]
                    * (
                        term_frequencies
                        * (
                            self.k1
                            + 1.0
                        )
                        / denominator
                    )
                ).astype(
                    np.float32,
                    copy=False,
                )

                np.add.at(
                    scores,
                    doc_indices,
                    contributions,
                )

        return scores

    @staticmethod
    def _rank_scores(
        scores: np.ndarray,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Rank positive scores by descending score and ascending document
        position for deterministic tie handling.
        """
        positive_indices = np.flatnonzero(
            scores > 0.0
        )

        if positive_indices.size == 0:
            return []

        if positive_indices.size > top_k:
            positive_scores = scores[
                positive_indices
            ]

            threshold = np.partition(
                positive_scores,
                -top_k,
            )[-top_k]

            candidate_indices = (
                positive_indices[
                    positive_scores
                    >= threshold
                ]
            )
        else:
            candidate_indices = (
                positive_indices
            )

        candidate_scores = scores[
            candidate_indices
        ]

        order = np.lexsort(
            (
                candidate_indices,
                -candidate_scores,
            )
        )

        ranked_indices = (
            candidate_indices[
                order[:top_k]
            ]
        )

        return [
            (
                int(doc_index),
                float(scores[doc_index]),
            )
            for doc_index in ranked_indices
        ]

    def _add_zero_score_fallback(
        self,
        ranked_documents: List[
            Tuple[int, float]
        ],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Fill short result lists deterministically in corpus order.

        This matches the fixed-length behavior expected by evaluation
        while avoiding arbitrary NumPy ordering for zero-score documents.
        """
        if len(ranked_documents) >= top_k:
            return ranked_documents[
                :top_k
            ]

        selected_indices = {
            doc_index
            for doc_index, score
            in ranked_documents
        }

        for doc_index in range(
            self.document_count
        ):
            if doc_index in selected_indices:
                continue

            ranked_documents.append(
                (
                    doc_index,
                    0.0,
                )
            )

            if len(ranked_documents) >= top_k:
                break

        return ranked_documents

    def _search_inverted(
        self,
        query: str,
        top_k: int,
        hydrate: bool,
        connection: Optional[
            sqlite3.Connection
        ] = None,
    ) -> List[Dict[str, Any]]:
        term_query_frequencies = (
            self._prepare_inverted_query(
                query
            )
        )

        if not term_query_frequencies:
            return []

        if connection is None:
            with self._postings_connection() as opened_connection:
                scores = self._score_query_terms(
                    term_query_frequencies,
                    opened_connection,
                )
        else:
            scores = self._score_query_terms(
                term_query_frequencies,
                connection,
            )

        ranked_documents = (
            self._rank_scores(
                scores=scores,
                top_k=top_k,
            )
        )

        ranked_documents = (
            self._add_zero_score_fallback(
                ranked_documents=ranked_documents,
                top_k=top_k,
            )
        )

        if hydrate:
            return self._hydrate_inverted_results(
                ranked_documents
            )

        return [
            {
                "rank": rank,
                "doc_id": self.doc_ids[
                    doc_index
                ],
                "score": round(
                    float(score),
                    6,
                ),
            }
            for rank, (
                doc_index,
                score,
            ) in enumerate(
                ranked_documents,
                start=1,
            )
        ]

    def _hydrate_inverted_results(
        self,
        ranked_documents: List[
            Tuple[int, float]
        ],
    ) -> List[Dict[str, Any]]:
        ordered_doc_ids = [
            self.doc_ids[
                doc_index
            ]
            for doc_index, score
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
            raise BM25IndexError(
                "BM25 returned document IDs missing "
                "from the document store. "
                f"Examples: {missing_doc_ids[:10]}"
            )

        results = []

        for rank, (
            (
                doc_index,
                score,
            ),
            doc_id,
        ) in enumerate(
            zip(
                ranked_documents,
                ordered_doc_ids,
            ),
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
        if self.bm25 is None:
            raise RuntimeError(
                "Legacy BM25 index is not "
                "initialized."
            )

        tokenized_query = (
            self.preprocessor
            .preprocess_tokens(query)
        )

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(
            tokenized_query
        )

        result_count = min(
            int(top_k),
            len(self.documents),
        )

        if result_count == 0:
            return []

        if result_count < len(scores):
            candidate_indices = np.argpartition(
                scores,
                -result_count,
            )[-result_count:]

            candidate_scores = scores[
                candidate_indices
            ]

            order = np.lexsort(
                (
                    candidate_indices,
                    -candidate_scores,
                )
            )

            ranked_indices = (
                candidate_indices[
                    order
                ]
            )
        else:
            all_indices = np.arange(
                len(scores)
            )

            order = np.lexsort(
                (
                    all_indices,
                    -scores,
                )
            )

            ranked_indices = (
                all_indices[
                    order
                ]
            )

        results = []

        for rank, index in enumerate(
            ranked_indices[:result_count],
            start=1,
        ):
            document_index = int(
                index
            )
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
        information: Dict[str, Any] = {
            "dataset": self.dataset_key,
            "mode": self.mode,
            "document_count": (
                self.document_count
            ),
            "k1": self.k1,
            "b": self.b,
            "saved_index_used": (
                self.use_saved_index
                and self.saved_index_exists()
            ),
            "index_directory": str(
                self.index_dir
            ),
            "preprocessing": (
                self.preprocessor
                .get_configuration()
            ),
        }

        if self.mode == "inverted":
            information.update({
                "index_type": (
                    self.manifest[
                        "index_type"
                    ]
                ),
                "vocabulary_size": len(
                    self.vocabulary
                ),
                "posting_count": int(
                    self.manifest[
                        "posting_count"
                    ]
                ),
                "average_document_length": (
                    self.average_document_length
                ),
                "average_idf": float(
                    self.manifest[
                        "average_idf"
                    ]
                ),
                "epsilon": float(
                    self.manifest[
                        "epsilon"
                    ]
                ),
                "min_df": int(
                    self.manifest[
                        "min_df"
                    ]
                ),
                "max_df": float(
                    self.manifest[
                        "max_df"
                    ]
                ),
                "max_features": (
                    self.manifest[
                        "max_features"
                    ]
                ),
                "postings_database": str(
                    self.postings_database_path
                ),
            })

        else:
            metadata = {}

            if self.legacy_metadata_path.is_file():
                metadata = json.loads(
                    self.legacy_metadata_path
                    .read_text(
                        encoding="utf-8"
                    )
                )

            information[
                "metadata"
            ] = metadata

        return information
