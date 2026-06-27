import gc
import json
import math
import os
import shutil
import sqlite3
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from document_store.repository import DocumentStoreRepository
from preprocessing.preprocessing_service import TextPreprocessor


_ATOMIC_JSON_LOCKS: Dict[str, threading.Lock] = {}
_ATOMIC_JSON_LOCKS_GUARD = threading.Lock()


class ScalableBm25BuildError(RuntimeError):
    """
    Raised when the scalable BM25 index cannot be built or validated.
    """


class ScalableBm25IndexBuilder:
    """
    Build a disk-backed BM25 inverted index from the SQLite document store.

    The build uses two streaming passes:

    1. Statistics pass
       - Stable document positions.
       - Document lengths.
       - Exact document frequency for every term.

    2. Postings pass
       - One posting per (term, document).
       - Raw term frequency is stored so k1 and b remain configurable
         at query time.

    Generated artifacts:

        indexes/<dataset>/bm25/
        ├── manifest.json
        ├── vocabulary.joblib
        ├── document_frequencies.npy
        ├── idf.npy
        ├── document_lengths.npy
        ├── doc_ids.joblib
        ├── preprocessing_config.json
        ├── postings.sqlite3
        └── _build/
            ├── checkpoint.json
            └── statistics.sqlite3   (removed after successful build)
    """

    CHECKPOINT_VERSION = 1
    INDEX_VERSION = 1

    def __init__(
        self,
        repository: DocumentStoreRepository,
        dataset_key: str,
        indexes_root: str | Path,
        batch_size: int = 1000,
        min_df: int = 1,
        max_df: float = 0.98,
        max_features: Optional[int] = None,
        epsilon: float = 0.25,
        checkpoint_every_docs: int = 1000,
    ):
        self.repository = repository
        self.dataset_key = str(dataset_key).strip()
        self.indexes_root = Path(indexes_root).expanduser().resolve()

        self.batch_size = int(batch_size)
        self.min_df = int(min_df)
        self.max_df = float(max_df)
        self.max_features = (
            int(max_features)
            if max_features is not None
            else None
        )
        self.epsilon = float(epsilon)
        self.checkpoint_every_docs = int(
            checkpoint_every_docs
        )

        self._validate_parameters()

        self.preprocessor = TextPreprocessor(
            dataset_key=self.dataset_key
        )

        self.index_dir = (
            self.indexes_root
            / self.dataset_key
            / "bm25"
        )
        self.work_dir = self.index_dir / "_build"

        self.statistics_database_path = (
            self.work_dir / "statistics.sqlite3"
        )
        self.postings_database_path = (
            self.index_dir / "postings.sqlite3"
        )
        self.checkpoint_path = (
            self.work_dir / "checkpoint.json"
        )

        self.vocabulary_path = (
            self.index_dir / "vocabulary.joblib"
        )
        self.document_frequencies_path = (
            self.index_dir / "document_frequencies.npy"
        )
        self.idf_path = self.index_dir / "idf.npy"
        self.document_lengths_path = (
            self.index_dir / "document_lengths.npy"
        )
        self.doc_ids_path = (
            self.index_dir / "doc_ids.joblib"
        )
        self.preprocessing_config_path = (
            self.index_dir / "preprocessing_config.json"
        )
        self.manifest_path = (
            self.index_dir / "manifest.json"
        )

        self.dataset_metadata: Optional[
            Dict[str, Any]
        ] = None
        self.document_count = 0

    def _validate_parameters(self):
        if not self.dataset_key:
            raise ValueError(
                "dataset_key cannot be empty."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if self.min_df <= 0:
            raise ValueError(
                "min_df must be greater than zero."
            )

        if self.max_df <= 0.0:
            raise ValueError(
                "max_df must be greater than zero."
            )

        if (
            self.max_features is not None
            and self.max_features <= 0
        ):
            raise ValueError(
                "max_features must be greater than zero."
            )

        if self.epsilon < 0.0:
            raise ValueError(
                "epsilon must be greater than or equal to zero."
            )

        if self.checkpoint_every_docs <= 0:
            raise ValueError(
                "checkpoint_every_docs must be greater than zero."
            )

    def build(
        self,
        overwrite: bool = False,
        resume: bool = False,
    ) -> Dict[str, Any]:
        if overwrite and resume:
            raise ValueError(
                "overwrite and resume cannot both be enabled."
            )

        self.repository.initialize()
        self._load_dataset_information()

        checkpoint = self._prepare_build(
            overwrite=overwrite,
            resume=resume,
        )

        if checkpoint["stage"] == "statistics":
            self._build_statistics(checkpoint)
            self._finalize_vocabulary_and_documents(
                checkpoint
            )

        if checkpoint["stage"] == "postings":
            self._build_postings(checkpoint)

        if checkpoint["stage"] != "complete":
            raise ScalableBm25BuildError(
                "BM25 build did not reach the complete stage."
            )

        # A previous Windows run may have completed every index artifact
        # but failed only while deleting the temporary statistics database.
        self._remove_statistics_database()

        return self.validate_index()

    def _load_dataset_information(self):
        dataset = self.repository.get_dataset(
            self.dataset_key
        )

        if dataset is None:
            raise ScalableBm25BuildError(
                f"Dataset '{self.dataset_key}' is not "
                "present in the document store."
            )

        counts = self.repository.get_dataset_counts(
            self.dataset_key
        )

        self.dataset_metadata = dataset
        self.document_count = int(
            counts["documents"]
        )

        if self.document_count <= 0:
            raise ScalableBm25BuildError(
                f"Dataset '{self.dataset_key}' contains "
                "no documents."
            )

    def _configuration(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "epsilon": self.epsilon,
            "preprocessing": (
                self.preprocessor.get_configuration()
            ),
        }

    def _new_checkpoint(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "dataset": self.dataset_key,
            "document_count": self.document_count,
            "configuration": self._configuration(),
            "stage": "statistics",
            "statistics": {
                "last_rowid": 0,
                "processed_documents": 0,
            },
            "postings": {
                "last_rowid": 0,
                "processed_documents": 0,
                "inserted_postings": 0,
            },
            "vocabulary_size": 0,
            "max_df_count": None,
            "average_document_length": None,
            "average_idf": None,
        }

    def _prepare_build(
        self,
        overwrite: bool,
        resume: bool,
    ) -> Dict[str, Any]:
        if overwrite:
            shutil.rmtree(
                self.index_dir,
                ignore_errors=True,
            )

        if resume:
            if not self.checkpoint_path.is_file():
                raise ScalableBm25BuildError(
                    "Cannot resume because no BM25 checkpoint "
                    f"exists: {self.checkpoint_path}"
                )

            checkpoint = self._load_checkpoint()
            self._validate_checkpoint(checkpoint)
            return checkpoint

        if self.manifest_path.is_file():
            raise ScalableBm25BuildError(
                "A completed BM25 index already exists. "
                "Use --overwrite to rebuild it."
            )

        if self.checkpoint_path.is_file():
            raise ScalableBm25BuildError(
                "An incomplete BM25 build exists. "
                "Use --resume or --overwrite."
            )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_statistics_database()

        checkpoint = self._new_checkpoint()
        self._save_checkpoint(checkpoint)
        return checkpoint

    def _load_checkpoint(self) -> Dict[str, Any]:
        with self.checkpoint_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _validate_checkpoint(
        self,
        checkpoint: Dict[str, Any],
    ):
        if (
            checkpoint.get("version")
            != self.CHECKPOINT_VERSION
        ):
            raise ScalableBm25BuildError(
                "Unsupported BM25 checkpoint version."
            )

        if checkpoint.get("dataset") != self.dataset_key:
            raise ScalableBm25BuildError(
                "The BM25 checkpoint belongs to another dataset."
            )

        if (
            int(checkpoint.get("document_count", -1))
            != self.document_count
        ):
            raise ScalableBm25BuildError(
                "The document-store count changed after "
                "the BM25 build started."
            )

        if (
            checkpoint.get("configuration")
            != self._configuration()
        ):
            raise ScalableBm25BuildError(
                "BM25 build parameters differ from the saved "
                "checkpoint. Use the original parameters or "
                "restart with --overwrite."
            )

        if (
            checkpoint.get("stage")
            in {"statistics", "postings"}
            and not self.statistics_database_path.is_file()
        ):
            raise ScalableBm25BuildError(
                "The BM25 statistics database required for "
                "resume is missing. Restart with --overwrite."
            )

    def _save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
    ):
        self._write_json_atomic(
            self.checkpoint_path,
            checkpoint,
        )

    @contextmanager
    def _statistics_connection(
        self,
    ):
        """
        Open the temporary statistics database and always close it.

        sqlite3.Connection used directly in a with-statement commits or
        rolls back, but does not close the underlying file handle. That
        behavior prevents deleting SQLite files on Windows.
        """
        self.statistics_database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            self.statistics_database_path,
            timeout=120.0,
        )

        try:
            connection.row_factory = sqlite3.Row
            connection.execute(
                "PRAGMA journal_mode = WAL"
            )
            connection.execute(
                "PRAGMA synchronous = NORMAL"
            )
            connection.execute(
                "PRAGMA temp_store = MEMORY"
            )
            connection.execute(
                "PRAGMA cache_size = -131072"
            )

            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    @contextmanager
    def _postings_connection(
        self,
    ):
        """
        Open the postings database and always close it.
        """
        self.postings_database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        database_exists = (
            self.postings_database_path.exists()
        )

        connection = sqlite3.connect(
            self.postings_database_path,
            timeout=120.0,
        )

        try:
            connection.row_factory = sqlite3.Row

            if not database_exists:
                connection.execute(
                    "PRAGMA page_size = 32768"
                )

            connection.execute(
                "PRAGMA journal_mode = WAL"
            )
            connection.execute(
                "PRAGMA synchronous = NORMAL"
            )
            connection.execute(
                "PRAGMA temp_store = MEMORY"
            )
            connection.execute(
                "PRAGMA cache_size = -262144"
            )

            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def _initialize_statistics_database(self):
        with self._statistics_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_index INTEGER PRIMARY KEY,
                    store_rowid INTEGER NOT NULL UNIQUE,
                    doc_id TEXT NOT NULL UNIQUE,
                    document_length INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS term_df (
                    term TEXT PRIMARY KEY,
                    document_frequency INTEGER NOT NULL
                ) WITHOUT ROWID;

                CREATE INDEX IF NOT EXISTS
                    idx_documents_store_rowid
                ON documents(store_rowid);

                CREATE INDEX IF NOT EXISTS
                    idx_term_df_frequency
                ON term_df(document_frequency);
                """
            )
            connection.commit()

    def _initialize_postings_database(self):
        with self._postings_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS postings (
                    term_id INTEGER NOT NULL,
                    doc_index INTEGER NOT NULL,
                    term_frequency INTEGER NOT NULL,
                    PRIMARY KEY (term_id, doc_index)
                ) WITHOUT ROWID;

                CREATE TABLE IF NOT EXISTS processed_documents (
                    store_rowid INTEGER PRIMARY KEY,
                    doc_index INTEGER NOT NULL UNIQUE
                ) WITHOUT ROWID;
                """
            )
            connection.commit()

    def _iter_document_rows(
        self,
        after_rowid: int,
    ) -> Iterable[Sequence[sqlite3.Row]]:
        with self.repository.connection() as connection:
            cursor = connection.execute(
                """
                SELECT
                    rowid AS store_rowid,
                    doc_id,
                    title,
                    raw_text
                FROM documents
                WHERE dataset_key = ?
                  AND rowid > ?
                ORDER BY rowid
                """,
                (
                    self.dataset_key,
                    int(after_rowid),
                ),
            )

            while True:
                rows = cursor.fetchmany(
                    self.batch_size
                )

                if not rows:
                    break

                yield rows

    @staticmethod
    def _chunked(
        values: Sequence[int],
        chunk_size: int = 800,
    ) -> Iterable[Sequence[int]]:
        for start in range(
            0,
            len(values),
            chunk_size,
        ):
            yield values[
                start:start + chunk_size
            ]

    def _existing_statistics_rowids(
        self,
        connection: sqlite3.Connection,
        rowids: Sequence[int],
    ) -> set[int]:
        existing: set[int] = set()

        for chunk in self._chunked(rowids):
            placeholders = ",".join(
                "?" for _ in chunk
            )

            rows = connection.execute(
                f"""
                SELECT store_rowid
                FROM documents
                WHERE store_rowid IN ({placeholders})
                """,
                list(chunk),
            ).fetchall()

            existing.update(
                int(row["store_rowid"])
                for row in rows
            )

        return existing

    def _existing_postings_rowids(
        self,
        connection: sqlite3.Connection,
        rowids: Sequence[int],
    ) -> set[int]:
        existing: set[int] = set()

        for chunk in self._chunked(rowids):
            placeholders = ",".join(
                "?" for _ in chunk
            )

            rows = connection.execute(
                f"""
                SELECT store_rowid
                FROM processed_documents
                WHERE store_rowid IN ({placeholders})
                """,
                list(chunk),
            ).fetchall()

            existing.update(
                int(row["store_rowid"])
                for row in rows
            )

        return existing

    def _build_statistics(
        self,
        checkpoint: Dict[str, Any],
    ):
        state = checkpoint["statistics"]

        print()
        print("=" * 70)
        print(
            "BM25 statistics pass: "
            f"{self.dataset_key}"
        )
        print(
            "Already processed: "
            f"{state['processed_documents']:,}"
        )
        print("=" * 70)

        self._initialize_statistics_database()
        last_checkpoint_documents = int(
            state["processed_documents"]
        )

        for rows in self._iter_document_rows(
            after_rowid=state["last_rowid"]
        ):
            rowids = [
                int(row["store_rowid"])
                for row in rows
            ]

            with self._statistics_connection() as connection:
                existing_rowids = (
                    self._existing_statistics_rowids(
                        connection,
                        rowids,
                    )
                )

                next_doc_index = int(
                    connection.execute(
                        """
                        SELECT COALESCE(MAX(doc_index), -1) + 1
                        FROM documents
                        """
                    ).fetchone()[0]
                )

                document_rows: List[
                    Tuple[int, int, str, int]
                ] = []
                batch_document_frequency = Counter()

                for row in rows:
                    store_rowid = int(
                        row["store_rowid"]
                    )

                    if store_rowid in existing_rowids:
                        continue

                    search_text = (
                        f"{row['title'] or ''} "
                        f"{row['raw_text'] or ''}"
                    ).strip()

                    tokens = (
                        self.preprocessor
                        .preprocess_tokens(search_text)
                    )

                    document_rows.append(
                        (
                            next_doc_index,
                            store_rowid,
                            str(row["doc_id"]),
                            len(tokens),
                        )
                    )

                    next_doc_index += 1

                    batch_document_frequency.update(
                        set(tokens)
                    )

                if document_rows:
                    connection.executemany(
                        """
                        INSERT INTO documents (
                            doc_index,
                            store_rowid,
                            doc_id,
                            document_length
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        document_rows,
                    )

                if batch_document_frequency:
                    connection.executemany(
                        """
                        INSERT INTO term_df (
                            term,
                            document_frequency
                        )
                        VALUES (?, ?)

                        ON CONFLICT(term)
                        DO UPDATE SET
                            document_frequency =
                                document_frequency
                                + excluded.document_frequency
                        """,
                        list(
                            batch_document_frequency.items()
                        ),
                    )

                connection.commit()

                statistics_row = connection.execute(
                    """
                    SELECT
                        COUNT(*) AS document_count,
                        COALESCE(MAX(store_rowid), 0)
                            AS last_rowid
                    FROM documents
                    """
                ).fetchone()

            state["processed_documents"] = int(
                statistics_row["document_count"]
            )
            state["last_rowid"] = int(
                statistics_row["last_rowid"]
            )

            if (
                state["processed_documents"]
                - last_checkpoint_documents
                >= self.checkpoint_every_docs
                or state["processed_documents"]
                >= self.document_count
            ):
                self._save_checkpoint(checkpoint)
                last_checkpoint_documents = int(
                    state["processed_documents"]
                )

            print(
                "Statistics documents: "
                f"{state['processed_documents']:,}/"
                f"{self.document_count:,}",
                flush=True,
            )

        if (
            int(state["processed_documents"])
            != self.document_count
        ):
            raise ScalableBm25BuildError(
                "BM25 statistics document count mismatch. "
                f"Expected {self.document_count}, processed "
                f"{state['processed_documents']}."
            )

    def _resolve_max_df_count(self) -> int:
        if self.max_df <= 1.0:
            max_df_count = int(
                math.floor(
                    self.max_df
                    * self.document_count
                )
            )
        else:
            max_df_count = int(
                self.max_df
            )

        return max(
            max_df_count,
            self.min_df,
        )

    def _select_vocabulary_rows(
        self,
        connection: sqlite3.Connection,
        max_df_count: int,
    ) -> List[sqlite3.Row]:
        sql = """
            SELECT
                term,
                document_frequency
            FROM term_df
            WHERE document_frequency >= ?
              AND document_frequency <= ?
            ORDER BY
                document_frequency DESC,
                term ASC
        """

        parameters: List[Any] = [
            self.min_df,
            max_df_count,
        ]

        if self.max_features is not None:
            sql += " LIMIT ?"
            parameters.append(
                self.max_features
            )

        return connection.execute(
            sql,
            parameters,
        ).fetchall()

    def _finalize_vocabulary_and_documents(
        self,
        checkpoint: Dict[str, Any],
    ):
        print()
        print("=" * 70)
        print(
            "Finalizing BM25 vocabulary and document arrays: "
            f"{self.dataset_key}"
        )
        print("=" * 70)

        max_df_count = self._resolve_max_df_count()

        with self._statistics_connection() as connection:
            vocabulary_rows = (
                self._select_vocabulary_rows(
                    connection,
                    max_df_count,
                )
            )

            if not vocabulary_rows:
                raise ScalableBm25BuildError(
                    "No terms survived BM25 document-frequency "
                    "filtering."
                )

            document_rows = connection.execute(
                """
                SELECT
                    doc_index,
                    doc_id,
                    document_length
                FROM documents
                ORDER BY doc_index
                """
            ).fetchall()

        if len(document_rows) != self.document_count:
            raise ScalableBm25BuildError(
                "BM25 document metadata count does not match "
                "the document store."
            )

        for expected_index, row in enumerate(
            document_rows
        ):
            if int(row["doc_index"]) != expected_index:
                raise ScalableBm25BuildError(
                    "BM25 document positions are not continuous."
                )

        vocabulary = {
            str(row["term"]): term_id
            for term_id, row in enumerate(
                vocabulary_rows
            )
        }

        document_frequencies = np.asarray(
            [
                int(row["document_frequency"])
                for row in vocabulary_rows
            ],
            dtype=np.int64,
        )

        raw_idf = (
            np.log(
                float(self.document_count)
                - document_frequencies
                + 0.5
            )
            - np.log(
                document_frequencies
                + 0.5
            )
        )

        average_idf = float(
            raw_idf.mean()
        )
        epsilon_floor = (
            self.epsilon
            * average_idf
        )

        idf = np.where(
            raw_idf < 0.0,
            epsilon_floor,
            raw_idf,
        ).astype(np.float32)

        doc_ids = [
            str(row["doc_id"])
            for row in document_rows
        ]

        document_lengths = np.asarray(
            [
                int(row["document_length"])
                for row in document_rows
            ],
            dtype=np.uint32,
        )

        average_document_length = float(
            document_lengths.mean()
        )

        self._write_joblib_atomic(
            self.vocabulary_path,
            vocabulary,
        )
        self._write_numpy_atomic(
            self.document_frequencies_path,
            document_frequencies,
        )
        self._write_numpy_atomic(
            self.idf_path,
            idf,
        )
        self._write_numpy_atomic(
            self.document_lengths_path,
            document_lengths,
        )
        self._write_joblib_atomic(
            self.doc_ids_path,
            doc_ids,
        )
        self._write_json_atomic(
            self.preprocessing_config_path,
            self.preprocessor.get_configuration(),
        )

        self._initialize_postings_database()

        checkpoint["vocabulary_size"] = len(
            vocabulary
        )
        checkpoint["max_df_count"] = max_df_count
        checkpoint["average_document_length"] = (
            average_document_length
        )
        checkpoint["average_idf"] = average_idf
        checkpoint["stage"] = "postings"

        self._save_checkpoint(checkpoint)

        print(
            "Vocabulary size: "
            f"{len(vocabulary):,}"
        )
        print(
            "Average document length: "
            f"{average_document_length:,.4f}"
        )
        print(
            "Average IDF: "
            f"{average_idf:,.6f}"
        )

    def _load_statistics_doc_index_map(
        self,
        rowids: Sequence[int],
    ) -> Dict[int, int]:
        result: Dict[int, int] = {}

        with self._statistics_connection() as connection:
            for chunk in self._chunked(rowids):
                placeholders = ",".join(
                    "?" for _ in chunk
                )

                rows = connection.execute(
                    f"""
                    SELECT
                        store_rowid,
                        doc_index
                    FROM documents
                    WHERE store_rowid IN ({placeholders})
                    """,
                    list(chunk),
                ).fetchall()

                result.update(
                    {
                        int(row["store_rowid"]): int(
                            row["doc_index"]
                        )
                        for row in rows
                    }
                )

        return result

    def _build_postings(
        self,
        checkpoint: Dict[str, Any],
    ):
        if not self.vocabulary_path.is_file():
            raise ScalableBm25BuildError(
                "BM25 vocabulary artifact is missing."
            )

        vocabulary = joblib.load(
            self.vocabulary_path
        )

        if not isinstance(vocabulary, dict):
            raise ScalableBm25BuildError(
                "BM25 vocabulary artifact must contain a dictionary."
            )

        self._initialize_postings_database()

        state = checkpoint["postings"]
        last_checkpoint_documents = int(
            state["processed_documents"]
        )

        print()
        print("=" * 70)
        print(
            "BM25 postings pass: "
            f"{self.dataset_key}"
        )
        print(
            "Already processed: "
            f"{state['processed_documents']:,}"
        )
        print("=" * 70)

        for rows in self._iter_document_rows(
            after_rowid=state["last_rowid"]
        ):
            rowids = [
                int(row["store_rowid"])
                for row in rows
            ]

            doc_index_by_rowid = (
                self._load_statistics_doc_index_map(
                    rowids
                )
            )

            if len(doc_index_by_rowid) != len(rows):
                raise ScalableBm25BuildError(
                    "BM25 postings pass could not resolve every "
                    "document position from the statistics database."
                )

            posting_rows: List[
                Tuple[int, int, int]
            ] = []
            processed_document_rows: List[
                Tuple[int, int]
            ] = []

            with self._postings_connection() as connection:
                existing_rowids = (
                    self._existing_postings_rowids(
                        connection,
                        rowids,
                    )
                )

                for row in rows:
                    store_rowid = int(
                        row["store_rowid"]
                    )

                    if store_rowid in existing_rowids:
                        continue

                    doc_index = doc_index_by_rowid[
                        store_rowid
                    ]

                    search_text = (
                        f"{row['title'] or ''} "
                        f"{row['raw_text'] or ''}"
                    ).strip()

                    tokens = (
                        self.preprocessor
                        .preprocess_tokens(search_text)
                    )

                    term_counts = Counter(tokens)

                    for term, frequency in term_counts.items():
                        term_id = vocabulary.get(term)

                        if term_id is None:
                            continue

                        posting_rows.append(
                            (
                                int(term_id),
                                int(doc_index),
                                int(frequency),
                            )
                        )

                    processed_document_rows.append(
                        (
                            store_rowid,
                            int(doc_index),
                        )
                    )

                if posting_rows:
                    connection.executemany(
                        """
                        INSERT INTO postings (
                            term_id,
                            doc_index,
                            term_frequency
                        )
                        VALUES (?, ?, ?)

                        ON CONFLICT(term_id, doc_index)
                        DO UPDATE SET
                            term_frequency =
                                excluded.term_frequency
                        """,
                        posting_rows,
                    )

                if processed_document_rows:
                    connection.executemany(
                        """
                        INSERT INTO processed_documents (
                            store_rowid,
                            doc_index
                        )
                        VALUES (?, ?)

                        ON CONFLICT(store_rowid)
                        DO UPDATE SET
                            doc_index = excluded.doc_index
                        """,
                        processed_document_rows,
                    )

                connection.commit()

                progress_row = connection.execute(
                    """
                    SELECT
                        COUNT(*) AS document_count,
                        COALESCE(MAX(store_rowid), 0)
                            AS last_rowid
                    FROM processed_documents
                    """
                ).fetchone()

            state["processed_documents"] = int(
                progress_row["document_count"]
            )
            state["last_rowid"] = int(
                progress_row["last_rowid"]
            )
            state["inserted_postings"] += len(
                posting_rows
            )

            if (
                state["processed_documents"]
                - last_checkpoint_documents
                >= self.checkpoint_every_docs
                or state["processed_documents"]
                >= self.document_count
            ):
                self._save_checkpoint(checkpoint)
                last_checkpoint_documents = int(
                    state["processed_documents"]
                )

            print(
                "Postings documents: "
                f"{state['processed_documents']:,}/"
                f"{self.document_count:,}; "
                "batch postings: "
                f"{len(posting_rows):,}",
                flush=True,
            )

        if (
            int(state["processed_documents"])
            != self.document_count
        ):
            raise ScalableBm25BuildError(
                "BM25 postings document count mismatch. "
                f"Expected {self.document_count}, processed "
                f"{state['processed_documents']}."
            )

        with self._postings_connection() as connection:
            posting_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM postings"
                ).fetchone()[0]
            )
            connection.execute("ANALYZE")
            connection.execute("PRAGMA optimize")
            connection.commit()

        manifest = self._build_manifest(
            checkpoint=checkpoint,
            posting_count=posting_count,
        )

        self._write_json_atomic(
            self.manifest_path,
            manifest,
        )

        checkpoint["stage"] = "complete"
        checkpoint["postings"][
            "stored_postings"
        ] = posting_count
        self._save_checkpoint(checkpoint)

        self._remove_statistics_database()

    def _build_manifest(
        self,
        checkpoint: Dict[str, Any],
        posting_count: int,
    ) -> Dict[str, Any]:
        if self.dataset_metadata is None:
            raise ScalableBm25BuildError(
                "Dataset metadata was not loaded."
            )

        return {
            "index_type": "sqlite_inverted_bm25",
            "version": self.INDEX_VERSION,
            "dataset": self.dataset_key,
            "display_name": self.dataset_metadata[
                "display_name"
            ],
            "ir_dataset_id": self.dataset_metadata[
                "ir_dataset_id"
            ],
            "document_count": self.document_count,
            "vocabulary_size": checkpoint[
                "vocabulary_size"
            ],
            "posting_count": int(posting_count),
            "average_document_length": checkpoint[
                "average_document_length"
            ],
            "average_idf": checkpoint[
                "average_idf"
            ],
            "epsilon": self.epsilon,
            "idf_formula": (
                "rank_bm25_okapi_with_epsilon_floor"
            ),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_df_count": checkpoint[
                "max_df_count"
            ],
            "max_features": self.max_features,
            "preprocessing": (
                self.preprocessor.get_configuration()
            ),
            "vocabulary_file": self.vocabulary_path.name,
            "document_frequencies_file": (
                self.document_frequencies_path.name
            ),
            "idf_file": self.idf_path.name,
            "document_lengths_file": (
                self.document_lengths_path.name
            ),
            "doc_ids_file": self.doc_ids_path.name,
            "postings_database_file": (
                self.postings_database_path.name
            ),
            "preprocessing_config_file": (
                self.preprocessing_config_path.name
            ),
            "postings_schema": {
                "table": "postings",
                "columns": [
                    "term_id",
                    "doc_index",
                    "term_frequency",
                ],
                "primary_key": [
                    "term_id",
                    "doc_index",
                ],
            },
        }

    def validate_index(self) -> Dict[str, Any]:
        required_paths = [
            self.manifest_path,
            self.vocabulary_path,
            self.document_frequencies_path,
            self.idf_path,
            self.document_lengths_path,
            self.doc_ids_path,
            self.preprocessing_config_path,
            self.postings_database_path,
        ]

        missing_paths = [
            path
            for path in required_paths
            if not path.is_file()
        ]

        if missing_paths:
            raise ScalableBm25BuildError(
                "BM25 index artifacts are missing:\n"
                + "\n".join(
                    f"- {path}"
                    for path in missing_paths
                )
            )

        manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if manifest.get("dataset") != self.dataset_key:
            raise ScalableBm25BuildError(
                "BM25 manifest dataset mismatch."
            )

        if (
            int(manifest.get("document_count", -1))
            != self.document_count
        ):
            raise ScalableBm25BuildError(
                "BM25 manifest document count mismatch."
            )

        vocabulary = joblib.load(
            self.vocabulary_path
        )
        doc_ids = joblib.load(
            self.doc_ids_path
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

        vocabulary_size = len(vocabulary)

        if not (
            vocabulary_size
            == len(document_frequencies)
            == len(idf)
            == int(manifest["vocabulary_size"])
        ):
            raise ScalableBm25BuildError(
                "BM25 vocabulary, document-frequency, IDF, "
                "and manifest sizes do not match."
            )

        if not (
            len(doc_ids)
            == len(document_lengths)
            == self.document_count
        ):
            raise ScalableBm25BuildError(
                "BM25 document ID and document-length counts "
                "do not match the document store."
            )

        with self._postings_connection() as connection:
            posting_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM postings"
                ).fetchone()[0]
            )

            processed_document_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM processed_documents"
                ).fetchone()[0]
            )

            invalid_postings = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM postings
                    WHERE term_id < 0
                       OR term_id >= ?
                       OR doc_index < 0
                       OR doc_index >= ?
                       OR term_frequency <= 0
                    """,
                    (
                        vocabulary_size,
                        self.document_count,
                    ),
                ).fetchone()[0]
            )

        if processed_document_count != self.document_count:
            raise ScalableBm25BuildError(
                "BM25 postings database does not mark every "
                "document as processed."
            )

        if posting_count != int(
            manifest["posting_count"]
        ):
            raise ScalableBm25BuildError(
                "BM25 posting count does not match the manifest."
            )

        if invalid_postings:
            raise ScalableBm25BuildError(
                "BM25 postings contain invalid term IDs, "
                "document positions, or term frequencies."
            )

        return {
            "dataset": self.dataset_key,
            "document_count": self.document_count,
            "vocabulary_size": vocabulary_size,
            "posting_count": posting_count,
            "average_document_length": float(
                manifest["average_document_length"]
            ),
            "index_directory": str(
                self.index_dir
            ),
            "manifest_path": str(
                self.manifest_path
            ),
            "postings_database_path": str(
                self.postings_database_path
            ),
            "validation_passed": True,
        }

    def _remove_statistics_database(
        self,
        attempts: int = 8,
        delay_seconds: float = 0.25,
    ):
        """
        Remove temporary statistics SQLite artifacts.

        Windows can retain a file lock briefly after the final connection
        closes. Retry deletion instead of failing an otherwise completed
        index build.
        """
        paths = [
            self.statistics_database_path,
            Path(
                f"{self.statistics_database_path}-wal"
            ),
            Path(
                f"{self.statistics_database_path}-shm"
            ),
        ]

        gc.collect()

        for path in paths:
            if not path.exists():
                continue

            last_error = None

            for attempt in range(attempts):
                try:
                    path.unlink(missing_ok=True)
                    last_error = None
                    break

                except PermissionError as error:
                    last_error = error

                    if attempt + 1 < attempts:
                        time.sleep(delay_seconds)

            if last_error is not None:
                print(
                    "Warning: could not remove temporary BM25 "
                    f"statistics file: {path}. "
                    f"Reason: {last_error}",
                    flush=True,
                )

    @staticmethod
    def _get_atomic_json_lock(
        output_path: Path,
    ) -> threading.Lock:
        lock_key = str(
            Path(output_path).expanduser().resolve()
        )

        with _ATOMIC_JSON_LOCKS_GUARD:
            lock = _ATOMIC_JSON_LOCKS.get(
                lock_key
            )

            if lock is None:
                lock = threading.Lock()
                _ATOMIC_JSON_LOCKS[lock_key] = lock

            return lock

    @staticmethod
    def _retryable_replace_error(
        error: OSError,
    ) -> bool:
        return (
            isinstance(error, PermissionError)
            or getattr(error, "winerror", None) == 5
        )

    @staticmethod
    def _unique_json_temp_path(
        output_path: Path,
    ) -> Path:
        return output_path.with_name(
            f"{output_path.name}."
            f"{os.getpid()}."
            f"{threading.get_ident()}."
            f"{uuid.uuid4().hex}.tmp"
        )

    @staticmethod
    def _write_json_atomic(
        output_path: Path,
        value: Dict[str, Any],
        attempts: int = 12,
        initial_delay_seconds: float = 0.05,
    ):
        output_path = Path(
            output_path
        ).expanduser().resolve()
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            ScalableBm25IndexBuilder
            ._unique_json_temp_path(output_path)
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    value,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            lock = (
                ScalableBm25IndexBuilder
                ._get_atomic_json_lock(output_path)
            )
            last_error: Optional[OSError] = None
            delay_seconds = float(
                initial_delay_seconds
            )

            with lock:
                for attempt in range(
                    max(1, int(attempts))
                ):
                    try:
                        os.replace(
                            str(temporary_path),
                            str(output_path),
                        )
                        return

                    except OSError as error:
                        last_error = error

                        if (
                            not ScalableBm25IndexBuilder
                            ._retryable_replace_error(error)
                            or attempt + 1 >= attempts
                        ):
                            break

                        time.sleep(
                            delay_seconds
                        )
                        delay_seconds = min(
                            delay_seconds * 2.0,
                            2.0,
                        )

        finally:
            try:
                temporary_path.unlink(
                    missing_ok=True,
                )
            except OSError:
                pass

        raise ScalableBm25BuildError(
            "Could not atomically write BM25 JSON file after "
            f"{attempts} attempt(s): {output_path}. "
            "Windows may still have a transient lock on the index "
            "directory from antivirus, indexing, Explorer preview, "
            "or another process. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _write_numpy_atomic(
        output_path: Path,
        value: np.ndarray,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = output_path.with_suffix(
            ".tmp.npy"
        )
        np.save(
            temporary_path,
            value,
            allow_pickle=False,
        )
        temporary_path.replace(output_path)

    @staticmethod
    def _write_joblib_atomic(
        output_path: Path,
        value: Any,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = output_path.with_suffix(
            ".tmp.joblib"
        )
        joblib.dump(
            value,
            temporary_path,
            compress=3,
        )
        temporary_path.replace(output_path)
