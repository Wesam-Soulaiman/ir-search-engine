import json
import math
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from scipy import sparse

from document_store.repository import (
    DocumentStoreRepository,
)
from preprocessing.preprocessing_service import (
    TextPreprocessor,
)


class ScalableTfidfBuildError(RuntimeError):
    """
    Raised when scalable TF-IDF index construction fails.
    """


class ScalableTfidfIndexBuilder:
    """
    Build a complete TF-IDF index without loading the entire corpus
    or sparse matrix into memory.

    Build stages:

    1. Stream documents and calculate exact document frequency.
    2. Select the vocabulary using min_df, max_df, and max_features.
    3. Calculate IDF values.
    4. Stream documents again and create normalized CSR shards.
    5. Save a manifest describing all generated artifacts.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        repository: DocumentStoreRepository,
        dataset_key: str,
        indexes_root: str | Path,
        batch_size: int = 1000,
        shard_size: int = 5000,
        min_df: int = 2,
        max_df: float = 0.98,
        max_features: Optional[int] = 100_000,
        sublinear_tf: bool = True,
    ):
        self.repository = repository
        self.dataset_key = str(
            dataset_key
        ).strip()

        self.indexes_root = Path(
            indexes_root
        ).expanduser().resolve()

        self.batch_size = int(batch_size)
        self.shard_size = int(shard_size)
        self.min_df = int(min_df)
        self.max_df = float(max_df)

        self.max_features = (
            int(max_features)
            if max_features is not None
            else None
        )

        self.sublinear_tf = bool(
            sublinear_tf
        )

        self._validate_parameters()

        self.preprocessor = TextPreprocessor(
            dataset_key=self.dataset_key
        )

        self.index_dir = (
            self.indexes_root
            / self.dataset_key
            / "tfidf"
        )

        self.shards_dir = (
            self.index_dir
            / "shards"
        )

        self.work_dir = (
            self.index_dir
            / "_build"
        )

        self.document_frequency_path = (
            self.work_dir
            / "document_frequency.sqlite3"
        )

        self.checkpoint_path = (
            self.work_dir
            / "checkpoint.json"
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

        self.manifest_path = (
            self.index_dir
            / "manifest.json"
        )

        self.dataset_metadata = None
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

        if self.shard_size <= 0:
            raise ValueError(
                "shard_size must be greater than zero."
            )

        if self.min_df <= 0:
            raise ValueError(
                "min_df must be greater than zero."
            )

        if self.max_df <= 0:
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

        if checkpoint["stage"] == "document_frequency":
            self._build_document_frequency(
                checkpoint
            )

            self._finalize_vocabulary(
                checkpoint
            )

        if checkpoint["stage"] == "shards":
            self._build_matrix_shards(
                checkpoint
            )

        if checkpoint["stage"] != "complete":
            raise ScalableTfidfBuildError(
                "TF-IDF build did not reach the complete stage."
            )

        summary = self.validate_index()

        return summary

    def _load_dataset_information(self):
        dataset = self.repository.get_dataset(
            self.dataset_key
        )

        if dataset is None:
            raise ScalableTfidfBuildError(
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
            raise ScalableTfidfBuildError(
                f"Dataset '{self.dataset_key}' contains "
                "no documents."
            )

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
                raise ScalableTfidfBuildError(
                    "Cannot resume because no TF-IDF "
                    f"checkpoint exists: {self.checkpoint_path}"
                )

            checkpoint = self._load_checkpoint()
            self._validate_checkpoint(
                checkpoint
            )

            return checkpoint

        if self.manifest_path.is_file():
            raise ScalableTfidfBuildError(
                "A completed TF-IDF index already exists. "
                "Use --overwrite to rebuild it."
            )

        if self.checkpoint_path.is_file():
            raise ScalableTfidfBuildError(
                "An incomplete TF-IDF build exists. "
                "Use --resume or --overwrite."
            )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.shards_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_document_frequency_database()

        checkpoint = self._new_checkpoint()
        self._save_checkpoint(checkpoint)

        return checkpoint

    def _new_checkpoint(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "dataset": self.dataset_key,
            "document_count": self.document_count,
            "configuration": self._configuration(),
            "stage": "document_frequency",
            "document_frequency": {
                "last_rowid": 0,
                "processed_documents": 0,
            },
            "shards": {
                "last_rowid": 0,
                "processed_documents": 0,
                "next_shard_id": 0,
                "items": [],
            },
            "vocabulary_size": 0,
            "max_df_count": None,
        }

    def _configuration(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "shard_size": self.shard_size,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "preprocessing": (
                self.preprocessor.get_configuration()
            ),
        }

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
            raise ScalableTfidfBuildError(
                "Unsupported TF-IDF checkpoint version."
            )

        if checkpoint.get("dataset") != self.dataset_key:
            raise ScalableTfidfBuildError(
                "The checkpoint belongs to another dataset."
            )

        if (
            int(
                checkpoint.get(
                    "document_count",
                    -1,
                )
            )
            != self.document_count
        ):
            raise ScalableTfidfBuildError(
                "The document-store count changed after "
                "the TF-IDF build started."
            )

        if (
            checkpoint.get("configuration")
            != self._configuration()
        ):
            raise ScalableTfidfBuildError(
                "TF-IDF build parameters differ from "
                "the saved checkpoint. Use the original "
                "parameters or restart with --overwrite."
            )

    def _save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
    ):
        self.work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            self.checkpoint_path.with_suffix(
                ".tmp"
            )
        )

        temporary_path.write_text(
            json.dumps(
                checkpoint,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            self.checkpoint_path
        )

    def _initialize_document_frequency_database(
        self,
    ):
        with self._document_frequency_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS term_df (
                    term TEXT PRIMARY KEY,
                    document_frequency INTEGER NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_term_df_frequency
                ON term_df(document_frequency)
                """
            )

    def _document_frequency_connection(
        self,
    ) -> sqlite3.Connection:
        self.document_frequency_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            self.document_frequency_path,
            timeout=120.0,
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

        return connection

    def _iter_document_rows(
        self,
        after_rowid: int,
        fetch_size: int,
    ):
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
                    int(fetch_size)
                )

                if not rows:
                    break

                yield rows

    def _build_document_frequency(
        self,
        checkpoint: Dict[str, Any],
    ):
        state = checkpoint[
            "document_frequency"
        ]

        print()
        print("=" * 70)
        print(
            f"TF-IDF document-frequency pass: "
            f"{self.dataset_key}"
        )
        print(
            "Already processed: "
            f"{state['processed_documents']:,}"
        )
        print("=" * 70)

        self._initialize_document_frequency_database()

        for rows in self._iter_document_rows(
            after_rowid=state["last_rowid"],
            fetch_size=self.batch_size,
        ):
            batch_frequencies = Counter()

            for row in rows:
                search_text = (
                    f"{row['title'] or ''} "
                    f"{row['raw_text'] or ''}"
                ).strip()

                tokens = (
                    self.preprocessor.preprocess_tokens(
                        search_text
                    )
                )

                unique_tokens = set(tokens)

                batch_frequencies.update(
                    unique_tokens
                )

            with self._document_frequency_connection() as connection:
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
                        batch_frequencies.items()
                    ),
                )

                connection.commit()

            state["processed_documents"] += len(
                rows
            )

            state["last_rowid"] = int(
                rows[-1]["store_rowid"]
            )

            self._save_checkpoint(
                checkpoint
            )

            print(
                "Document-frequency documents: "
                f"{state['processed_documents']:,}/"
                f"{self.document_count:,}",
                flush=True,
            )

        if (
            state["processed_documents"]
            != self.document_count
        ):
            raise ScalableTfidfBuildError(
                "Document-frequency pass count mismatch. "
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

    def _finalize_vocabulary(
        self,
        checkpoint: Dict[str, Any],
    ):
        print()
        print("=" * 70)
        print(
            f"Selecting TF-IDF vocabulary: "
            f"{self.dataset_key}"
        )
        print("=" * 70)

        max_df_count = (
            self._resolve_max_df_count()
        )

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

        with self._document_frequency_connection() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        if not rows:
            raise ScalableTfidfBuildError(
                "No vocabulary terms survived TF-IDF "
                "document-frequency filtering."
            )

        vocabulary = {
            str(term): index
            for index, (
                term,
                document_frequency,
            ) in enumerate(rows)
        }

        document_frequencies = np.asarray(
            [
                int(document_frequency)
                for (
                    term,
                    document_frequency,
                ) in rows
            ],
            dtype=np.int64,
        )

        idf = (
            np.log(
                (
                    1.0
                    + float(self.document_count)
                )
                / (
                    1.0
                    + document_frequencies
                )
            )
            + 1.0
        ).astype(np.float32)

        temporary_vocabulary_path = (
            self.vocabulary_path.with_suffix(
                ".tmp.joblib"
            )
        )

        joblib.dump(
            vocabulary,
            temporary_vocabulary_path,
            compress=3,
        )

        temporary_vocabulary_path.replace(
            self.vocabulary_path
        )

        temporary_idf_path = (
            self.idf_path.with_suffix(
                ".tmp.npy"
            )
        )

        np.save(
            temporary_idf_path,
            idf,
        )

        temporary_idf_path.replace(
            self.idf_path
        )

        self._write_json_atomic(
            self.preprocessing_config_path,
            self.preprocessor.get_configuration(),
        )

        checkpoint["vocabulary_size"] = len(
            vocabulary
        )

        checkpoint["max_df_count"] = (
            max_df_count
        )

        checkpoint["stage"] = "shards"

        self._save_checkpoint(
            checkpoint
        )

        print(
            "Vocabulary size: "
            f"{len(vocabulary):,}"
        )

        print(
            "Maximum document frequency: "
            f"{max_df_count:,}"
        )

    def _build_matrix_shards(
        self,
        checkpoint: Dict[str, Any],
    ):
        if not self.vocabulary_path.is_file():
            raise ScalableTfidfBuildError(
                "TF-IDF vocabulary artifact is missing."
            )

        if not self.idf_path.is_file():
            raise ScalableTfidfBuildError(
                "TF-IDF IDF artifact is missing."
            )

        vocabulary = joblib.load(
            self.vocabulary_path
        )

        idf = np.load(
            self.idf_path
        )

        if len(vocabulary) != len(idf):
            raise ScalableTfidfBuildError(
                "Vocabulary and IDF sizes do not match."
            )

        state = checkpoint["shards"]

        print()
        print("=" * 70)
        print(
            f"Building TF-IDF sparse shards: "
            f"{self.dataset_key}"
        )
        print(
            "Already processed: "
            f"{state['processed_documents']:,}"
        )
        print("=" * 70)

        for rows in self._iter_document_rows(
            after_rowid=state["last_rowid"],
            fetch_size=self.shard_size,
        ):
            shard_id = int(
                state["next_shard_id"]
            )

            matrix, doc_ids = (
                self._create_sparse_shard(
                    rows=rows,
                    vocabulary=vocabulary,
                    idf=idf,
                )
            )

            matrix_filename = (
                f"matrix_{shard_id:05d}.npz"
            )

            doc_ids_filename = (
                f"doc_ids_{shard_id:05d}.joblib"
            )

            matrix_path = (
                self.shards_dir
                / matrix_filename
            )

            doc_ids_path = (
                self.shards_dir
                / doc_ids_filename
            )

            temporary_matrix_path = (
                self.shards_dir
                / f"matrix_{shard_id:05d}.tmp.npz"
            )

            temporary_doc_ids_path = (
                self.shards_dir
                / f"doc_ids_{shard_id:05d}.tmp.joblib"
            )

            sparse.save_npz(
                temporary_matrix_path,
                matrix,
                compressed=True,
            )

            joblib.dump(
                doc_ids,
                temporary_doc_ids_path,
                compress=3,
            )

            temporary_matrix_path.replace(
                matrix_path
            )

            temporary_doc_ids_path.replace(
                doc_ids_path
            )

            first_rowid = int(
                rows[0]["store_rowid"]
            )

            last_rowid = int(
                rows[-1]["store_rowid"]
            )

            shard_record = {
                "shard_id": shard_id,
                "matrix_file": (
                    f"shards/{matrix_filename}"
                ),
                "doc_ids_file": (
                    f"shards/{doc_ids_filename}"
                ),
                "rows": int(matrix.shape[0]),
                "columns": int(matrix.shape[1]),
                "nnz": int(matrix.nnz),
                "first_store_rowid": first_rowid,
                "last_store_rowid": last_rowid,
            }

            state["items"].append(
                shard_record
            )

            state["processed_documents"] += len(
                rows
            )

            state["last_rowid"] = last_rowid
            state["next_shard_id"] = (
                shard_id + 1
            )

            self._save_checkpoint(
                checkpoint
            )

            print(
                f"Shard {shard_id:05d}: "
                f"{matrix.shape[0]:,} documents, "
                f"{matrix.nnz:,} non-zero values. "
                f"Total: "
                f"{state['processed_documents']:,}/"
                f"{self.document_count:,}",
                flush=True,
            )

        if (
            state["processed_documents"]
            != self.document_count
        ):
            raise ScalableTfidfBuildError(
                "TF-IDF shard count mismatch. "
                f"Expected {self.document_count}, built "
                f"{state['processed_documents']}."
            )

        manifest = self._build_manifest(
            checkpoint
        )

        self._write_json_atomic(
            self.manifest_path,
            manifest,
        )

        checkpoint["stage"] = "complete"

        self._save_checkpoint(
            checkpoint
        )

        self._remove_document_frequency_database()

    def _create_sparse_shard(
        self,
        rows,
        vocabulary: Dict[str, int],
        idf: np.ndarray,
    ):
        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0]
        doc_ids: List[str] = []

        vocabulary_size = len(
            vocabulary
        )

        for row in rows:
            doc_ids.append(
                str(row["doc_id"])
            )

            search_text = (
                f"{row['title'] or ''} "
                f"{row['raw_text'] or ''}"
            ).strip()

            tokens = (
                self.preprocessor.preprocess_tokens(
                    search_text
                )
            )

            term_counts = Counter(tokens)

            weighted_terms = []

            for term, count in term_counts.items():
                term_index = vocabulary.get(
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
                    * float(idf[term_index])
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

            for term_index, value in weighted_terms:
                normalized_value = (
                    value / norm
                    if norm > 0.0
                    else value
                )

                indices.append(
                    term_index
                )

                data.append(
                    normalized_value
                )

            indptr.append(
                len(data)
            )

        matrix = sparse.csr_matrix(
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
                len(rows),
                vocabulary_size,
            ),
            dtype=np.float32,
        )

        matrix.sort_indices()

        return matrix, doc_ids

    def _build_manifest(
        self,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        shard_items = checkpoint[
            "shards"
        ]["items"]

        return {
            "index_type": "sharded_tfidf",
            "version": 1,
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
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_df_count": checkpoint[
                "max_df_count"
            ],
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "normalization": "l2",
            "dtype": "float32",
            "vocabulary_file": (
                self.vocabulary_path.name
            ),
            "idf_file": self.idf_path.name,
            "preprocessing_config_file": (
                self.preprocessing_config_path.name
            ),
            "preprocessing": (
                self.preprocessor.get_configuration()
            ),
            "shard_count": len(
                shard_items
            ),
            "shards": shard_items,
        }

    def validate_index(self) -> Dict[str, Any]:
        if not self.manifest_path.is_file():
            raise ScalableTfidfBuildError(
                "TF-IDF manifest is missing."
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        if (
            manifest["dataset"]
            != self.dataset_key
        ):
            raise ScalableTfidfBuildError(
                "TF-IDF manifest dataset mismatch."
            )

        if (
            manifest["document_count"]
            != self.document_count
        ):
            raise ScalableTfidfBuildError(
                "TF-IDF manifest document-count mismatch."
            )

        total_rows = 0
        total_nnz = 0

        for shard in manifest["shards"]:
            matrix_path = (
                self.index_dir
                / shard["matrix_file"]
            )

            doc_ids_path = (
                self.index_dir
                / shard["doc_ids_file"]
            )

            if not matrix_path.is_file():
                raise ScalableTfidfBuildError(
                    f"Missing matrix shard: {matrix_path}"
                )

            if not doc_ids_path.is_file():
                raise ScalableTfidfBuildError(
                    f"Missing document-ID shard: {doc_ids_path}"
                )

            total_rows += int(
                shard["rows"]
            )

            total_nnz += int(
                shard["nnz"]
            )

        if total_rows != self.document_count:
            raise ScalableTfidfBuildError(
                "TF-IDF shard rows do not match "
                "the document-store count."
            )

        return {
            "dataset": self.dataset_key,
            "document_count": self.document_count,
            "vocabulary_size": manifest[
                "vocabulary_size"
            ],
            "shard_count": manifest[
                "shard_count"
            ],
            "total_rows": total_rows,
            "total_nnz": total_nnz,
            "index_directory": str(
                self.index_dir
            ),
            "manifest_path": str(
                self.manifest_path
            ),
            "validation_passed": True,
        }

    def _remove_document_frequency_database(
        self,
    ):
        for path in [
            self.document_frequency_path,
            Path(
                f"{self.document_frequency_path}-wal"
            ),
            Path(
                f"{self.document_frequency_path}-shm"
            ),
        ]:
            path.unlink(
                missing_ok=True
            )

    @staticmethod
    def _write_json_atomic(
        output_path: Path,
        value: Dict[str, Any],
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            output_path.with_suffix(
                ".tmp"
            )
        )

        temporary_path.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            output_path
        )