import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from document_store.repository import (
    DocumentStoreRepository,
)
from indexing.scalable_bm25_index import (
    ScalableBm25BuildError,
    ScalableBm25IndexBuilder,
)
from preprocessing.preprocessing_service import (
    TextPreprocessor,
)


DEFAULT_DISTRIBUTED_NUM_SHARDS = 4
MIN_DISTRIBUTED_NUM_SHARDS = 1
MAX_DISTRIBUTED_NUM_SHARDS = 1024
DEFAULT_DISTRIBUTED_RRF_K = 60
SHARDING_STRATEGY = (
    "stable_sha256_doc_id_mod_num_shards"
)
MERGE_METHOD = "RRF"


class DistributedBm25BuildError(RuntimeError):
    """
    Raised when a distributed BM25 index cannot be built or validated.
    """


def validate_num_shards(
    num_shards: int,
) -> int:
    try:
        parsed = int(num_shards)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "num_shards must be an integer."
        ) from error

    if parsed < MIN_DISTRIBUTED_NUM_SHARDS:
        raise ValueError(
            "num_shards must be at least "
            f"{MIN_DISTRIBUTED_NUM_SHARDS}."
        )

    if parsed > MAX_DISTRIBUTED_NUM_SHARDS:
        raise ValueError(
            "num_shards must not exceed "
            f"{MAX_DISTRIBUTED_NUM_SHARDS}."
        )

    return parsed


def stable_hash(
    value: str,
) -> int:
    normalized_value = str(value).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        normalized_value
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def assign_shard(
    doc_id: str,
    num_shards: int,
) -> int:
    parsed_num_shards = validate_num_shards(
        num_shards
    )

    normalized_doc_id = str(doc_id).strip()

    if not normalized_doc_id:
        raise ValueError(
            "doc_id cannot be empty."
        )

    return stable_hash(
        normalized_doc_id
    ) % parsed_num_shards


def _chunked(
    values: Sequence[Any],
    chunk_size: int = 1000,
) -> Iterable[Sequence[Any]]:
    for start in range(
        0,
        len(values),
        chunk_size,
    ):
        yield values[start:start + chunk_size]


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def build_source_corpus_signature(
    repository: DocumentStoreRepository,
    dataset_key: str,
) -> Dict[str, Any]:
    dataset = repository.get_dataset(
        dataset_key
    )

    counts = repository.get_dataset_counts(
        dataset_key
    )

    with repository.connection() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS document_count,
                MIN(rowid) AS min_document_rowid,
                MAX(rowid) AS max_document_rowid,
                MIN(doc_id) AS min_doc_id,
                MAX(doc_id) AS max_doc_id
            FROM documents
            WHERE dataset_key = ?
            """,
            (dataset_key,),
        ).fetchone()

    return {
        "dataset_key": dataset_key,
        "document_count": int(
            row["document_count"]
        ),
        "query_count": counts["queries"],
        "qrel_count": counts["qrels"],
        "min_document_rowid": row[
            "min_document_rowid"
        ],
        "max_document_rowid": row[
            "max_document_rowid"
        ],
        "min_doc_id": row["min_doc_id"],
        "max_doc_id": row["max_doc_id"],
        "dataset_updated_at": (
            dataset.get("updated_at")
            if dataset
            else None
        ),
        "dataset_imported_at": (
            dataset.get("imported_at")
            if dataset
            else None
        ),
    }


def write_shard_assignments_database(
    repository: DocumentStoreRepository,
    dataset_key: str,
    output_path: Path,
    num_shards: int,
    batch_size: int = 5000,
) -> Dict[str, int]:
    parsed_num_shards = validate_num_shards(
        num_shards
    )

    counts = {
        f"shard_{shard_id}": 0
        for shard_id in range(parsed_num_shards)
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    connection = sqlite3.connect(
        output_path,
        timeout=120.0,
    )

    try:
        connection.execute(
            "PRAGMA journal_mode = WAL"
        )
        connection.execute(
            "PRAGMA synchronous = NORMAL"
        )
        connection.executescript(
            """
            CREATE TABLE shard_assignments (
                doc_id TEXT PRIMARY KEY,
                shard_id INTEGER NOT NULL
            ) WITHOUT ROWID;

            CREATE INDEX idx_shard_assignments_shard_id
            ON shard_assignments(shard_id);
            """
        )

        with repository.connection() as source_connection:
            cursor = source_connection.execute(
                """
                SELECT doc_id
                FROM documents
                WHERE dataset_key = ?
                ORDER BY rowid
                """,
                (dataset_key,),
            )

            while True:
                rows = cursor.fetchmany(
                    batch_size
                )

                if not rows:
                    break

                assignment_rows = []

                for row in rows:
                    doc_id = str(row["doc_id"])
                    shard_id = assign_shard(
                        doc_id,
                        parsed_num_shards,
                    )
                    counts[
                        f"shard_{shard_id}"
                    ] += 1
                    assignment_rows.append(
                        (
                            doc_id,
                            shard_id,
                        )
                    )

                connection.executemany(
                    """
                    INSERT INTO shard_assignments (
                        doc_id,
                        shard_id
                    )
                    VALUES (?, ?)
                    """,
                    assignment_rows,
                )
                connection.commit()

        connection.execute("ANALYZE")
        connection.commit()

    finally:
        connection.close()

    return counts


class DistributedBm25ShardIndexBuilder(
    ScalableBm25IndexBuilder
):
    """
    Build one independent BM25 shard under distributed_bm25/shards.
    """

    def __init__(
        self,
        repository: DocumentStoreRepository,
        dataset_key: str,
        distributed_index_dir: str | Path,
        shard_id: int,
        num_shards: int,
        expected_document_count: int,
        total_documents: int,
        batch_size: int = 1000,
        min_df: int = 1,
        max_df: float = 0.98,
        max_features: int | None = None,
        epsilon: float = 0.25,
        checkpoint_every_docs: int = 1000,
    ):
        self.shard_id = int(shard_id)
        self.num_shards = validate_num_shards(
            num_shards
        )

        if not 0 <= self.shard_id < self.num_shards:
            raise ValueError(
                "shard_id must be in the range "
                "0 <= shard_id < num_shards."
            )

        self.distributed_index_dir = Path(
            distributed_index_dir
        ).expanduser().resolve()
        self.expected_document_count = int(
            expected_document_count
        )
        self.total_documents = int(
            total_documents
        )

        super().__init__(
            repository=repository,
            dataset_key=dataset_key,
            indexes_root=self.distributed_index_dir,
            batch_size=batch_size,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            epsilon=epsilon,
            checkpoint_every_docs=checkpoint_every_docs,
        )

        self.index_dir = (
            self.distributed_index_dir
            / "shards"
            / f"shard_{self.shard_id}"
        )
        self._configure_artifact_paths()

    def _configure_artifact_paths(self):
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
            self.index_dir
            / "document_frequencies.npy"
        )
        self.idf_path = (
            self.index_dir / "idf.npy"
        )
        self.document_lengths_path = (
            self.index_dir
            / "document_lengths.npy"
        )
        self.doc_ids_path = (
            self.index_dir / "doc_ids.joblib"
        )
        self.preprocessing_config_path = (
            self.index_dir
            / "preprocessing_config.json"
        )
        self.manifest_path = (
            self.index_dir / "manifest.json"
        )
        self.shard_manifest_path = (
            self.index_dir / "shard_manifest.json"
        )

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

        actual_total_documents = int(
            counts["documents"]
        )

        if actual_total_documents != self.total_documents:
            raise ScalableBm25BuildError(
                "Document-store count changed while building "
                "the distributed BM25 index. "
                f"Expected {self.total_documents}, found "
                f"{actual_total_documents}."
            )

        self.dataset_metadata = dataset
        self.document_count = int(
            self.expected_document_count
        )

        if self.document_count <= 0:
            raise ScalableBm25BuildError(
                f"Shard {self.shard_id} contains no "
                "documents. Use fewer shards."
            )

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

                shard_rows = [
                    row
                    for row in rows
                    if assign_shard(
                        str(row["doc_id"]),
                        self.num_shards,
                    )
                    == self.shard_id
                ]

                if shard_rows:
                    yield shard_rows

    def build(
        self,
        overwrite: bool = False,
        resume: bool = False,
    ) -> Dict[str, Any]:
        summary = super().build(
            overwrite=overwrite,
            resume=resume,
        )

        self._write_shard_manifest(
            summary
        )

        summary.update({
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "shard_manifest_path": str(
                self.shard_manifest_path
            ),
        })

        return summary

    def _write_shard_manifest(
        self,
        summary: Dict[str, Any],
    ):
        base_manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        shard_manifest = {
            "dataset": self.dataset_key,
            "model": "distributed_bm25",
            "shard_id": self.shard_id,
            "shard_name": (
                f"shard_{self.shard_id}"
            ),
            "num_shards": self.num_shards,
            "document_count": int(
                summary["document_count"]
            ),
            "total_documents": (
                self.total_documents
            ),
            "sharding_strategy": (
                SHARDING_STRATEGY
            ),
            "bm25_manifest_file": (
                self.manifest_path.name
            ),
            "doc_ids_file": (
                self.doc_ids_path.name
            ),
            "postings_database_file": (
                self.postings_database_path.name
            ),
            "preprocessing": base_manifest.get(
                "preprocessing"
            ),
            "created_at": _utc_now(),
        }

        self._write_json_atomic(
            self.shard_manifest_path,
            shard_manifest,
        )


class DistributedBm25IndexBuilder:
    """
    Build local multi-shard BM25 indexes and a coordinator manifest.
    """

    def __init__(
        self,
        repository: DocumentStoreRepository,
        dataset_key: str,
        indexes_root: str | Path,
        num_shards: int = DEFAULT_DISTRIBUTED_NUM_SHARDS,
        batch_size: int = 1000,
        min_df: int = 1,
        max_df: float = 0.98,
        max_features: int | None = None,
        epsilon: float = 0.25,
        rrf_k: int = DEFAULT_DISTRIBUTED_RRF_K,
        checkpoint_every_docs: int = 1000,
    ):
        self.repository = repository
        self.dataset_key = str(
            dataset_key
        ).strip()
        self.indexes_root = Path(
            indexes_root
        ).expanduser().resolve()
        self.num_shards = validate_num_shards(
            num_shards
        )
        self.batch_size = int(batch_size)
        self.min_df = int(min_df)
        self.max_df = float(max_df)
        self.max_features = (
            int(max_features)
            if max_features is not None
            else None
        )
        self.epsilon = float(epsilon)
        self.rrf_k = int(rrf_k)
        self.checkpoint_every_docs = int(
            checkpoint_every_docs
        )

        self._validate_parameters()

        self.index_dir = (
            self.indexes_root
            / self.dataset_key
            / "distributed_bm25"
        )
        self.shards_dir = (
            self.index_dir / "shards"
        )
        self.manifest_path = (
            self.index_dir / "manifest.json"
        )
        self.assignment_database_path = (
            self.index_dir
            / "shard_assignments.sqlite3"
        )
        self.preprocessor = TextPreprocessor(
            dataset_key=self.dataset_key
        )

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

        if self.rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        if self.checkpoint_every_docs <= 0:
            raise ValueError(
                "checkpoint_every_docs must be greater than zero."
            )

    def build(
        self,
        force: bool = False,
        clean: bool = False,
    ) -> Dict[str, Any]:
        self.repository.initialize()

        dataset = self.repository.get_dataset(
            self.dataset_key
        )

        if dataset is None:
            raise DistributedBm25BuildError(
                f"Dataset '{self.dataset_key}' is not "
                "present in the document store. Run "
                "the ingestion script before building "
                "distributed BM25."
            )

        counts = self.repository.get_dataset_counts(
            self.dataset_key
        )
        total_documents = int(
            counts["documents"]
        )

        if total_documents <= 0:
            raise DistributedBm25BuildError(
                f"Dataset '{self.dataset_key}' contains "
                "no documents."
            )

        if self.num_shards > total_documents:
            raise DistributedBm25BuildError(
                "num_shards cannot exceed the corpus "
                "document count. "
                f"num_shards={self.num_shards}, "
                f"documents={total_documents}."
            )

        if self.index_dir.exists():
            if not (force or clean):
                raise DistributedBm25BuildError(
                    "A distributed BM25 index or incomplete build "
                    "already exists. Use --clean to remove the "
                    "distributed_bm25 directory and rebuild from "
                    "scratch, or --force to overwrite an existing "
                    "completed index."
                )

            shutil.rmtree(
                self.index_dir,
                ignore_errors=True,
            )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "Computing deterministic shard assignments..."
        )

        shard_document_counts = (
            write_shard_assignments_database(
                repository=self.repository,
                dataset_key=self.dataset_key,
                output_path=(
                    self.assignment_database_path
                ),
                num_shards=self.num_shards,
                batch_size=max(
                    self.batch_size,
                    1000,
                ),
            )
        )

        empty_shards = [
            shard_name
            for shard_name, count
            in shard_document_counts.items()
            if int(count) <= 0
        ]

        if empty_shards:
            raise DistributedBm25BuildError(
                "Some shards received no documents. "
                "Use fewer shards. Empty shards: "
                + ", ".join(empty_shards)
            )

        if (
            sum(shard_document_counts.values())
            != total_documents
        ):
            raise DistributedBm25BuildError(
                "Shard assignment counts do not match "
                "the corpus document count."
            )

        shard_summaries = []

        for shard_id in range(
            self.num_shards
        ):
            shard_name = f"shard_{shard_id}"
            print()
            print("=" * 70)
            print(
                "Building distributed BM25 "
                f"{shard_name}/"
                f"{self.num_shards - 1}"
            )
            print(
                "Shard documents: "
                f"{shard_document_counts[shard_name]:,}"
            )
            print("=" * 70)

            shard_builder = (
                DistributedBm25ShardIndexBuilder(
                    repository=self.repository,
                    dataset_key=self.dataset_key,
                    distributed_index_dir=(
                        self.index_dir
                    ),
                    shard_id=shard_id,
                    num_shards=self.num_shards,
                    expected_document_count=(
                        shard_document_counts[
                            shard_name
                        ]
                    ),
                    total_documents=(
                        total_documents
                    ),
                    batch_size=self.batch_size,
                    min_df=self.min_df,
                    max_df=self.max_df,
                    max_features=self.max_features,
                    epsilon=self.epsilon,
                    checkpoint_every_docs=(
                        self.checkpoint_every_docs
                    ),
                )
            )

            shard_summaries.append(
                shard_builder.build(
                    overwrite=False,
                    resume=False,
                )
            )

        observed_total = sum(
            int(summary["document_count"])
            for summary in shard_summaries
        )

        if observed_total != total_documents:
            raise DistributedBm25BuildError(
                "Built shard document counts do not "
                "match the corpus document count. "
                f"Expected {total_documents}, built "
                f"{observed_total}."
            )

        manifest = self._build_root_manifest(
            dataset=dataset,
            total_documents=total_documents,
            shard_document_counts=(
                shard_document_counts
            ),
            shard_summaries=(
                shard_summaries
            ),
        )

        ScalableBm25IndexBuilder._write_json_atomic(
            self.manifest_path,
            manifest,
        )

        return {
            "dataset": self.dataset_key,
            "model": "distributed_bm25",
            "num_shards": self.num_shards,
            "total_documents": total_documents,
            "shard_document_counts": (
                shard_document_counts
            ),
            "index_directory": str(
                self.index_dir
            ),
            "manifest_path": str(
                self.manifest_path
            ),
            "assignment_database_path": str(
                self.assignment_database_path
            ),
            "validation_passed": True,
        }

    def _build_root_manifest(
        self,
        dataset: Dict[str, Any],
        total_documents: int,
        shard_document_counts: Dict[str, int],
        shard_summaries: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_key,
            "display_name": dataset.get(
                "display_name"
            ),
            "ir_dataset_id": dataset.get(
                "ir_dataset_id"
            ),
            "model": "distributed_bm25",
            "version": 1,
            "num_shards": self.num_shards,
            "total_documents": total_documents,
            "sharding_strategy": (
                SHARDING_STRATEGY
            ),
            "merge_method": MERGE_METHOD,
            "rrf_k": self.rrf_k,
            "created_at": _utc_now(),
            "shard_document_counts": (
                shard_document_counts
            ),
            "shards_directory": "shards",
            "shard_manifest_file": (
                "shard_manifest.json"
            ),
            "assignment_metadata": {
                "type": "sqlite",
                "file": (
                    self.assignment_database_path.name
                ),
                "table": "shard_assignments",
                "columns": [
                    "doc_id",
                    "shard_id",
                ],
            },
            "preprocessing": (
                self.preprocessor
                .get_configuration()
            ),
            "source_corpus_signature": (
                build_source_corpus_signature(
                    repository=self.repository,
                    dataset_key=self.dataset_key,
                )
            ),
            "bm25_parameters": {
                "min_df": self.min_df,
                "max_df": self.max_df,
                "max_features": (
                    self.max_features
                ),
                "epsilon": self.epsilon,
            },
            "shards": [
                {
                    "shard_id": int(
                        summary["shard_id"]
                    ),
                    "shard_name": (
                        f"shard_"
                        f"{summary['shard_id']}"
                    ),
                    "document_count": int(
                        summary[
                            "document_count"
                        ]
                    ),
                    "manifest_path": (
                        "shards/"
                        f"shard_{summary['shard_id']}"
                        "/shard_manifest.json"
                    ),
                    "bm25_manifest_path": (
                        "shards/"
                        f"shard_{summary['shard_id']}"
                        "/manifest.json"
                    ),
                }
                for summary in shard_summaries
            ],
        }
