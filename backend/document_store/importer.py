import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets.dataset_registry import get_dataset_config
from document_store.repository import DocumentStoreRepository
from preprocessing.preprocessing_service import TextPreprocessor


IR_DATASET_IDS = {
    "quora": "beir/quora/test",
    "clinical_trials": "clinicaltrials/2017/trec-pm-2018",
}


class DatasetIngestionError(RuntimeError):
    """
    Raised when dataset ingestion or validation fails.
    """


class DatasetIngestionService:
    """
    Stream a prepared dataset into the raw-document SQLite store.

    Features:

    - Imports documents, queries, and qrels in batches.
    - Does not load the full collection into memory.
    - Stores original raw document text.
    - Stores dataset-aware processed query text.
    - Supports resuming after interruption.
    - Uses idempotent UPSERT operations.
    - Detects changes to source files before resuming.
    - Rejects reduced development samples unless explicitly allowed.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        repository: DocumentStoreRepository,
        dataset_key: str,
        batch_size: int = 1000,
        checkpoint_dir: Optional[str | Path] = None,
        allow_development_sample: bool = False,
    ):
        self.repository = repository
        self.dataset_key = str(dataset_key).strip()
        self.batch_size = int(batch_size)
        self.allow_development_sample = bool(
            allow_development_sample
        )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if self.dataset_key not in IR_DATASET_IDS:
            available = ", ".join(
                sorted(IR_DATASET_IDS)
            )

            raise ValueError(
                f"Dataset '{self.dataset_key}' is not supported "
                f"by the ingestion service. Available: {available}"
            )

        self.dataset_config = get_dataset_config(
            self.dataset_key
        )

        self.documents_path = Path(
            self.dataset_config["documents_path"]
        ).resolve()

        self.queries_path = Path(
            self.dataset_config["queries_path"]
        ).resolve()

        self.qrels_path = Path(
            self.dataset_config["qrels_path"]
        ).resolve()

        self.metadata_path = (
            self.documents_path.parent
            / "metadata.json"
        )

        if checkpoint_dir is None:
            checkpoint_dir = (
                self.repository.database_path.parent
                / "checkpoints"
            )

        self.checkpoint_dir = Path(
            checkpoint_dir
        ).expanduser().resolve()

        self.checkpoint_path = (
            self.checkpoint_dir
            / f"{self.dataset_key}.json"
        )

        self.preprocessor = TextPreprocessor(
            dataset_key=self.dataset_key
        )

        self.source_metadata: Dict[str, Any] = {}

    def ingest(
        self,
        overwrite: bool = False,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """
        Import the dataset and return a validation summary.
        """
        if overwrite and resume:
            raise ValueError(
                "overwrite and resume cannot be enabled together."
            )

        self.repository.initialize()
        self._validate_source_files()

        self.source_metadata = self._load_source_metadata()
        development_sample = self._is_development_sample(
            self.source_metadata
        )

        if (
            development_sample
            and not self.allow_development_sample
        ):
            raise DatasetIngestionError(
                f"Dataset '{self.dataset_key}' is a reduced "
                "development sample. Re-run with "
                "--allow-development-sample only for testing, "
                "or prepare the complete dataset first."
            )

        existing_dataset = self.repository.get_dataset(
            self.dataset_key
        )

        if overwrite:
            print(
                f"Removing existing data for: "
                f"{self.dataset_key}"
            )

            self.repository.delete_dataset(
                self.dataset_key
            )

            self._delete_checkpoint()

            checkpoint = self._new_checkpoint()

        elif resume:
            if not self.checkpoint_path.is_file():
                raise DatasetIngestionError(
                    "Cannot resume because no checkpoint exists: "
                    f"{self.checkpoint_path}"
                )

            checkpoint = self._load_checkpoint()
            self._validate_checkpoint_sources(
                checkpoint
            )

        else:
            if existing_dataset is not None:
                raise DatasetIngestionError(
                    f"Dataset '{self.dataset_key}' already exists "
                    "in the document store. Use --resume to continue "
                    "an interrupted import or --overwrite to replace it."
                )

            if self.checkpoint_path.exists():
                raise DatasetIngestionError(
                    "A previous ingestion checkpoint exists. "
                    "Use --resume or --overwrite. "
                    f"Checkpoint: {self.checkpoint_path}"
                )

            checkpoint = self._new_checkpoint()

        self.repository.upsert_dataset(
            dataset_key=self.dataset_key,
            display_name=self.dataset_config["name"],
            ir_dataset_id=self.source_metadata.get(
                "ir_datasets_id",
                IR_DATASET_IDS[self.dataset_key],
            ),
            metadata={
                "source_metadata": self.source_metadata,
                "development_sample": development_sample,
                "ingestion_checkpoint_version": (
                    self.CHECKPOINT_VERSION
                ),
            },
        )

        self._save_checkpoint(checkpoint)

        if not checkpoint["stages"]["documents"]["complete"]:
            self._import_documents(checkpoint)

        if not checkpoint["stages"]["queries"]["complete"]:
            self._import_queries(checkpoint)

        if not checkpoint["stages"]["qrels"]["complete"]:
            self._import_qrels(checkpoint)

        counts = self.repository.update_dataset_counts(
            self.dataset_key,
            mark_imported=True,
        )

        validation = self._validate_import(counts)

        checkpoint["complete"] = True
        checkpoint["database_counts"] = counts
        checkpoint["validation"] = validation

        self._save_checkpoint(checkpoint)

        return {
            "dataset": self.dataset_key,
            "database_path": str(
                self.repository.database_path
            ),
            "checkpoint_path": str(
                self.checkpoint_path
            ),
            "development_sample": development_sample,
            "counts": counts,
            "validation": validation,
        }

    def _validate_source_files(self):
        required_paths = [
            self.documents_path,
            self.queries_path,
            self.qrels_path,
            self.metadata_path,
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

            raise FileNotFoundError(
                "Required prepared dataset files are missing:\n"
                f"{formatted}"
            )

        if self.dataset_config.get("format") != "jsonl":
            raise DatasetIngestionError(
                "This ingestion service currently expects "
                "JSONL document files. "
                f"Dataset: {self.dataset_key}"
            )

    def _load_source_metadata(self) -> Dict[str, Any]:
        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        if not isinstance(metadata, dict):
            raise DatasetIngestionError(
                "metadata.json must contain a JSON object."
            )

        metadata_dataset = metadata.get(
            "dataset_alias"
        )

        if (
            metadata_dataset
            and metadata_dataset != self.dataset_key
        ):
            raise DatasetIngestionError(
                "Dataset metadata does not match the requested "
                f"dataset. Requested '{self.dataset_key}', "
                f"metadata contains '{metadata_dataset}'."
            )

        return metadata

    @staticmethod
    def _is_development_sample(
        metadata: Dict[str, Any],
    ) -> bool:
        return bool(
            metadata.get("evaluation_sample")
            or metadata.get("document_limit") is not None
            or metadata.get("query_limit") is not None
        )

    @staticmethod
    def _source_descriptor(
        path: Path,
    ) -> Dict[str, Any]:
        stat = path.stat()

        return {
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _current_sources(self) -> Dict[str, Any]:
        return {
            "documents": self._source_descriptor(
                self.documents_path
            ),
            "queries": self._source_descriptor(
                self.queries_path
            ),
            "qrels": self._source_descriptor(
                self.qrels_path
            ),
            "metadata": self._source_descriptor(
                self.metadata_path
            ),
        }

    def _new_checkpoint(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "dataset": self.dataset_key,
            "complete": False,
            "sources": self._current_sources(),
            "stages": {
                "documents": {
                    "offset": 0,
                    "processed": 0,
                    "complete": False,
                },
                "queries": {
                    "offset": 0,
                    "processed": 0,
                    "complete": False,
                },
                "qrels": {
                    "offset": 0,
                    "processed": 0,
                    "complete": False,
                },
            },
        }

    def _load_checkpoint(self) -> Dict[str, Any]:
        with self.checkpoint_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            checkpoint = json.load(file)

        if checkpoint.get("version") != self.CHECKPOINT_VERSION:
            raise DatasetIngestionError(
                "Unsupported ingestion checkpoint version."
            )

        if checkpoint.get("dataset") != self.dataset_key:
            raise DatasetIngestionError(
                "Checkpoint belongs to a different dataset."
            )

        return checkpoint

    def _validate_checkpoint_sources(
        self,
        checkpoint: Dict[str, Any],
    ):
        saved_sources = checkpoint.get(
            "sources",
            {},
        )

        current_sources = self._current_sources()

        if saved_sources != current_sources:
            raise DatasetIngestionError(
                "Prepared dataset files changed after the "
                "checkpoint was created. Resume is unsafe. "
                "Use --overwrite to start a new import."
            )

    def _save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
    ):
        self.checkpoint_dir.mkdir(
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

    def _delete_checkpoint(self):
        self.checkpoint_path.unlink(
            missing_ok=True
        )

        temporary_path = (
            self.checkpoint_path.with_suffix(
                ".tmp"
            )
        )

        temporary_path.unlink(
            missing_ok=True
        )

    def _import_documents(
        self,
        checkpoint: Dict[str, Any],
    ):
        stage = checkpoint["stages"]["documents"]
        batch = []

        print()
        print("=" * 70)
        print(
            f"Importing documents for: "
            f"{self.dataset_key}"
        )
        print(
            f"Starting byte offset: "
            f"{stage['offset']:,}"
        )
        print("=" * 70)

        with self.documents_path.open(
            "rb"
        ) as file:
            file.seek(
                int(stage["offset"])
            )

            while True:
                line = file.readline()

                if not line:
                    break

                if not line.strip():
                    stage["offset"] = file.tell()
                    continue

                try:
                    raw = json.loads(
                        line.decode("utf-8")
                    )
                except Exception as error:
                    raise DatasetIngestionError(
                        "Invalid JSONL document near byte "
                        f"offset {stage['offset']}: {error}"
                    ) from error

                doc_id = str(
                    raw.get("doc_id", "")
                ).strip()

                if not doc_id:
                    raise DatasetIngestionError(
                        "A document is missing doc_id near "
                        f"byte offset {stage['offset']}."
                    )

                batch.append({
                    "doc_id": doc_id,
                    "title": str(
                        raw.get("title", "")
                        or ""
                    ),
                    "text": str(
                        raw.get("text", "")
                        or ""
                    ),
                    "metadata": (
                        raw.get("metadata")
                        if isinstance(
                            raw.get("metadata"),
                            dict,
                        )
                        else {}
                    ),
                })

                if len(batch) >= self.batch_size:
                    self.repository.bulk_upsert_documents(
                        self.dataset_key,
                        batch,
                    )

                    stage["processed"] += len(
                        batch
                    )

                    stage["offset"] = file.tell()

                    self._save_checkpoint(
                        checkpoint
                    )

                    print(
                        "Documents processed: "
                        f"{stage['processed']:,}",
                        flush=True,
                    )

                    batch.clear()

            if batch:
                self.repository.bulk_upsert_documents(
                    self.dataset_key,
                    batch,
                )

                stage["processed"] += len(batch)
                stage["offset"] = file.tell()

        stage["complete"] = True
        self._save_checkpoint(checkpoint)

        print(
            "Document import completed: "
            f"{stage['processed']:,}"
        )

    def _import_queries(
        self,
        checkpoint: Dict[str, Any],
    ):
        stage = checkpoint["stages"]["queries"]
        batch = []

        print()
        print("=" * 70)
        print(
            f"Importing queries for: "
            f"{self.dataset_key}"
        )
        print("=" * 70)

        with self.queries_path.open(
            "rb"
        ) as file:
            file.seek(
                int(stage["offset"])
            )

            while True:
                line = file.readline()

                if not line:
                    break

                decoded = line.decode(
                    "utf-8"
                ).rstrip("\r\n")

                if not decoded.strip():
                    stage["offset"] = file.tell()
                    continue

                parts = decoded.split(
                    "\t",
                    1,
                )

                if len(parts) != 2:
                    raise DatasetIngestionError(
                        "Invalid query TSV row near byte "
                        f"offset {stage['offset']}."
                    )

                query_id = parts[0].strip()
                raw_query = parts[1].strip()

                if not query_id or not raw_query:
                    raise DatasetIngestionError(
                        "A query row contains an empty ID or "
                        f"text near byte offset {stage['offset']}."
                    )

                batch.append({
                    "query_id": query_id,
                    "query": raw_query,
                    "processed_query": (
                        self.preprocessor.preprocess(
                            raw_query
                        )
                    ),
                })

                if len(batch) >= self.batch_size:
                    self.repository.bulk_upsert_queries(
                        self.dataset_key,
                        batch,
                    )

                    stage["processed"] += len(
                        batch
                    )

                    stage["offset"] = file.tell()

                    self._save_checkpoint(
                        checkpoint
                    )

                    print(
                        "Queries processed: "
                        f"{stage['processed']:,}",
                        flush=True,
                    )

                    batch.clear()

            if batch:
                self.repository.bulk_upsert_queries(
                    self.dataset_key,
                    batch,
                )

                stage["processed"] += len(batch)
                stage["offset"] = file.tell()

        stage["complete"] = True
        self._save_checkpoint(checkpoint)

        print(
            "Query import completed: "
            f"{stage['processed']:,}"
        )

    def _import_qrels(
        self,
        checkpoint: Dict[str, Any],
    ):
        stage = checkpoint["stages"]["qrels"]
        batch = []

        print()
        print("=" * 70)
        print(
            f"Importing qrels for: "
            f"{self.dataset_key}"
        )
        print("=" * 70)

        with self.qrels_path.open(
            "rb"
        ) as file:
            file.seek(
                int(stage["offset"])
            )

            while True:
                line = file.readline()

                if not line:
                    break

                decoded = line.decode(
                    "utf-8"
                ).rstrip("\r\n")

                if not decoded.strip():
                    stage["offset"] = file.tell()
                    continue

                parts = decoded.split("\t")

                if len(parts) == 3:
                    query_id = parts[0]
                    iteration = "0"
                    doc_id = parts[1]
                    relevance_text = parts[2]

                elif len(parts) >= 4:
                    query_id = parts[0]
                    iteration = parts[1]
                    doc_id = parts[2]
                    relevance_text = parts[3]

                else:
                    raise DatasetIngestionError(
                        "Invalid qrels TSV row near byte "
                        f"offset {stage['offset']}."
                    )

                query_id = query_id.strip()
                doc_id = doc_id.strip()
                iteration = iteration.strip() or "0"

                if not query_id or not doc_id:
                    raise DatasetIngestionError(
                        "A qrel row contains an empty query ID "
                        f"or document ID near offset "
                        f"{stage['offset']}."
                    )

                try:
                    relevance = int(
                        relevance_text
                    )
                except ValueError as error:
                    raise DatasetIngestionError(
                        "Invalid qrel relevance value near "
                        f"byte offset {stage['offset']}."
                    ) from error

                batch.append({
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "relevance": relevance,
                    "iteration": iteration,
                })

                if len(batch) >= self.batch_size:
                    self.repository.bulk_upsert_qrels(
                        self.dataset_key,
                        batch,
                    )

                    stage["processed"] += len(
                        batch
                    )

                    stage["offset"] = file.tell()

                    self._save_checkpoint(
                        checkpoint
                    )

                    print(
                        "Qrels processed: "
                        f"{stage['processed']:,}",
                        flush=True,
                    )

                    batch.clear()

            if batch:
                self.repository.bulk_upsert_qrels(
                    self.dataset_key,
                    batch,
                )

                stage["processed"] += len(batch)
                stage["offset"] = file.tell()

        stage["complete"] = True
        self._save_checkpoint(checkpoint)

        print(
            "Qrel import completed: "
            f"{stage['processed']:,}"
        )

    def _validate_import(
        self,
        counts: Dict[str, int],
    ) -> Dict[str, Any]:
        expected_documents = (
            self.source_metadata.get(
                "unique_document_ids"
            )
            or self.source_metadata.get(
                "documents_exported"
            )
        )

        expected_queries = (
            self.source_metadata.get(
                "queries_exported"
            )
        )

        expected_qrels = (
            self.source_metadata.get(
                "qrels_exported"
            )
        )

        comparisons = {
            "documents": {
                "expected": expected_documents,
                "actual": counts["documents"],
            },
            "queries": {
                "expected": expected_queries,
                "actual": counts["queries"],
            },
            "qrels": {
                "expected": expected_qrels,
                "actual": counts["qrels"],
            },
        }

        mismatches = []

        for name, comparison in comparisons.items():
            expected = comparison["expected"]
            actual = comparison["actual"]

            if (
                expected is not None
                and int(expected) != int(actual)
            ):
                mismatches.append(
                    f"{name}: expected {expected}, "
                    f"stored {actual}"
                )

        if mismatches:
            raise DatasetIngestionError(
                "Database counts do not match metadata:\n"
                + "\n".join(
                    f"- {item}"
                    for item in mismatches
                )
            )

        missing_documents = (
            self.repository.find_missing_qrel_documents(
                self.dataset_key,
                limit=100,
            )
        )

        missing_queries = (
            self.repository.find_missing_qrel_queries(
                self.dataset_key,
                limit=100,
            )
        )

        if missing_documents:
            raise DatasetIngestionError(
                "Some qrels reference documents that are "
                "missing from the document store. Examples: "
                f"{missing_documents[:10]}"
            )

        if missing_queries:
            raise DatasetIngestionError(
                "Some qrels reference queries that are "
                "missing from the document store. Examples: "
                f"{missing_queries[:10]}"
            )

        return {
            "metadata_counts_match": True,
            "missing_qrel_documents": 0,
            "missing_qrel_queries": 0,
            "comparisons": comparisons,
        }