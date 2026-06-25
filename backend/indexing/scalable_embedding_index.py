import gc
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from document_store.repository import DocumentStoreRepository


class ScalableEmbeddingBuildError(RuntimeError):
    """
    Raised when the scalable embedding index cannot be built or validated.
    """


class ScalableEmbeddingIndexBuilder:
    """
    Build a disk-backed SentenceTransformer + FAISS vector index from the
    SQLite document store.

    The build is intentionally split into two resumable stages:

    1. Embedding stage
       - Stream documents from SQLite.
       - Encode them on CPU or GPU in small batches.
       - Write float32 normalized vectors to a NumPy memmap on disk.
       - Save doc_ids separately so all documents keep stable positions.

    2. FAISS stage
       - Train an approximate compressed FAISS index from a sample.
       - Add the memmapped vectors in batches.
       - Save the completed FAISS index and a manifest.

    Generated artifacts:

        indexes/<dataset>/embedding/
        ├── manifest.json
        ├── faiss.index
        ├── doc_ids.joblib
        ├── embedding_config.json
        └── _build/
            ├── checkpoint.json
            ├── embeddings.float32.memmap
            └── faiss.index.tmp
    """

    CHECKPOINT_VERSION = 1
    INDEX_VERSION = 1

    SUPPORTED_DATASETS = {
        "quora",
        "clinical_trials",
    }

    DEFAULT_TEXT_MAX_CHARACTERS = {
        "quora": 512,
        "clinical_trials": 2000,
    }

    DEFAULT_NLIST = {
        "quora": 2048,
        "clinical_trials": 1024,
    }

    DEFAULT_TRAIN_SAMPLE_SIZE = {
        "quora": 100_000,
        "clinical_trials": 75_000,
    }

    def __init__(
        self,
        repository: DocumentStoreRepository,
        dataset_key: str,
        indexes_root: str | Path,
        model_path: str | Path,
        batch_size: int = 128,
        add_batch_size: int = 8192,
        device: str = "auto",
        text_max_characters: Optional[int] = None,
        index_type: str = "ivfpq",
        nlist: Optional[int] = None,
        pq_m: int = 48,
        pq_nbits: int = 8,
        nprobe: int = 32,
        train_sample_size: Optional[int] = None,
        random_seed: int = 13,
    ):
        self.repository = repository
        self.dataset_key = str(dataset_key).strip()
        self.indexes_root = Path(indexes_root).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()

        self.batch_size = int(batch_size)
        self.add_batch_size = int(add_batch_size)
        self.device = self._resolve_device(device)
        self.index_type = str(index_type).strip().lower()

        self.text_max_characters = (
            int(text_max_characters)
            if text_max_characters is not None
            else self.DEFAULT_TEXT_MAX_CHARACTERS.get(
                self.dataset_key,
                1000,
            )
        )

        self.nlist = (
            int(nlist)
            if nlist is not None
            else self.DEFAULT_NLIST.get(
                self.dataset_key,
                1024,
            )
        )

        self.pq_m = int(pq_m)
        self.pq_nbits = int(pq_nbits)
        self.nprobe = int(nprobe)
        self.train_sample_size = (
            int(train_sample_size)
            if train_sample_size is not None
            else self.DEFAULT_TRAIN_SAMPLE_SIZE.get(
                self.dataset_key,
                50_000,
            )
        )
        self.random_seed = int(random_seed)

        self._validate_parameters()

        self.index_dir = (
            self.indexes_root
            / self.dataset_key
            / "embedding"
        )
        self.work_dir = self.index_dir / "_build"

        self.checkpoint_path = self.work_dir / "checkpoint.json"
        self.memmap_path = self.work_dir / "embeddings.float32.memmap"
        self.temporary_faiss_path = self.work_dir / "faiss.index.tmp"

        self.faiss_index_path = self.index_dir / "faiss.index"
        self.doc_ids_path = self.index_dir / "doc_ids.joblib"
        self.embedding_config_path = self.index_dir / "embedding_config.json"
        self.manifest_path = self.index_dir / "manifest.json"

        self.dataset_metadata: Optional[Dict[str, Any]] = None
        self.document_count = 0
        self.embedding_dimension: Optional[int] = None

    def _resolve_device(self, requested_device: str) -> str:
        normalized = str(requested_device).strip().lower()

        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if normalized in {"cuda", "cuda:0"}:
            if not torch.cuda.is_available():
                raise ScalableEmbeddingBuildError(
                    "CUDA was requested, but torch.cuda.is_available() "
                    "returned False."
                )
            return "cuda"

        if normalized == "cpu":
            return "cpu"

        raise ValueError(
            "device must be one of: auto, cpu, cuda."
        )

    def _validate_parameters(self):
        if not self.dataset_key:
            raise ValueError(
                "dataset_key cannot be empty."
            )

        if self.dataset_key not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{self.dataset_key}'."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Embedding model path does not exist: {self.model_path}"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if self.add_batch_size <= 0:
            raise ValueError(
                "add_batch_size must be greater than zero."
            )

        if self.text_max_characters <= 0:
            raise ValueError(
                "text_max_characters must be greater than zero."
            )

        if self.index_type not in {"ivfpq", "hnsw", "flat"}:
            raise ValueError(
                "index_type must be one of: ivfpq, hnsw, flat."
            )

        if self.nlist <= 0:
            raise ValueError(
                "nlist must be greater than zero."
            )

        if self.pq_m <= 0:
            raise ValueError(
                "pq_m must be greater than zero."
            )

        if self.pq_nbits <= 0:
            raise ValueError(
                "pq_nbits must be greater than zero."
            )

        if self.nprobe <= 0:
            raise ValueError(
                "nprobe must be greater than zero."
            )

        if self.train_sample_size <= 0:
            raise ValueError(
                "train_sample_size must be greater than zero."
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

        model = self._load_model()
        self.embedding_dimension = int(
            self._get_model_embedding_dimension(model)
        )

        if (
            self.index_type == "ivfpq"
            and self.embedding_dimension % self.pq_m != 0
        ):
            raise ScalableEmbeddingBuildError(
                "For IVF-PQ, embedding_dimension must be divisible "
                f"by pq_m. Got dimension={self.embedding_dimension}, "
                f"pq_m={self.pq_m}."
            )

        checkpoint = self._prepare_build(
            overwrite=overwrite,
            resume=resume,
        )

        if checkpoint["stage"] == "embedding":
            self._build_embeddings(
                checkpoint=checkpoint,
                model=model,
            )

        # Explicitly release model memory before FAISS training/addition.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if checkpoint["stage"] == "faiss":
            self._build_faiss_index(checkpoint)

        if checkpoint["stage"] != "complete":
            raise ScalableEmbeddingBuildError(
                "Embedding build did not reach the complete stage."
            )

        return self.validate_index()

    def _load_dataset_information(self):
        dataset = self.repository.get_dataset(
            self.dataset_key
        )

        if dataset is None:
            raise ScalableEmbeddingBuildError(
                f"Dataset '{self.dataset_key}' is not "
                "present in the document store."
            )

        counts = self.repository.get_dataset_counts(
            self.dataset_key
        )

        self.dataset_metadata = dataset
        self.document_count = int(counts["documents"])

        if self.document_count <= 0:
            raise ScalableEmbeddingBuildError(
                f"Dataset '{self.dataset_key}' contains no documents."
            )

    def _load_model(self) -> SentenceTransformer:
        print(
            f"Loading embedding model from {self.model_path} "
            f"on {self.device}..."
        )
        model = SentenceTransformer(
            str(self.model_path),
            device=self.device,
            local_files_only=True,
        )
        return model

    @staticmethod
    def _get_model_embedding_dimension(
        model: SentenceTransformer,
    ) -> int:
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())

        return int(model.get_sentence_embedding_dimension())

    def _configuration(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "batch_size": self.batch_size,
            "add_batch_size": self.add_batch_size,
            "text_max_characters": self.text_max_characters,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "pq_m": self.pq_m,
            "pq_nbits": self.pq_nbits,
            "nprobe": self.nprobe,
            "train_sample_size": self.train_sample_size,
            "random_seed": self.random_seed,
        }

    def _new_checkpoint(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "dataset": self.dataset_key,
            "document_count": self.document_count,
            "embedding_dimension": self.embedding_dimension,
            "configuration": self._configuration(),
            "stage": "embedding",
            "embedding": {
                "last_rowid": 0,
                "processed_documents": 0,
            },
            "faiss": {
                "trained": False,
                "added_vectors": 0,
            },
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
                raise ScalableEmbeddingBuildError(
                    "Cannot resume because no embedding checkpoint "
                    f"exists: {self.checkpoint_path}"
                )

            checkpoint = self._load_checkpoint()
            self._validate_checkpoint(checkpoint)
            return checkpoint

        if self.manifest_path.is_file():
            raise ScalableEmbeddingBuildError(
                "A completed embedding index already exists. "
                "Use --overwrite to rebuild it."
            )

        if self.checkpoint_path.is_file():
            raise ScalableEmbeddingBuildError(
                "An incomplete embedding build exists. "
                "Use --resume or --overwrite."
            )

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

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
        if checkpoint.get("version") != self.CHECKPOINT_VERSION:
            raise ScalableEmbeddingBuildError(
                "Unsupported embedding checkpoint version."
            )

        if checkpoint.get("dataset") != self.dataset_key:
            raise ScalableEmbeddingBuildError(
                "The embedding checkpoint belongs to another dataset."
            )

        if int(checkpoint.get("document_count", -1)) != self.document_count:
            raise ScalableEmbeddingBuildError(
                "The document-store count changed after "
                "the embedding build started."
            )

        if int(checkpoint.get("embedding_dimension", -1)) != int(
            self.embedding_dimension
        ):
            raise ScalableEmbeddingBuildError(
                "The embedding dimension changed after the build started."
            )

        if checkpoint.get("configuration") != self._configuration():
            raise ScalableEmbeddingBuildError(
                "Embedding build parameters differ from the saved "
                "checkpoint. Use the original parameters or restart "
                "with --overwrite."
            )

        if (
            checkpoint.get("stage") in {"embedding", "faiss"}
            and not self.memmap_path.is_file()
            and int(
                checkpoint.get("embedding", {}).get(
                    "processed_documents",
                    0,
                )
            ) > 0
        ):
            raise ScalableEmbeddingBuildError(
                "The memmap file required for resume is missing. "
                "Restart with --overwrite."
            )

    def _save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
    ):
        """
        Save checkpoints robustly on Windows.

        A fixed checkpoint.tmp file plus very frequent os-level replace calls can
        occasionally fail on Windows when antivirus, indexing, or another process
        briefly touches the destination file.  Use a unique temporary file and
        retry os.replace before failing.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(
            checkpoint,
            ensure_ascii=False,
            indent=2,
        )

        temporary_path = (
            self.work_dir
            / f"checkpoint.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )

        temporary_path.write_text(
            payload,
            encoding="utf-8",
        )

        last_error: Optional[BaseException] = None

        for attempt in range(12):
            try:
                os.replace(
                    str(temporary_path),
                    str(self.checkpoint_path),
                )
                return
            except PermissionError as error:
                last_error = error
                time.sleep(0.25 * (attempt + 1))
            except OSError as error:
                last_error = error
                time.sleep(0.25 * (attempt + 1))

        try:
            if temporary_path.exists():
                temporary_path.unlink()
        except OSError:
            pass

        raise ScalableEmbeddingBuildError(
            "Could not save embedding checkpoint after several retries. "
            "Close editors, Explorer preview panes, antivirus scans, or any "
            "process touching the index directory, then rerun with --resume. "
            f"Last error: {last_error}"
        )

    def _open_memmap(
        self,
        mode: str,
    ) -> np.memmap:
        return np.memmap(
            self.memmap_path,
            dtype="float32",
            mode=mode,
            shape=(
                self.document_count,
                int(self.embedding_dimension),
            ),
        )

    def _iter_document_rows(
        self,
        after_rowid: int,
    ) -> Iterable[Sequence[Any]]:
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
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break
                yield rows

    def _document_to_text(
        self,
        row: Any,
    ) -> str:
        title = (row["title"] or "").strip()
        raw_text = (row["raw_text"] or "").strip()

        if title and raw_text:
            if raw_text.startswith(title):
                text = raw_text
            else:
                text = f"{title}. {raw_text}"
        else:
            text = title or raw_text

        # Keep whitespace stable and prevent very long clinical records from
        # dominating encoding time. Transformer max_seq_length is still the
        # final token-level cap.
        text = " ".join(text.split())

        if len(text) > self.text_max_characters:
            text = text[: self.text_max_characters]

        return text

    def _build_embeddings(
        self,
        checkpoint: Dict[str, Any],
        model: SentenceTransformer,
    ):
        if not self.memmap_path.is_file():
            embeddings = self._open_memmap("w+")
            embeddings.flush()
            del embeddings

        embeddings = self._open_memmap("r+")

        processed = int(
            checkpoint["embedding"]["processed_documents"]
        )
        last_rowid = int(
            checkpoint["embedding"]["last_rowid"]
        )
        print(
            "Embedding stage: "
            f"starting at document {processed:,}/{self.document_count:,} "
            f"after rowid {last_rowid}."
        )

        start_time = time.perf_counter()
        last_checkpoint_time = start_time
        checkpoint_every_seconds = 5.0

        for rows in self._iter_document_rows(after_rowid=last_rowid):
            texts = [self._document_to_text(row) for row in rows]
            rowids = [int(row["store_rowid"]) for row in rows]

            batch_embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32", copy=False)

            batch_size = int(batch_embeddings.shape[0])

            start = processed
            end = processed + batch_size

            if end > self.document_count:
                raise ScalableEmbeddingBuildError(
                    "Encoded more documents than expected."
                )

            embeddings[start:end, :] = batch_embeddings
            embeddings.flush()

            processed = end
            last_rowid = max(rowids)

            checkpoint["embedding"]["processed_documents"] = processed
            checkpoint["embedding"]["last_rowid"] = last_rowid

            now = time.perf_counter()
            if (
                now - last_checkpoint_time >= checkpoint_every_seconds
                or processed == self.document_count
            ):
                self._save_checkpoint(checkpoint)
                last_checkpoint_time = now

            elapsed = max(
                time.perf_counter() - start_time,
                1e-9,
            )
            docs_per_second = processed / elapsed

            print(
                f"Embedded {processed:,}/{self.document_count:,} "
                f"documents ({docs_per_second:.2f} docs/s)."
            )

        embeddings.flush()
        del embeddings

        if processed != self.document_count:
            raise ScalableEmbeddingBuildError(
                "Embedding stage ended before all documents were processed."
            )

        checkpoint["stage"] = "faiss"
        self._save_checkpoint(checkpoint)

        print("Embedding stage complete.")

    def _build_faiss_index(
        self,
        checkpoint: Dict[str, Any],
    ):
        print("FAISS stage: loading memmap...")
        embeddings = self._open_memmap("r")

        index = self._create_faiss_index()

        if not index.is_trained:
            training_vectors = self._sample_training_vectors(
                embeddings
            )
            print(
                "Training FAISS index with "
                f"{training_vectors.shape[0]:,} vectors..."
            )
            index.train(training_vectors)
            del training_vectors
            gc.collect()

        added = int(
            checkpoint.get("faiss", {}).get(
                "added_vectors",
                0,
            )
        )

        # Resume always restarts FAISS addition from zero because the partial
        # FAISS index is not persisted after every add batch. Embeddings are the
        # expensive stage; rebuilding FAISS from memmap is usually fast and safe.
        if added > 0:
            print(
                "Resuming during FAISS stage. Rebuilding FAISS "
                "addition from vector 0 for safety."
            )
            added = 0
            checkpoint["faiss"]["added_vectors"] = 0
            self._save_checkpoint(checkpoint)

        total = self.document_count
        start_time = time.perf_counter()
        last_checkpoint_time = start_time
        checkpoint_every_seconds = 5.0

        while added < total:
            end = min(added + self.add_batch_size, total)
            batch = np.asarray(
                embeddings[added:end],
                dtype="float32",
            )
            index.add(batch)
            added = end

            checkpoint["faiss"]["added_vectors"] = added

            now = time.perf_counter()
            if (
                now - last_checkpoint_time >= checkpoint_every_seconds
                or added == total
            ):
                self._save_checkpoint(checkpoint)
                last_checkpoint_time = now

            elapsed = max(
                time.perf_counter() - start_time,
                1e-9,
            )
            vectors_per_second = added / elapsed
            print(
                f"Added {added:,}/{total:,} vectors to FAISS "
                f"({vectors_per_second:.2f} vectors/s)."
            )

        if index.ntotal != self.document_count:
            raise ScalableEmbeddingBuildError(
                "FAISS index vector count does not match document count."
            )

        if hasattr(index, "nprobe"):
            index.nprobe = self.nprobe

        print(f"Saving FAISS index to {self.faiss_index_path}...")
        faiss.write_index(index, str(self.faiss_index_path))

        doc_ids = self._load_all_doc_ids()
        if len(doc_ids) != self.document_count:
            raise ScalableEmbeddingBuildError(
                "doc_ids count does not match document count."
            )

        joblib.dump(doc_ids, self.doc_ids_path)
        self._write_embedding_config()
        self._write_manifest(index)

        checkpoint["stage"] = "complete"
        self._save_checkpoint(checkpoint)

        del embeddings
        del index
        gc.collect()

        print("FAISS stage complete.")

    def _create_faiss_index(self):
        dimension = int(self.embedding_dimension)

        if self.index_type == "flat":
            return faiss.IndexFlatIP(dimension)

        if self.index_type == "hnsw":
            # HNSW keeps full vectors in RAM. It is more accurate than PQ, but
            # less suitable for low-memory machines.
            index = faiss.IndexHNSWFlat(
                dimension,
                32,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
            return index

        if self.index_type == "ivfpq":
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(
                quantizer,
                dimension,
                self.nlist,
                self.pq_m,
                self.pq_nbits,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.nprobe = self.nprobe
            return index

        raise ScalableEmbeddingBuildError(
            f"Unsupported FAISS index type: {self.index_type}"
        )

    def _sample_training_vectors(
        self,
        embeddings: np.memmap,
    ) -> np.ndarray:
        sample_size = min(
            self.train_sample_size,
            self.document_count,
        )

        if sample_size <= self.nlist and self.index_type == "ivfpq":
            raise ScalableEmbeddingBuildError(
                "train_sample_size must be greater than nlist for IVF-PQ."
            )

        rng = np.random.default_rng(self.random_seed)
        indices = np.sort(
            rng.choice(
                self.document_count,
                size=sample_size,
                replace=False,
            )
        )
        sample = np.asarray(
            embeddings[indices],
            dtype="float32",
        )
        return sample


    def _load_all_doc_ids(self) -> List[str]:
        doc_ids: List[str] = []

        with self.repository.connection() as connection:
            cursor = connection.execute(
                """
                SELECT doc_id
                FROM documents
                WHERE dataset_key = ?
                ORDER BY rowid
                """,
                (self.dataset_key,),
            )

            while True:
                rows = cursor.fetchmany(10000)
                if not rows:
                    break
                doc_ids.extend(str(row["doc_id"]) for row in rows)

        return doc_ids

    def _write_embedding_config(self):
        payload = {
            "index_version": self.INDEX_VERSION,
            "dataset": self.dataset_key,
            "document_count": self.document_count,
            "embedding_dimension": self.embedding_dimension,
            "configuration": self._configuration(),
            "text_strategy": "title_plus_raw_text_prefix",
            "vectors_are_normalized": True,
            "similarity": "cosine_via_inner_product",
        }
        self.embedding_config_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_manifest(self, index: Any):
        payload = {
            "index_version": self.INDEX_VERSION,
            "index_type": "sentence_transformer_faiss",
            "faiss_index_type": self.index_type,
            "dataset": self.dataset_key,
            "dataset_metadata": self.dataset_metadata,
            "document_count": self.document_count,
            "embedding_dimension": self.embedding_dimension,
            "model_path": str(self.model_path),
            "device_used_for_build": self.device,
            "text_max_characters": self.text_max_characters,
            "vectors_are_normalized": True,
            "similarity": "cosine_via_inner_product",
            "faiss_ntotal": int(index.ntotal),
            "nlist": self.nlist if self.index_type == "ivfpq" else None,
            "pq_m": self.pq_m if self.index_type == "ivfpq" else None,
            "pq_nbits": self.pq_nbits if self.index_type == "ivfpq" else None,
            "nprobe": self.nprobe if self.index_type == "ivfpq" else None,
            "train_sample_size": self.train_sample_size,
            "created_at_unix": time.time(),
            "files": {
                "faiss_index": str(self.faiss_index_path.name),
                "doc_ids": str(self.doc_ids_path.name),
                "embedding_config": str(self.embedding_config_path.name),
            },
        }
        self.manifest_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def validate_index(self) -> Dict[str, Any]:
        required_paths = [
            self.faiss_index_path,
            self.doc_ids_path,
            self.embedding_config_path,
            self.manifest_path,
        ]

        missing_paths = [
            str(path)
            for path in required_paths
            if not path.is_file()
        ]

        if missing_paths:
            raise ScalableEmbeddingBuildError(
                "Missing embedding artifacts: "
                + ", ".join(missing_paths)
            )

        index = faiss.read_index(
            str(self.faiss_index_path)
        )
        doc_ids = joblib.load(self.doc_ids_path)

        manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8",
            )
        )

        if int(index.ntotal) != self.document_count:
            raise ScalableEmbeddingBuildError(
                "FAISS index ntotal does not match document count."
            )

        if len(doc_ids) != self.document_count:
            raise ScalableEmbeddingBuildError(
                "doc_ids length does not match document count."
            )

        return {
            "dataset": self.dataset_key,
            "index_dir": str(self.index_dir),
            "index_type": manifest.get("index_type"),
            "faiss_index_type": manifest.get("faiss_index_type"),
            "document_count": self.document_count,
            "embedding_dimension": manifest.get("embedding_dimension"),
            "faiss_ntotal": int(index.ntotal),
            "model_path": manifest.get("model_path"),
            "text_max_characters": manifest.get("text_max_characters"),
            "index_size_mb": round(
                self.faiss_index_path.stat().st_size / (1024 * 1024),
                2,
            ),
            "doc_ids_size_mb": round(
                self.doc_ids_path.stat().st_size / (1024 * 1024),
                2,
            ),
        }
