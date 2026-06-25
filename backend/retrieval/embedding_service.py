import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import joblib
import numpy as np
import torch
from django.conf import settings
from sentence_transformers import SentenceTransformer

from datasets.dataset_loader import DatasetLoader
from document_store.repository import DocumentStoreRepository


class EmbeddingIndexError(RuntimeError):
    """
    Raised when a saved embedding index is invalid or incomplete.
    """


class EmbeddingRetrievalService:
    """
    Dataset-aware dense retrieval service using SentenceTransformer + FAISS.

    Supported modes:

    1. Saved FAISS index
       Used by the complete Quora and Clinical Trials collections.
       The service loads only the FAISS index, doc_ids, manifest, and the
       local SentenceTransformer model.

    2. In-memory development index
       Used by sample_dataset and tests when no saved FAISS index exists.
    """

    SAVED_INDEX_REQUIRED_DATASETS = {
        "quora",
        "clinical_trials",
    }

    DEFAULT_MODEL_DIRECTORY_NAME = "multi-qa-MiniLM-L6-cos-v1"

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        use_saved_index: bool = True,
        device: str = "auto",
        model_path: Optional[str | Path] = None,
        index_name: str = "embedding",
        index_dir: Optional[str | Path] = None,
    ):
        self.dataset_key = str(dataset_key).strip()
        self.use_saved_index = bool(use_saved_index)
        self.device = self._resolve_device(device)
        self.index_name = str(index_name).strip()

        if not self.dataset_key:
            raise ValueError("dataset_key cannot be empty.")

        if not self.index_name:
            raise ValueError("index_name cannot be empty.")

        indexes_root = Path(
            getattr(
                settings,
                "INDEXES_DIR",
                settings.BASE_DIR.parent / "indexes",
            )
        ).expanduser().resolve()

        self.index_dir = (
            Path(index_dir).expanduser().resolve()
            if index_dir is not None
            else indexes_root / self.dataset_key / self.index_name
        )
        self.manifest_path = self.index_dir / "manifest.json"
        self.faiss_index_path = self.index_dir / "faiss.index"
        self.doc_ids_path = self.index_dir / "doc_ids.joblib"
        self.embedding_config_path = self.index_dir / "embedding_config.json"

        artifacts_root = Path(
            getattr(
                settings,
                "ARTIFACTS_DIR",
                settings.BASE_DIR.parent / "artifacts",
            )
        ).expanduser().resolve()

        self.default_model_path = (
            artifacts_root
            / "models"
            / self.DEFAULT_MODEL_DIRECTORY_NAME
        )

        self.requested_model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else None
        )

        self.mode: Optional[str] = None
        self.manifest: Dict[str, Any] = {}
        self.embedding_config: Dict[str, Any] = {}
        self.index: Optional[Any] = None
        self.doc_ids: List[str] = []
        self.document_count = 0
        self.embedding_dimension = 0
        self.model_path: Optional[Path] = None
        self.model: Optional[SentenceTransformer] = None

        # In-memory development attributes.
        self.documents: List[Dict[str, Any]] = []

        self._document_repository: Optional[DocumentStoreRepository] = None

        self._initialize_index()

    def _resolve_device(self, requested_device: str) -> str:
        normalized = str(requested_device).strip().lower()

        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if normalized in {"cuda", "cuda:0"}:
            if not torch.cuda.is_available():
                raise EmbeddingIndexError(
                    "CUDA was requested, but torch.cuda.is_available() "
                    "returned False."
                )
            return "cuda"

        if normalized == "cpu":
            return "cpu"

        raise ValueError("device must be one of: auto, cpu, cuda.")

    def _initialize_index(self):
        if self.use_saved_index and self.saved_index_exists():
            self.load_saved_index()
            return

        if self.dataset_key in self.SAVED_INDEX_REQUIRED_DATASETS:
            raise FileNotFoundError(
                "The complete dataset requires a saved embedding FAISS "
                f"index, but no manifest was found: {self.manifest_path}"
            )

        self.load_documents()
        self.build_in_memory_index()

    def saved_index_exists(self) -> bool:
        return all(
            path.is_file()
            for path in [
                self.manifest_path,
                self.faiss_index_path,
                self.doc_ids_path,
                self.embedding_config_path,
            ]
        )

    def load_saved_index(self):
        with self.manifest_path.open("r", encoding="utf-8") as file:
            self.manifest = json.load(file)

        with self.embedding_config_path.open("r", encoding="utf-8") as file:
            self.embedding_config = json.load(file)

        self._validate_manifest()

        self.index = faiss.read_index(str(self.faiss_index_path))

        nprobe = self.manifest.get("nprobe")
        if nprobe is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = int(nprobe)

        self.doc_ids = joblib.load(self.doc_ids_path)

        if not isinstance(self.doc_ids, list):
            raise EmbeddingIndexError(
                "Invalid embedding doc_ids artifact: expected a list."
            )

        self.document_count = int(self.manifest["document_count"])
        self.embedding_dimension = int(self.manifest["embedding_dimension"])

        if int(self.index.ntotal) != self.document_count:
            raise EmbeddingIndexError(
                "FAISS index vector count does not match the manifest. "
                f"FAISS: {self.index.ntotal}, manifest: {self.document_count}."
            )

        if len(self.doc_ids) != self.document_count:
            raise EmbeddingIndexError(
                "Embedding doc_ids count does not match the manifest. "
                f"doc_ids: {len(self.doc_ids)}, manifest: {self.document_count}."
            )

        self.model_path = self._resolve_model_path_from_manifest()
        self.model = self._load_model(self.model_path)
        self.mode = "faiss"

    def _validate_manifest(self):
        required_keys = [
            "index_type",
            "faiss_index_type",
            "dataset",
            "document_count",
            "embedding_dimension",
            "model_path",
            "vectors_are_normalized",
            "similarity",
            "faiss_ntotal",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in self.manifest
        ]

        if missing_keys:
            raise EmbeddingIndexError(
                "Embedding manifest is missing keys: "
                + ", ".join(missing_keys)
            )

        if self.manifest["dataset"] != self.dataset_key:
            raise EmbeddingIndexError(
                "Embedding manifest belongs to a different dataset. "
                f"Expected {self.dataset_key}, got {self.manifest['dataset']}."
            )

        if self.manifest["index_type"] != "sentence_transformer_faiss":
            raise EmbeddingIndexError(
                "Unsupported embedding index_type: "
                f"{self.manifest['index_type']}"
            )

        if not bool(self.manifest.get("vectors_are_normalized")):
            raise EmbeddingIndexError(
                "The embedding service expects normalized document vectors."
            )

        if self.manifest.get("similarity") != "cosine_via_inner_product":
            raise EmbeddingIndexError(
                "Unsupported embedding similarity mode: "
                f"{self.manifest.get('similarity')}"
            )

    def _resolve_model_path_from_manifest(self) -> Path:
        if self.requested_model_path is not None:
            candidate = self.requested_model_path
        else:
            candidate = Path(str(self.manifest.get("model_path", ""))).expanduser()

            if not candidate.is_absolute():
                candidate = (settings.BASE_DIR.parent / candidate).resolve()

            candidate = candidate.resolve()

            if not candidate.exists() and self.default_model_path.exists():
                candidate = self.default_model_path

        if not candidate.exists():
            raise FileNotFoundError(
                f"Embedding model path does not exist: {candidate}"
            )

        return candidate

    def _load_model(self, model_path: Path) -> SentenceTransformer:
        model = SentenceTransformer(
            str(model_path),
            device=self.device,
            local_files_only=True,
        )

        dimension = self._get_model_embedding_dimension(model)

        if self.embedding_dimension and dimension != self.embedding_dimension:
            raise EmbeddingIndexError(
                "Embedding model dimension does not match the index. "
                f"Model: {dimension}, index: {self.embedding_dimension}."
            )

        return model

    @staticmethod
    def _get_model_embedding_dimension(model: SentenceTransformer) -> int:
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())

        return int(model.get_sentence_embedding_dimension())

    def _get_document_repository(self) -> DocumentStoreRepository:
        if self._document_repository is None:
            database_path = getattr(settings, "CORPUS_DATABASE_PATH", None)

            if database_path is None:
                raise EmbeddingIndexError(
                    "CORPUS_DATABASE_PATH is not configured."
                )

            self._document_repository = DocumentStoreRepository(database_path)
            self._document_repository.initialize()

        return self._document_repository

    def _encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int,
    ) -> np.ndarray:
        if self.model is None:
            raise EmbeddingIndexError("Embedding model is not loaded.")

        cleaned_queries = [str(query).strip() for query in queries]

        encode_kwargs = {
            "batch_size": int(batch_size),
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }

        if hasattr(self.model, "encode_query"):
            embeddings = self.model.encode_query(
                cleaned_queries,
                **encode_kwargs,
            )
        else:
            embeddings = self.model.encode(
                cleaned_queries,
                **encode_kwargs,
            )

        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.embedding_dimension:
            raise EmbeddingIndexError(
                "Query embedding dimension does not match the index. "
                f"Query: {embeddings.shape[1]}, index: {self.embedding_dimension}."
            )

        return np.ascontiguousarray(embeddings, dtype="float32")

    @staticmethod
    def _sanitize_top_k(top_k: int, document_count: int) -> int:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return min(int(top_k), int(document_count))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        query = query.strip()

        if not query:
            return []

        result_count = self._sanitize_top_k(top_k, self.document_count)

        if self.mode == "faiss":
            query_embedding = self._encode_queries(
                [query],
                batch_size=1,
            )
            distances, indices = self._search_faiss_vectors(
                query_embedding,
                top_k=result_count,
            )
            ranked_documents = self._format_faiss_row(
                distances[0],
                indices[0],
                result_count,
            )
            return self._hydrate_faiss_results(ranked_documents)

        if self.mode == "in_memory":
            return self._search_in_memory(query, result_count)

        raise RuntimeError("Embedding service has no initialized index mode.")

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        query_batch_size: int = 64,
        hydrate: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        if not isinstance(queries, list):
            raise ValueError("queries must be a list of strings.")

        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be greater than zero.")

        normalized_queries: List[str] = []

        for query_index, query in enumerate(queries):
            if not isinstance(query, str):
                raise ValueError(
                    "Every query must be a string. "
                    f"Invalid query position: {query_index}."
                )
            normalized_queries.append(query.strip())

        if not normalized_queries:
            return []

        result_count = self._sanitize_top_k(top_k, self.document_count)

        if self.mode != "faiss":
            return [
                self.search(query=query, top_k=result_count) if query else []
                for query in normalized_queries
            ]

        all_results: List[List[Dict[str, Any]]] = []

        for batch_start in range(0, len(normalized_queries), query_batch_size):
            batch_queries = normalized_queries[
                batch_start:batch_start + query_batch_size
            ]

            non_empty_positions: List[int] = []
            non_empty_queries: List[str] = []
            batch_results: List[List[Dict[str, Any]]] = [
                [] for _ in batch_queries
            ]

            for position, query in enumerate(batch_queries):
                if query:
                    non_empty_positions.append(position)
                    non_empty_queries.append(query)

            if non_empty_queries:
                query_embeddings = self._encode_queries(
                    non_empty_queries,
                    batch_size=min(query_batch_size, len(non_empty_queries)),
                )
                distances, indices = self._search_faiss_vectors(
                    query_embeddings,
                    top_k=result_count,
                )

                for row_number, original_position in enumerate(non_empty_positions):
                    ranked_documents = self._format_faiss_row(
                        distances[row_number],
                        indices[row_number],
                        result_count,
                    )

                    if hydrate:
                        batch_results[original_position] = (
                            self._hydrate_faiss_results(ranked_documents)
                        )
                    else:
                        batch_results[original_position] = [
                            {
                                "rank": rank,
                                "doc_id": doc_id,
                                "score": round(float(score), 6),
                            }
                            for rank, (doc_id, score) in enumerate(
                                ranked_documents,
                                start=1,
                            )
                        ]

            all_results.extend(batch_results)

        return all_results

    def _search_faiss_vectors(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise EmbeddingIndexError("FAISS index is not loaded.")

        distances, indices = self.index.search(query_embeddings, int(top_k))
        return distances, indices

    def _format_faiss_row(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        ranked_documents: List[Tuple[str, float]] = []

        for score, doc_index in zip(distances, indices):
            doc_index = int(doc_index)

            if doc_index < 0:
                continue

            if doc_index >= len(self.doc_ids):
                raise EmbeddingIndexError(
                    "FAISS returned an out-of-range document index: "
                    f"{doc_index}."
                )

            ranked_documents.append(
                (
                    self.doc_ids[doc_index],
                    float(score),
                )
            )

            if len(ranked_documents) >= top_k:
                break

        return ranked_documents

    def _hydrate_faiss_results(
        self,
        ranked_documents: List[Tuple[str, float]],
    ) -> List[Dict[str, Any]]:
        if not ranked_documents:
            return []

        ordered_doc_ids = [doc_id for doc_id, score in ranked_documents]
        repository = self._get_document_repository()

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
            raise EmbeddingIndexError(
                "Embedding search returned document IDs missing "
                "from the document store. "
                f"Examples: {missing_doc_ids[:10]}"
            )

        results: List[Dict[str, Any]] = []

        for rank, (doc_id, score) in enumerate(ranked_documents, start=1):
            document = documents_by_id[doc_id]
            raw_text = str(document.get("raw_text", "") or "")

            results.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "title": str(document.get("title", "") or ""),
                    "snippet": raw_text[:250],
                    "score": round(float(score), 6),
                }
            )

        return results

    def load_documents(self):
        self.documents = DatasetLoader.load_documents(self.dataset_key)

        if not self.documents:
            raise ValueError(
                f"No documents found for dataset '{self.dataset_key}'."
            )

    def build_in_memory_index(self):
        self.model_path = (
            self.requested_model_path
            if self.requested_model_path is not None
            else self.default_model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "No saved embedding index exists and the local model "
                f"was not found: {self.model_path}"
            )

        self.model = self._load_model(self.model_path)
        self.embedding_dimension = self._get_model_embedding_dimension(self.model)

        texts = [
            self._document_to_text(document)
            for document in self.documents
        ]

        vectors = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32", copy=False)

        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))

        self.doc_ids = [
            str(document.get("doc_id", ""))
            for document in self.documents
        ]
        self.document_count = len(self.documents)
        self.mode = "in_memory"

    @staticmethod
    def _document_to_text(document: Dict[str, Any]) -> str:
        title = str(document.get("title", "") or "").strip()
        text = str(
            document.get("text", document.get("raw_text", "")) or ""
        ).strip()

        if title and text:
            return f"{title}. {text}"

        return title or text

    def _search_in_memory(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        query_embedding = self._encode_queries([query], batch_size=1)
        distances, indices = self._search_faiss_vectors(
            query_embedding,
            top_k=top_k,
        )

        results: List[Dict[str, Any]] = []

        for rank, (score, document_index) in enumerate(
            zip(distances[0], indices[0]),
            start=1,
        ):
            document_index = int(document_index)

            if document_index < 0:
                continue

            document = self.documents[document_index]
            text = str(document.get("text", "") or "")

            results.append(
                {
                    "rank": rank,
                    "doc_id": str(document.get("doc_id", "")),
                    "title": str(document.get("title", "") or ""),
                    "snippet": text[:250],
                    "score": round(float(score), 6),
                }
            )

        return results

    def get_index_information(self) -> Dict[str, Any]:
        information: Dict[str, Any] = {
            "dataset": self.dataset_key,
            "mode": self.mode,
            "document_count": self.document_count,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "saved_index_used": self.use_saved_index and self.saved_index_exists(),
            "index_directory": str(self.index_dir),
            "model_path": str(self.model_path) if self.model_path else None,
        }

        if self.mode == "faiss":
            information.update(
                {
                    "index_type": self.manifest.get("index_type"),
                    "faiss_index_type": self.manifest.get("faiss_index_type"),
                    "similarity": self.manifest.get("similarity"),
                    "vectors_are_normalized": self.manifest.get(
                        "vectors_are_normalized"
                    ),
                    "text_max_characters": self.manifest.get(
                        "text_max_characters"
                    ),
                    "faiss_ntotal": int(self.index.ntotal)
                    if self.index is not None
                    else None,
                    "nlist": self.manifest.get("nlist"),
                    "nprobe": getattr(self.index, "nprobe", None)
                    if self.index is not None
                    else None,
                    "pq_m": self.manifest.get("pq_m"),
                    "pq_nbits": self.manifest.get("pq_nbits"),
                    "index_size_mb": round(
                        self.faiss_index_path.stat().st_size / (1024 * 1024),
                        2,
                    ),
                    "doc_ids_size_mb": round(
                        self.doc_ids_path.stat().st_size / (1024 * 1024),
                        2,
                    ),
                }
            )

        return information
