import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class DocumentClusteringError(RuntimeError):
    """Raised when document clustering cannot be built or loaded."""


class DocumentClusteringService:
    """
    Build and load document clusters from the saved embedding memmap.

    This feature is intentionally independent from the core retrieval
    pipeline. It uses the already-built SentenceTransformer embeddings
    stored under indexes/<dataset>/embedding/_build and saves small
    clustering artifacts under artifacts/clustering.
    """

    SUPPORTED_DATASETS = {"quora", "clinical_trials"}

    DEFAULT_CLUSTERS = {
        "quora": 30,
        "clinical_trials": 20,
    }

    def __init__(
        self,
        dataset_key: str,
        project_root: str | Path,
        indexes_root: str | Path | None = None,
        artifacts_root: str | Path | None = None,
        reports_root: str | Path | None = None,
    ):
        self.dataset_key = str(dataset_key).strip()
        self.project_root = Path(project_root).expanduser().resolve()
        self.indexes_root = (
            Path(indexes_root).expanduser().resolve()
            if indexes_root is not None
            else self.project_root / "indexes"
        )
        self.artifacts_root = (
            Path(artifacts_root).expanduser().resolve()
            if artifacts_root is not None
            else self.project_root / "artifacts" / "clustering"
        )
        self.reports_root = (
            Path(reports_root).expanduser().resolve()
            if reports_root is not None
            else self.project_root / "reports" / "clustering"
        )

        self._validate_dataset()

        self.embedding_dir = (
            self.indexes_root / self.dataset_key / "embedding"
        )
        self.embedding_build_dir = self.embedding_dir / "_build"
        self.embedding_manifest_path = self.embedding_dir / "manifest.json"
        self.embedding_memmap_path = (
            self.embedding_build_dir / "embeddings.float32.memmap"
        )
        self.doc_ids_path = self.embedding_dir / "doc_ids.joblib"

        self.output_dir = self.artifacts_root / self.dataset_key
        self.cluster_manifest_path = self.output_dir / "manifest.json"
        self.model_path = self.output_dir / "kmeans.joblib"
        self.assignments_path = self.output_dir / "cluster_assignments.joblib"
        self.summary_report_path = (
            self.reports_root / f"{self.dataset_key}_cluster_summary.csv"
        )

    def _validate_dataset(self):
        if self.dataset_key not in self.SUPPORTED_DATASETS:
            raise ValueError(
                "Unsupported dataset_key. Expected one of: "
                f"{sorted(self.SUPPORTED_DATASETS)}"
            )

    def _load_embedding_manifest(self) -> Dict[str, Any]:
        if not self.embedding_manifest_path.is_file():
            raise DocumentClusteringError(
                "Embedding manifest not found. Build the embedding "
                f"index first: {self.embedding_manifest_path}"
            )

        with self.embedding_manifest_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_doc_ids(self) -> List[str]:
        if not self.doc_ids_path.is_file():
            raise DocumentClusteringError(
                f"doc_ids.joblib not found: {self.doc_ids_path}"
            )

        doc_ids = joblib.load(self.doc_ids_path)
        return [str(doc_id) for doc_id in doc_ids]

    def _open_embedding_memmap(
        self,
        document_count: int,
        embedding_dimension: int,
    ) -> np.memmap:
        if not self.embedding_memmap_path.is_file():
            raise DocumentClusteringError(
                "Embedding memmap not found. Do not delete the _build "
                "folder if clustering will be used. Missing file: "
                f"{self.embedding_memmap_path}"
            )

        return np.memmap(
            self.embedding_memmap_path,
            dtype="float32",
            mode="r",
            shape=(int(document_count), int(embedding_dimension)),
        )

    @staticmethod
    def _iter_slices(total: int, batch_size: int):
        start = 0
        while start < total:
            end = min(start + batch_size, total)
            yield start, end
            start = end

    def build(
        self,
        n_clusters: Optional[int] = None,
        train_sample_size: int = 100_000,
        train_batch_size: int = 4096,
        predict_batch_size: int = 8192,
        random_seed: int = 13,
        max_iter: int = 100,
        n_init: int = 3,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Train MiniBatchKMeans and assign every document to a cluster."""
        if n_clusters is None:
            n_clusters = self.DEFAULT_CLUSTERS[self.dataset_key]

        n_clusters = int(n_clusters)
        train_sample_size = int(train_sample_size)
        train_batch_size = int(train_batch_size)
        predict_batch_size = int(predict_batch_size)
        random_seed = int(random_seed)
        max_iter = int(max_iter)
        n_init = int(n_init)

        self._validate_build_parameters(
            n_clusters=n_clusters,
            train_sample_size=train_sample_size,
            train_batch_size=train_batch_size,
            predict_batch_size=predict_batch_size,
            max_iter=max_iter,
            n_init=n_init,
        )

        if self.cluster_manifest_path.is_file() and not overwrite:
            raise DocumentClusteringError(
                "A clustering artifact already exists. Use --overwrite "
                "to rebuild it."
            )

        manifest = self._load_embedding_manifest()
        document_count = int(manifest["document_count"])
        embedding_dimension = int(manifest["embedding_dimension"])

        doc_ids = self._load_doc_ids()
        if len(doc_ids) != document_count:
            raise DocumentClusteringError(
                "doc_ids length does not match embedding document_count."
            )

        embeddings = self._open_embedding_memmap(
            document_count=document_count,
            embedding_dimension=embedding_dimension,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)

        sample_size = min(train_sample_size, document_count)
        rng = np.random.default_rng(random_seed)
        sample_indices = np.sort(
            rng.choice(
                document_count,
                size=sample_size,
                replace=False,
            )
        )

        print(
            "Training MiniBatchKMeans with "
            f"{sample_size:,} sampled vectors, "
            f"n_clusters={n_clusters}."
        )

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            batch_size=train_batch_size,
            max_iter=max_iter,
            n_init=n_init,
            reassignment_ratio=0.01,
            verbose=0,
        )

        training_vectors = np.asarray(
            embeddings[sample_indices],
            dtype="float32",
        )
        kmeans.fit(training_vectors)

        assignments = np.empty(document_count, dtype=np.int32)
        cluster_counts = np.zeros(n_clusters, dtype=np.int64)
        best_distance = np.full(n_clusters, np.inf, dtype=np.float64)
        representative_indices = np.full(n_clusters, -1, dtype=np.int64)

        print("Assigning all documents to clusters...")

        for start, end in self._iter_slices(document_count, predict_batch_size):
            batch = np.asarray(embeddings[start:end], dtype="float32")
            predicted = kmeans.predict(batch).astype(np.int32, copy=False)
            assignments[start:end] = predicted

            for cluster_id in predicted:
                cluster_counts[int(cluster_id)] += 1

            centers = kmeans.cluster_centers_[predicted]
            distances = np.sum((batch - centers) ** 2, axis=1)

            for offset, cluster_id in enumerate(predicted):
                cluster_id = int(cluster_id)
                distance = float(distances[offset])

                if distance < best_distance[cluster_id]:
                    best_distance[cluster_id] = distance
                    representative_indices[cluster_id] = start + offset

            if end == document_count or end % (predict_batch_size * 10) == 0:
                print(f"Assigned {end:,}/{document_count:,} documents.")

        joblib.dump(kmeans, self.model_path)
        joblib.dump(assignments, self.assignments_path)

        summary_rows = self._write_summary_report(
            doc_ids=doc_ids,
            cluster_counts=cluster_counts,
            representative_indices=representative_indices,
            document_count=document_count,
        )

        clustering_manifest = {
            "feature": "document_clustering",
            "dataset": self.dataset_key,
            "algorithm": "MiniBatchKMeans",
            "embedding_source": str(self.embedding_dir),
            "document_count": document_count,
            "embedding_dimension": embedding_dimension,
            "n_clusters": n_clusters,
            "train_sample_size": sample_size,
            "train_batch_size": train_batch_size,
            "predict_batch_size": predict_batch_size,
            "random_seed": random_seed,
            "max_iter": max_iter,
            "n_init": n_init,
            "files": {
                "model": self.model_path.name,
                "assignments": self.assignments_path.name,
                "summary_report": str(self.summary_report_path),
            },
        }

        self.cluster_manifest_path.write_text(
            json.dumps(clustering_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "dataset": self.dataset_key,
            "document_count": document_count,
            "embedding_dimension": embedding_dimension,
            "n_clusters": n_clusters,
            "artifact_dir": str(self.output_dir),
            "summary_report": str(self.summary_report_path),
            "model_path": str(self.model_path),
            "assignments_path": str(self.assignments_path),
            "non_empty_clusters": int(np.count_nonzero(cluster_counts)),
            "largest_cluster_size": int(cluster_counts.max()),
            "smallest_cluster_size": int(cluster_counts.min()),
            "summary_preview": summary_rows[:5],
        }

    def _validate_build_parameters(
        self,
        n_clusters: int,
        train_sample_size: int,
        train_batch_size: int,
        predict_batch_size: int,
        max_iter: int,
        n_init: int,
    ):
        if n_clusters <= 1:
            raise ValueError("n_clusters must be greater than one.")
        if train_sample_size <= 0:
            raise ValueError("train_sample_size must be greater than zero.")
        if train_batch_size <= 0:
            raise ValueError("train_batch_size must be greater than zero.")
        if predict_batch_size <= 0:
            raise ValueError("predict_batch_size must be greater than zero.")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than zero.")
        if n_init <= 0:
            raise ValueError("n_init must be greater than zero.")

    def _write_summary_report(
        self,
        doc_ids: List[str],
        cluster_counts: np.ndarray,
        representative_indices: np.ndarray,
        document_count: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for cluster_id, count in enumerate(cluster_counts.tolist()):
            representative_index = int(representative_indices[cluster_id])
            representative_doc_id = (
                doc_ids[representative_index]
                if representative_index >= 0
                else ""
            )

            rows.append(
                {
                    "dataset": self.dataset_key,
                    "cluster_id": cluster_id,
                    "document_count": int(count),
                    "percentage": round(
                        (float(count) / max(document_count, 1)) * 100.0,
                        4,
                    ),
                    "representative_doc_id": representative_doc_id,
                }
            )

        with self.summary_report_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "dataset",
                    "cluster_id",
                    "document_count",
                    "percentage",
                    "representative_doc_id",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return rows

    def load_summary(self) -> List[Dict[str, Any]]:
        if not self.summary_report_path.is_file():
            raise DocumentClusteringError(
                f"Cluster summary does not exist: {self.summary_report_path}"
            )

        with self.summary_report_path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)

    def get_document_cluster(self, doc_position: int) -> int:
        if not self.assignments_path.is_file():
            raise DocumentClusteringError(
                f"Cluster assignments do not exist: {self.assignments_path}"
            )

        assignments = joblib.load(self.assignments_path)
        return int(assignments[int(doc_position)])
