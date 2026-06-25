import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import joblib

from preprocessing.preprocessing_service import TextPreprocessor


class TopicDetectionError(RuntimeError):
    """Raised when cluster topic detection cannot run."""


class ClusterTopicDetectionService:
    """
    Build simple topic labels for previously-built document clusters.

    This service reads:
    - artifacts/clustering/<dataset>/cluster_assignments.joblib
    - indexes/<dataset>/embedding/doc_ids.joblib
    - artifacts/database/corpus.sqlite3

    It then samples documents from every cluster, extracts frequent
    dataset-aware preprocessed terms, removes generic/domain boilerplate
    terms, and writes a small CSV report:

    reports/topics/<dataset>_cluster_topics.csv

    The feature is intentionally independent from core retrieval so it
    can be demonstrated as an additional IR feature.
    """

    SUPPORTED_DATASETS = {"quora", "clinical_trials"}

    def __init__(
        self,
        dataset_key: str,
        project_root: str | Path,
        artifacts_root: str | Path | None = None,
        indexes_root: str | Path | None = None,
        reports_root: str | Path | None = None,
        database_path: str | Path | None = None,
    ):
        self.dataset_key = str(dataset_key).strip()

        if self.dataset_key not in self.SUPPORTED_DATASETS:
            raise ValueError(
                "Unsupported dataset_key. Expected one of: "
                f"{sorted(self.SUPPORTED_DATASETS)}"
            )

        self.project_root = Path(project_root).expanduser().resolve()

        self.artifacts_root = (
            Path(artifacts_root).expanduser().resolve()
            if artifacts_root is not None
            else self.project_root / "artifacts"
        )

        self.indexes_root = (
            Path(indexes_root).expanduser().resolve()
            if indexes_root is not None
            else self.project_root / "indexes"
        )

        self.reports_root = (
            Path(reports_root).expanduser().resolve()
            if reports_root is not None
            else self.project_root / "reports" / "topics"
        )

        self.database_path = (
            Path(database_path).expanduser().resolve()
            if database_path is not None
            else self.project_root
            / "artifacts"
            / "database"
            / "corpus.sqlite3"
        )

        self.clustering_dir = (
            self.artifacts_root / "clustering" / self.dataset_key
        )
        self.cluster_manifest_path = self.clustering_dir / "manifest.json"
        self.assignments_path = (
            self.clustering_dir / "cluster_assignments.joblib"
        )

        self.embedding_dir = (
            self.indexes_root / self.dataset_key / "embedding"
        )
        self.doc_ids_path = self.embedding_dir / "doc_ids.joblib"

        self.output_dir = self.artifacts_root / "topics" / self.dataset_key
        self.topic_manifest_path = self.output_dir / "manifest.json"
        self.topic_report_path = (
            self.reports_root
            / f"{self.dataset_key}_cluster_topics.csv"
        )

        self.preprocessor = TextPreprocessor(
            dataset_key=self.dataset_key,
        )

        self.domain_stopwords = self._get_domain_stopwords()

    def build(
        self,
        top_terms: int = 10,
        sample_docs_per_cluster: int = 500,
        fetch_batch_size: int = 500,
        min_term_length: int = 3,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        top_terms = int(top_terms)
        sample_docs_per_cluster = int(sample_docs_per_cluster)
        fetch_batch_size = int(fetch_batch_size)
        min_term_length = int(min_term_length)

        self._validate_build_parameters(
            top_terms=top_terms,
            sample_docs_per_cluster=sample_docs_per_cluster,
            fetch_batch_size=fetch_batch_size,
            min_term_length=min_term_length,
        )

        if self.topic_report_path.is_file() and not overwrite:
            raise TopicDetectionError(
                "Topic report already exists. Use --overwrite to rebuild it."
            )

        self._validate_required_files()

        assignments = joblib.load(self.assignments_path)
        doc_ids = [
            str(doc_id)
            for doc_id in joblib.load(self.doc_ids_path)
        ]

        if len(assignments) != len(doc_ids):
            raise TopicDetectionError(
                "Cluster assignments length does not match doc_ids length."
            )

        cluster_manifest = self._load_cluster_manifest()
        n_clusters = int(cluster_manifest["n_clusters"])

        cluster_doc_ids = self._sample_doc_ids_by_cluster(
            assignments=assignments,
            doc_ids=doc_ids,
            n_clusters=n_clusters,
            sample_docs_per_cluster=sample_docs_per_cluster,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []

        print(
            f"Building topic labels for {n_clusters} clusters "
            f"from dataset '{self.dataset_key}'."
        )

        for cluster_id in range(n_clusters):
            sampled_ids = cluster_doc_ids.get(cluster_id, [])

            documents = self._get_documents_by_ids(
                doc_ids=sampled_ids,
                fetch_batch_size=fetch_batch_size,
            )

            term_counter = self._count_cluster_terms(
                documents=documents,
                min_term_length=min_term_length,
            )

            selected_terms = [
                term
                for term, _ in term_counter.most_common(top_terms)
            ]

            topic_label = ", ".join(selected_terms[:5])

            rows.append(
                {
                    "dataset": self.dataset_key,
                    "cluster_id": cluster_id,
                    "document_count": int(
                        self._count_cluster_documents(
                            assignments=assignments,
                            cluster_id=cluster_id,
                        )
                    ),
                    "sample_document_count": len(documents),
                    "top_terms": " ".join(selected_terms),
                    "topic_label": topic_label,
                    "representative_doc_ids": " ".join(sampled_ids[:5]),
                }
            )

            print(
                f"Cluster {cluster_id}: "
                f"{len(documents):,} sampled docs, "
                f"topic='{topic_label}'"
            )

        self._write_topic_report(rows)

        manifest = {
            "feature": "topic_detection",
            "dataset": self.dataset_key,
            "source_feature": "document_clustering",
            "cluster_manifest": str(self.cluster_manifest_path),
            "document_count": len(doc_ids),
            "n_clusters": n_clusters,
            "top_terms": top_terms,
            "sample_docs_per_cluster": sample_docs_per_cluster,
            "fetch_batch_size": fetch_batch_size,
            "min_term_length": min_term_length,
            "domain_stopwords_count": len(self.domain_stopwords),
            "report": str(self.topic_report_path),
        }

        self.topic_manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "dataset": self.dataset_key,
            "document_count": len(doc_ids),
            "n_clusters": n_clusters,
            "topic_report": str(self.topic_report_path),
            "topic_manifest": str(self.topic_manifest_path),
            "preview": rows[:5],
        }

    def _validate_build_parameters(
        self,
        top_terms: int,
        sample_docs_per_cluster: int,
        fetch_batch_size: int,
        min_term_length: int,
    ):
        if top_terms <= 0:
            raise ValueError(
                "top_terms must be greater than zero."
            )

        if sample_docs_per_cluster <= 0:
            raise ValueError(
                "sample_docs_per_cluster must be greater than zero."
            )

        if fetch_batch_size <= 0:
            raise ValueError(
                "fetch_batch_size must be greater than zero."
            )

        if min_term_length <= 0:
            raise ValueError(
                "min_term_length must be greater than zero."
            )

    def _validate_required_files(self):
        missing_paths = [
            path
            for path in [
                self.cluster_manifest_path,
                self.assignments_path,
                self.doc_ids_path,
                self.database_path,
            ]
            if not path.is_file()
        ]

        if missing_paths:
            joined = "\n".join(str(path) for path in missing_paths)

            raise TopicDetectionError(
                "Topic detection cannot run because required files "
                f"are missing:\n{joined}"
            )

    def _load_cluster_manifest(self) -> Dict[str, Any]:
        with self.cluster_manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _count_cluster_documents(
        assignments: Sequence[int],
        cluster_id: int,
    ) -> int:
        return sum(
            1
            for assignment in assignments
            if int(assignment) == int(cluster_id)
        )

    def _sample_doc_ids_by_cluster(
        self,
        assignments: Sequence[int],
        doc_ids: List[str],
        n_clusters: int,
        sample_docs_per_cluster: int,
    ) -> Dict[int, List[str]]:
        cluster_doc_ids: Dict[int, List[str]] = {
            cluster_id: []
            for cluster_id in range(n_clusters)
        }

        for index, cluster_id in enumerate(assignments):
            cluster_id = int(cluster_id)

            if len(cluster_doc_ids[cluster_id]) >= sample_docs_per_cluster:
                continue

            cluster_doc_ids[cluster_id].append(doc_ids[index])

            if all(
                len(values) >= sample_docs_per_cluster
                for values in cluster_doc_ids.values()
            ):
                break

        return cluster_doc_ids

    def _get_documents_by_ids(
        self,
        doc_ids: List[str],
        fetch_batch_size: int,
    ) -> List[Dict[str, str]]:
        if not doc_ids:
            return []

        documents_by_id: Dict[str, Dict[str, str]] = {}

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row

            for batch in self._batched(doc_ids, fetch_batch_size):
                placeholders = ",".join("?" for _ in batch)

                rows = connection.execute(
                    f"""
                    SELECT doc_id, title, raw_text
                    FROM documents
                    WHERE dataset_key = ?
                    AND doc_id IN ({placeholders})
                    """,
                    [self.dataset_key, *batch],
                ).fetchall()

                for row in rows:
                    documents_by_id[str(row["doc_id"])] = {
                        "doc_id": str(row["doc_id"]),
                        "title": str(row["title"] or ""),
                        "raw_text": str(row["raw_text"] or ""),
                    }

        return [
            documents_by_id[doc_id]
            for doc_id in doc_ids
            if doc_id in documents_by_id
        ]

    @staticmethod
    def _batched(
        items: List[str],
        batch_size: int,
    ) -> Iterable[List[str]]:
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    def _count_cluster_terms(
        self,
        documents: List[Dict[str, str]],
        min_term_length: int,
    ) -> Counter:
        term_counter = Counter()

        for document in documents:
            text = " ".join(
                [
                    document.get("title", ""),
                    document.get("raw_text", ""),
                ]
            ).strip()

            if not text:
                continue

            tokens = self.preprocessor.preprocess_tokens(text)

            # Document-frequency style counting reduces domination by
            # repeated boilerplate terms inside a single long document.
            unique_terms = {
                token
                for token in tokens
                if self._is_useful_topic_term(
                    token=token,
                    min_term_length=min_term_length,
                )
            }

            for token in unique_terms:
                term_counter[token] += 1

        return term_counter

    def _is_useful_topic_term(
        self,
        token: str,
        min_term_length: int,
    ) -> bool:
        if not token:
            return False

        token = str(token).strip().lower()

        if len(token) < min_term_length:
            return False

        if token in self.domain_stopwords:
            return False

        if token.isdigit():
            return False

        # Remove terms that are mostly numeric, such as 1000mg or 18-65.
        digit_count = sum(
            character.isdigit()
            for character in token
        )

        if digit_count and digit_count >= len(token) / 2:
            return False

        return True

    def _get_domain_stopwords(self) -> Set[str]:
        """
        Extra stopwords for topic labeling only.

        These are not used in retrieval. They only remove generic words
        that dominate cluster topic labels and make the reports less useful.
        """
        clinical_trials_stopwords = {
            "able",
            "abnormal",
            "adult",
            "adults",
            "age",
            "aged",
            "also",
            "arm",
            "arms",
            "assess",
            "assessment",
            "available",
            "based",
            "care",
            "clinical",
            "clinically",
            "condition",
            "conditions",
            "consent",
            "control",
            "controlled",
            "current",
            "currently",
            "criteria",
            "data",
            "day",
            "days",
            "disease",
            "diseases",
            "dose",
            "doses",
            "drug",
            "eligible",
            "eligibility",
            "efficacy",
            "evaluate",
            "evaluation",
            "evidence",
            "excluded",
            "exclusion",
            "female",
            "follow",
            "group",
            "groups",
            "health",
            "history",
            "including",
            "inclusion",
            "intervention",
            "known",
            "male",
            "measure",
            "measures",
            "medical",
            "month",
            "months",
            "not",
            "objective",
            "objectives",
            "outcome",
            "outcomes",
            "participant",
            "participants",
            "patient",
            "patients",
            "phase",
            "placebo",
            "previous",
            "previously",
            "primary",
            "prior",
            "procedure",
            "receive",
            "received",
            "receiving",
            "recruiting",
            "related",
            "response",
            "safety",
            "secondary",
            "serious",
            "significant",
            "status",
            "study",
            "studies",
            "subject",
            "subjects",
            "therapy",
            "therapies",
            "treatment",
            "treatments",
            "trial",
            "trials",
            "use",
            "used",
            "using",
            "visit",
            "weeks",
            "willing",
            "with",
            "without",
            "written",
            "year",
            "years",
        }

        quora_stopwords = {
            "best",
            "better",
            "compar",
            "compare",
            "difference",
            "differ",
            "doe",
            "does",
            "did",
            "good",
            "happen",
            "how",
            "like",
            "make",
            "mean",
            "not",
            "people",
            "peopl",
            "question",
            "quora",
            "thing",
            "things",
            "use",
            "way",
            "ways",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "work",
        }

        if self.dataset_key == "clinical_trials":
            return clinical_trials_stopwords

        if self.dataset_key == "quora":
            return quora_stopwords

        return set()

    def _write_topic_report(
        self,
        rows: List[Dict[str, Any]],
    ):
        fieldnames = [
            "dataset",
            "cluster_id",
            "document_count",
            "sample_document_count",
            "top_terms",
            "topic_label",
            "representative_doc_ids",
        ]

        with self.topic_report_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    def load_topics(self) -> List[Dict[str, Any]]:
        if not self.topic_report_path.is_file():
            raise TopicDetectionError(
                f"Topic report does not exist: {self.topic_report_path}"
            )

        with self.topic_report_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)
            return list(reader)