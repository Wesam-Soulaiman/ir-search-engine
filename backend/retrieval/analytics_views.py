import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from datasets.dataset_registry import get_dataset_config
from document_store.repository import DocumentStoreError
from retrieval.evaluation_csv_service import (
    EvaluationCsvError,
    list_evaluation_csv_files,
    load_evaluation_dashboard,
)
from retrieval.result_enrichment import (
    document_store_is_required,
    get_document_store_repository,
)


def _get_reports_root() -> Path:
    return Path(
        getattr(
            settings,
            "REPORTS_DIR",
            settings.BASE_DIR.parent / "reports",
        )
    ).expanduser().resolve()


def _get_artifacts_root() -> Path:
    return Path(
        getattr(
            settings,
            "ARTIFACTS_DIR",
            settings.BASE_DIR.parent / "artifacts",
        )
    ).expanduser().resolve()


def _validate_dataset(dataset_key: str) -> str:
    dataset_key = str(dataset_key or "").strip().lower()

    if not dataset_key:
        raise ValueError("dataset_key is required.")

    get_dataset_config(dataset_key)
    return dataset_key


def _coerce_value(value: str) -> Any:
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return ""

    try:
        if "." in value:
            return float(value)

        return int(value)
    except ValueError:
        return value


def _coerce_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _read_csv_report(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Report file was not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [
            {key: _coerce_value(value) for key, value in row.items()}
            for row in reader
        ]


def _read_optional_json(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _split_document_ids(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    return [
        item.strip()
        for item in re.split(r"[\s,;|]+", str(value))
        if item.strip()
    ]


def _parse_top_terms(
    value: Any,
    max_terms: int = 10,
) -> List[Dict[str, Any]]:
    if value is None:
        return []

    parsed_terms: List[Dict[str, Any]] = []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                term = str(item.get("term", "")).strip()
                weight = _coerce_number(item.get("weight"), default=0.0)
            else:
                term = str(item).strip()
                weight = 0.0

            if term:
                parsed_terms.append({
                    "term": term,
                    "weight": weight,
                })

    else:
        text = str(value or "").strip()

        if text.startswith("["):
            try:
                return _parse_top_terms(
                    json.loads(text),
                    max_terms=max_terms,
                )
            except json.JSONDecodeError:
                pass

        raw_terms = [
            item.strip()
            for item in re.split(r"[\s,;|]+", text)
            if item.strip()
        ]

        for raw_term in raw_terms:
            term = raw_term
            weight = 0.0

            if ":" in raw_term:
                maybe_term, maybe_weight = raw_term.rsplit(":", 1)
                numeric_weight = _coerce_number(
                    maybe_weight,
                    default=-1.0,
                )

                if numeric_weight >= 0:
                    term = maybe_term
                    weight = numeric_weight

            if term:
                parsed_terms.append({
                    "term": term,
                    "weight": weight,
                })

    parsed_terms = parsed_terms[:max_terms]
    has_numeric_weights = any(
        term["weight"] > 0
        for term in parsed_terms
    )

    if parsed_terms and not has_numeric_weights:
        total_terms = len(parsed_terms)

        for index, term in enumerate(parsed_terms):
            term["weight"] = round(
                (total_terms - index) / total_terms,
                4,
            )

    return parsed_terms


def _top_terms_to_label(
    top_terms: Iterable[Dict[str, Any]],
    fallback: str,
) -> str:
    label_terms = [
        str(term.get("term") or "").strip()
        for term in top_terms
        if str(term.get("term") or "").strip()
    ][:5]

    if label_terms:
        return " ".join(label_terms)

    return fallback


def _load_example_documents(
    dataset_key: str,
    doc_ids: Iterable[str],
) -> List[Dict[str, str]]:
    unique_doc_ids = list(
        dict.fromkeys(
            str(doc_id).strip()
            for doc_id in doc_ids
            if str(doc_id).strip()
        )
    )

    if not unique_doc_ids:
        return []

    documents_by_id: Dict[str, Dict[str, Any]] = {}

    if document_store_is_required(dataset_key):
        try:
            repository = get_document_store_repository()
            documents = repository.get_documents(
                dataset_key=dataset_key,
                doc_ids=unique_doc_ids,
            )
            documents_by_id = {
                str(document["doc_id"]): document
                for document in documents
            }
        except (DocumentStoreError, FileNotFoundError, RuntimeError):
            documents_by_id = {}

    examples = []

    for doc_id in unique_doc_ids[:5]:
        document = documents_by_id.get(doc_id, {})
        title = str(document.get("title") or "").strip()

        examples.append({
            "doc_id": doc_id,
            "title": title or f"Document {doc_id}",
        })

    return examples


def _topic_row_by_cluster_id(
    topic_rows: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result = {}

    for row in topic_rows:
        cluster_id = str(row.get("cluster_id", "")).strip()

        if cluster_id:
            result[cluster_id] = row

    return result


def _build_cluster_payload(dataset_key: str) -> Dict[str, Any]:
    reports_root = _get_reports_root()
    artifacts_root = _get_artifacts_root()
    cluster_report_path = (
        reports_root
        / "clustering"
        / f"{dataset_key}_cluster_summary.csv"
    )
    topic_report_path = (
        reports_root
        / "topics"
        / f"{dataset_key}_cluster_topics.csv"
    )
    manifest_path = (
        artifacts_root
        / "clustering"
        / dataset_key
        / "manifest.json"
    )

    cluster_rows = _read_csv_report(cluster_report_path)

    try:
        topic_rows = _read_csv_report(topic_report_path)
    except FileNotFoundError:
        topic_rows = []

    topics_by_cluster_id = _topic_row_by_cluster_id(topic_rows)
    clusters = []

    for cluster_row in cluster_rows:
        cluster_id = str(cluster_row.get("cluster_id", "")).strip()
        topic_row = topics_by_cluster_id.get(cluster_id, {})
        top_terms = _parse_top_terms(topic_row.get("top_terms"))
        label = (
            str(topic_row.get("topic_label") or "").strip()
            or _top_terms_to_label(
                top_terms,
                fallback=f"Cluster {cluster_id}",
            )
        )
        representative_ids = [
            *(_split_document_ids(topic_row.get("representative_doc_ids"))),
            *(_split_document_ids(cluster_row.get("representative_doc_id"))),
        ]

        clusters.append({
            "cluster_id": int(cluster_id) if cluster_id.isdigit() else cluster_id,
            "label": label,
            "size": int(_coerce_number(cluster_row.get("document_count"))),
            "percentage": _coerce_number(cluster_row.get("percentage")),
            "top_terms": top_terms,
            "examples": _load_example_documents(
                dataset_key=dataset_key,
                doc_ids=representative_ids,
            ),
        })

    return {
        "dataset": dataset_key,
        "num_clusters": len(clusters),
        "report_path": str(cluster_report_path),
        "topic_report_path": str(topic_report_path),
        "manifest": _read_optional_json(manifest_path),
        "clusters": clusters,
    }


def _build_topic_payload(dataset_key: str) -> Dict[str, Any]:
    reports_root = _get_reports_root()
    artifacts_root = _get_artifacts_root()
    topic_report_path = (
        reports_root
        / "topics"
        / f"{dataset_key}_cluster_topics.csv"
    )
    manifest_path = (
        artifacts_root
        / "topics"
        / dataset_key
        / "manifest.json"
    )

    topic_rows = _read_csv_report(topic_report_path)
    topics = []

    for row in topic_rows:
        cluster_id = str(row.get("cluster_id", "")).strip()
        top_terms = _parse_top_terms(row.get("top_terms"))
        topic_label = (
            str(row.get("topic_label") or "").strip()
            or _top_terms_to_label(
                top_terms,
                fallback=f"Topic {cluster_id}",
            )
        )
        representative_ids = _split_document_ids(
            row.get("representative_doc_ids")
        )

        topics.append({
            "topic": topic_label,
            "cluster_id": int(cluster_id) if cluster_id.isdigit() else cluster_id,
            "count": int(_coerce_number(row.get("document_count"))),
            "sample_document_count": int(
                _coerce_number(row.get("sample_document_count"))
            ),
            "top_terms": top_terms,
            "examples": _load_example_documents(
                dataset_key=dataset_key,
                doc_ids=representative_ids,
            ),
        })

    return {
        "dataset": dataset_key,
        "topic_count": len(topics),
        "report_path": str(topic_report_path),
        "manifest": _read_optional_json(manifest_path),
        "topics": topics,
    }


@api_view(["GET"])
def evaluation_analytics_view(request):
    try:
        return Response(
            load_evaluation_dashboard(
                reports_root=_get_reports_root(),
            )
        )

    except EvaluationCsvError as error:
        return Response(
            {
                "error": str(error),
                "rows": [],
                "quality_rows": [],
                "before_after_rows": [],
                "speed_rows": [],
            },
            status=404,
        )

    except Exception:
        return Response(
            {
                "error": (
                    "Unexpected error while loading evaluation CSV files."
                ),
                "rows": [],
                "quality_rows": [],
                "before_after_rows": [],
                "speed_rows": [],
            },
            status=500,
        )


@api_view(["GET"])
def evaluation_files_view(request):
    try:
        return Response(
            list_evaluation_csv_files(
                reports_root=_get_reports_root(),
            )
        )

    except Exception:
        return Response(
            {
                "error": (
                    "Unexpected error while inspecting evaluation CSV files."
                ),
                "files": [],
                "file_count": 0,
            },
            status=500,
        )


@api_view(["GET"])
def clustering_analytics_view(request, dataset_key: str):
    try:
        dataset_key = _validate_dataset(dataset_key)
        return Response(_build_cluster_payload(dataset_key))

    except ValueError as error:
        return Response(
            {
                "error": str(error),
                "clusters": [],
            },
            status=400,
        )

    except FileNotFoundError:
        return Response(
            {
                "error": (
                    "No clustering data available. Build or run "
                    "clustering first."
                ),
                "clusters": [],
            },
            status=404,
        )

    except Exception:
        return Response(
            {
                "error": (
                    "Unexpected error while loading clustering analytics."
                ),
                "clusters": [],
            },
            status=500,
        )


@api_view(["GET"])
def topic_detection_analytics_view(request, dataset_key: str):
    try:
        dataset_key = _validate_dataset(dataset_key)
        return Response(_build_topic_payload(dataset_key))

    except ValueError as error:
        return Response(
            {
                "error": str(error),
                "topics": [],
            },
            status=400,
        )

    except FileNotFoundError:
        return Response(
            {
                "error": "No topic detection data available.",
                "topics": [],
            },
            status=404,
        )

    except Exception:
        return Response(
            {
                "error": (
                    "Unexpected error while loading topic detection "
                    "analytics."
                ),
                "topics": [],
            },
            status=500,
        )
