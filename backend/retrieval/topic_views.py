import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from datasets.dataset_registry import get_dataset_config


def _get_project_root() -> Path:
    """
    Resolve the repository root from Django's BASE_DIR.
    """
    backend_dir = Path(settings.BASE_DIR).resolve()
    return backend_dir.parent


def _get_reports_root() -> Path:
    return Path(
        getattr(
            settings,
            "REPORTS_DIR",
            _get_project_root() / "reports",
        )
    ).expanduser().resolve()


def _get_artifacts_root() -> Path:
    return Path(
        getattr(
            settings,
            "ARTIFACTS_DIR",
            _get_project_root() / "artifacts",
        )
    ).expanduser().resolve()


def _validate_dataset(dataset_key: str) -> str:
    dataset_key = str(dataset_key or "").strip().lower()

    if not dataset_key:
        raise ValueError("dataset_key is required.")

    get_dataset_config(dataset_key)
    return dataset_key


def _coerce_value(value: str) -> Any:
    """
    Convert CSV string values into convenient JSON values.
    """
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


def _read_csv_report(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Report file was not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
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


@api_view(["GET"])
def cluster_summary_view(request, dataset_key: str):
    """
    Return the document clustering summary for one dataset.
    """
    try:
        dataset_key = _validate_dataset(dataset_key)
        reports_root = _get_reports_root()
        artifacts_root = _get_artifacts_root()
        report_path = (
            reports_root
            / "clustering"
            / f"{dataset_key}_cluster_summary.csv"
        )
        manifest_path = (
            artifacts_root
            / "clustering"
            / dataset_key
            / "manifest.json"
        )

        clusters = _read_csv_report(report_path)
        manifest = _read_optional_json(manifest_path)

        return Response(
            {
                "dataset": dataset_key,
                "feature": "document_clustering",
                "cluster_count": len(clusters),
                "report_path": str(report_path),
                "manifest": manifest,
                "clusters": clusters,
            }
        )

    except ValueError as error:
        return Response({"error": str(error), "clusters": []}, status=400)

    except FileNotFoundError as error:
        return Response({"error": str(error), "clusters": []}, status=404)

    except Exception:
        return Response(
            {
                "error": "Unexpected error while loading cluster summary.",
                "clusters": [],
            },
            status=500,
        )


@api_view(["GET"])
def cluster_topics_view(request, dataset_key: str):
    """
    Return topic labels inferred for document clusters.
    """
    try:
        dataset_key = _validate_dataset(dataset_key)
        reports_root = _get_reports_root()
        artifacts_root = _get_artifacts_root()
        report_path = (
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

        topics = _read_csv_report(report_path)
        manifest = _read_optional_json(manifest_path)

        return Response(
            {
                "dataset": dataset_key,
                "feature": "topic_detection",
                "topic_count": len(topics),
                "report_path": str(report_path),
                "manifest": manifest,
                "topics": topics,
            }
        )

    except ValueError as error:
        return Response({"error": str(error), "topics": []}, status=400)

    except FileNotFoundError as error:
        return Response({"error": str(error), "topics": []}, status=404)

    except Exception:
        return Response(
            {
                "error": "Unexpected error while loading cluster topics.",
                "topics": [],
            },
            status=500,
        )
