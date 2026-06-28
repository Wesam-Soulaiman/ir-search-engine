import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class EvaluationCsvError(RuntimeError):
    """Raised when evaluation CSV files cannot be loaded."""


NO_REPORT_MANIFEST_MESSAGE = (
    "No report manifest found. Create "
    "reports/evaluation/report_manifest.json to select final report CSV files."
)


SECTION_DEFINITIONS = {
    "model_comparison": {
        "label": "Model Quality Comparison",
        "description": (
            "Report-ready model quality rows selected by the manifest."
        ),
    },
    "before_after_features": {
        "label": "Before/After Extra Features",
        "description": (
            "Final report rows comparing baseline retrieval with integrated "
            "extra features."
        ),
    },
    "query_refinement_before_after": {
        "label": "Query Refinement Before/After",
        "description": (
            "Rows selected for before/after query refinement comparison."
        ),
    },
    "spelling_correction": {
        "label": "Spelling Correction",
        "description": (
            "Clean, misspelled, and corrected-query runs selected for "
            "spelling correction analysis."
        ),
    },
    "runtime_comparison": {
        "label": "Runtime Comparison",
        "description": (
            "Rows selected for evaluation time, latency, or QPS charts."
        ),
    },
    "extra_features": {
        "label": "Extra Features",
        "description": (
            "Rows selected for integrated extra IR features such as LTR, "
            "distributed BM25, biomedical retrieval, and batch hybrid runs."
        ),
    },
    "clustering_evaluation": {
        "label": "Clustering Evaluation",
        "description": (
            "Intrinsic clustering summary rows and representative cluster "
            "examples selected for the final report."
        ),
    },
    "topic_detection_evaluation": {
        "label": "Topic Detection Evaluation",
        "description": (
            "Topic frequency, representative terms, and intrinsic topic "
            "summary rows selected for the final report."
        ),
    },
}


SECTION_ORDER = list(SECTION_DEFINITIONS.keys())


def _normalize_column_name(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value or "").strip().lower(),
    )


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    value = value.strip()

    if value == "":
        return ""

    try:
        if "." in value:
            return float(value)

        return int(value)
    except ValueError:
        return value


def _coerce_number(value: Any) -> float | None:
    value = _coerce_value(value)

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    normalized_value = str(value or "").strip().lower()

    if normalized_value in {"true", "1", "yes", "y", "on"}:
        return True

    if normalized_value in {"false", "0", "no", "n", "off"}:
        return False

    return None


def _get_value(row: Dict[str, Any], aliases: Iterable[str]) -> Any:
    normalized_row = {
        _normalize_column_name(key): value
        for key, value in row.items()
    }

    for alias in aliases:
        normalized_alias = _normalize_column_name(alias)

        if normalized_alias in normalized_row:
            value = normalized_row[normalized_alias]

            if str(value or "").strip() != "":
                return value

    return None


def _get_metric(
    row: Dict[str, Any],
    preferred_aliases: Iterable[str],
    fallback_prefixes: Iterable[str],
) -> Tuple[float | None, str | None]:
    normalized_row = {
        _normalize_column_name(key): (key, value)
        for key, value in row.items()
    }

    for alias in preferred_aliases:
        normalized_alias = _normalize_column_name(alias)
        match = normalized_row.get(normalized_alias)

        if match is None:
            continue

        original_key, value = match
        number_value = _coerce_number(value)

        if number_value is not None:
            return number_value, original_key

    normalized_prefixes = [
        _normalize_column_name(prefix)
        for prefix in fallback_prefixes
    ]

    for normalized_key, (original_key, value) in normalized_row.items():
        if not any(
            normalized_key.startswith(prefix)
            for prefix in normalized_prefixes
        ):
            continue

        number_value = _coerce_number(value)

        if number_value is not None:
            return number_value, original_key

    return None, None


def _evaluation_dir(reports_root: str | Path) -> Path:
    return Path(reports_root).expanduser().resolve() / "evaluation"


def _manifest_path(reports_root: str | Path) -> Path:
    return _evaluation_dir(reports_root) / "report_manifest.json"


def _relative_to_reports(path: Path, reports_root: Path) -> str:
    try:
        return str(path.relative_to(reports_root)).replace("\\", "/")
    except ValueError:
        return path.name


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        return [
            {key: _coerce_value(value) for key, value in row.items()}
            for row in reader
        ]


def _inspect_csv_file(
    path: Path,
    reports_root: Path,
) -> Dict[str, Any]:
    columns: List[str] = []
    row_count = 0
    error = None

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            columns = list(reader.fieldnames or [])
            row_count = sum(1 for _ in reader)
    except Exception as exc:
        error = str(exc)

    return {
        "name": path.name,
        "relative_path": _relative_to_reports(path, reports_root),
        "size": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(
            path.stat().st_mtime,
            timezone.utc,
        ).isoformat(),
        "detected_columns": columns,
        "row_count": row_count,
        "error": error,
    }


def _source_metadata(
    path: Path,
    reports_root: Path,
    row_count: int,
) -> Dict[str, Any]:
    return {
        "name": path.name,
        "relative_path": _relative_to_reports(path, reports_root),
        "path": str(path),
        "size": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(
            path.stat().st_mtime,
            timezone.utc,
        ).isoformat(),
        "row_count": int(row_count),
    }


def _read_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise EvaluationCsvError("report_manifest.json must contain an object.")

    return manifest


def _normalize_section_manifest(
    manifest: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    raw_sections = manifest.get("sections", {})
    normalized_sections: Dict[str, Dict[str, Any]] = {}

    if isinstance(raw_sections, list):
        for section in raw_sections:
            if not isinstance(section, dict):
                continue

            key = str(section.get("key") or "").strip()

            if key:
                normalized_sections[key] = dict(section)

    elif isinstance(raw_sections, dict):
        for key, section in raw_sections.items():
            if isinstance(section, dict):
                normalized_sections[str(key)] = dict(section)

    for key, defaults in SECTION_DEFINITIONS.items():
        section = normalized_sections.setdefault(key, {})
        section.setdefault("label", defaults["label"])
        section.setdefault("description", defaults["description"])
        section.setdefault("files", [])

    return normalized_sections


def _resolve_manifest_csv_path(
    evaluation_dir: Path,
    csv_reference: str,
) -> Path:
    if not str(csv_reference or "").strip():
        raise EvaluationCsvError("Manifest contains an empty CSV reference.")

    candidate = Path(str(csv_reference).strip())

    if candidate.is_absolute():
        csv_path = candidate.expanduser().resolve()
    else:
        csv_path = (evaluation_dir / candidate).resolve()

    evaluation_root = evaluation_dir.resolve()

    try:
        csv_path.relative_to(evaluation_root)
    except ValueError as exc:
        raise EvaluationCsvError(
            f"Manifest CSV path escapes reports/evaluation: {csv_reference}"
        ) from exc

    if csv_path.suffix.lower() != ".csv":
        raise EvaluationCsvError(
            f"Manifest entry is not a CSV file: {csv_reference}"
        )

    return csv_path


def _infer_scenario_from_source(source_path: Path) -> str:
    stem = source_path.stem.lower()

    if "bm25_refinement_improved" in stem:
        return "after_refinement_improved"

    if "bm25_refinement_depth" in stem or "bm25_refinement" in stem:
        return "after_refinement"

    if "bm25_baseline" in stem:
        return "before_refinement"

    if "full_bm25" in stem:
        return "bm25_baseline"

    if "full_tfidf" in stem:
        return "tfidf_baseline"

    if "distributed_bm25" in stem:
        return "distributed"

    if "hybrid_parallel_batch" in stem:
        return "hybrid_parallel_batch"

    if "biomedical" in stem:
        return "biomedical"

    if "ltr" in stem:
        return "learning_to_rank"

    if "baseline" in stem:
        return "baseline"

    if "with_refinement" in stem or "refinement" in stem:
        return "after_refinement"

    retrieval_mode = stem.split("_", 2)[-1] if "_" in stem else stem
    return retrieval_mode


def _infer_comparison_phase(
    row: Dict[str, Any],
    source_file: str,
    scenario: str,
) -> str | None:
    explicit_phase = _get_value(
        row,
        [
            "comparison_phase",
            "phase",
        ],
    )

    if explicit_phase:
        return str(explicit_phase).strip()

    refinement_value = _get_value(
        row,
        [
            "use_query_refinement",
            "query_refinement",
            "refinement",
        ],
    )
    refinement_enabled = _coerce_bool(refinement_value)

    if refinement_enabled is True:
        return "after"

    if refinement_enabled is False:
        return "before"

    if scenario in {
        "before_refinement",
        "bm25_baseline",
        "tfidf_baseline",
        "baseline",
        "clean_queries",
        "misspelled_without_correction",
        "central_bm25",
        "best_non_ltr_baseline",
    }:
        return "before"

    if scenario in {
        "after_refinement",
        "after_refinement_improved",
        "misspelled_with_correction",
        "distributed_bm25",
        "biomedical_embedding",
        "hybrid_parallel_biomedical",
        "learning_to_rank",
    }:
        return "after"

    lowered_source = source_file.lower()

    if (
        "with_refinement" in lowered_source
        or "refinement" in lowered_source
        or "_prf_" in lowered_source
        or lowered_source.startswith("prf_")
    ):
        return "after"

    if "baseline" in lowered_source or "full_bm25" in lowered_source:
        return "before"

    return None


def _build_scenario(
    row: Dict[str, Any],
    source_path: Path,
) -> str:
    explicit_scenario = _get_value(
        row,
        [
            "scenario",
            "run_name",
            "run",
            "name",
        ],
    )

    if explicit_scenario:
        return str(explicit_scenario).strip()

    return _infer_scenario_from_source(source_path)


def _normalize_evaluation_row(
    row: Dict[str, Any],
    source_path: Path,
    reports_root: Path,
    section_key: str | None = None,
) -> Dict[str, Any]:
    map_value, map_column = _get_metric(
        row,
        ["MAP", "MAP@1000", "MAP@500", "MAP@100"],
        ["MAP"],
    )
    precision_at_10, precision_column = _get_metric(
        row,
        ["Precision@10", "precision_at_10", "P@10"],
        ["Precision"],
    )
    recall_value, recall_column = _get_metric(
        row,
        [
            "Recall@1000",
            "Recall@500",
            "Recall@100",
            "Recall@10",
            "Recall@5",
            "Recall@K",
            "Recall",
        ],
        ["Recall"],
    )
    ndcg_value, ndcg_column = _get_metric(
        row,
        [
            "nDCG@10",
            "NDCG@10",
            "nDCG@5",
            "NDCG@5",
            "nDCG",
            "NDCG",
        ],
        ["nDCG", "NDCG"],
    )
    wall_time_seconds, wall_time_column = _get_metric(
        row,
        [
            "EvaluationWallTimeSeconds",
            "wall_time_seconds",
            "WallTimeSeconds",
        ],
        ["EvaluationWallTimeSeconds", "WallTime"],
    )
    average_latency_ms, latency_column = _get_metric(
        row,
        [
            "AverageLatencyMs",
            "AverageQueryTimeMs",
            "AvgLatencyMs",
            "LatencyMs",
        ],
        ["AverageLatency", "AverageQueryTime", "Latency"],
    )
    qps, qps_column = _get_metric(
        row,
        [
            "QPS",
            "QueriesPerSecond",
            "queries_per_second",
        ],
        ["QPS", "QueriesPerSecond"],
    )

    source_file = source_path.name
    source_csv = _relative_to_reports(source_path, reports_root)
    scenario = _build_scenario(row=row, source_path=source_path)
    comparison_phase = _infer_comparison_phase(
        row=row,
        source_file=source_file,
        scenario=scenario,
    )
    dataset = _get_value(row, ["dataset", "dataset_key"]) or ""
    model = (
        _get_value(
            row,
            [
                "model",
                "retrieval_model",
                "model_used",
                "method",
            ],
        )
        or ""
    )

    return {
        "dataset": str(dataset).strip(),
        "model": str(model).strip(),
        "scenario": scenario,
        "run_name": str(_get_value(row, ["run_name"]) or source_path.stem),
        "feature_group": str(
            _get_value(row, ["feature_group", "feature"]) or ""
        ),
        "status": str(_get_value(row, ["status"]) or ""),
        "warning": str(_get_value(row, ["warning"]) or ""),
        "source_file": source_file,
        "source_csv": source_csv,
        "section_key": section_key,
        "map": map_value,
        "map_column": map_column,
        "precision_at_10": precision_at_10,
        "precision_column": precision_column,
        "recall": recall_value,
        "recall_column": recall_column,
        "ndcg": ndcg_value,
        "ndcg_column": ndcg_column,
        "wall_time_seconds": wall_time_seconds,
        "wall_time_column": wall_time_column,
        "average_latency_ms": average_latency_ms,
        "latency_column": latency_column,
        "qps": qps,
        "qps_column": qps_column,
        "comparison_phase": comparison_phase,
        "raw": row,
    }


def _empty_section(key: str) -> Dict[str, Any]:
    defaults = SECTION_DEFINITIONS.get(
        key,
        {
            "label": key.replace("_", " ").title(),
            "description": "",
        },
    )

    return {
        "key": key,
        "label": defaults["label"],
        "description": defaults["description"],
        "sources": [],
        "source_csv_paths": [],
        "rows": [],
        "row_count": 0,
    }


def _empty_dashboard(
    reports_root: Path,
    message: str,
) -> Dict[str, Any]:
    evaluation_dir = _evaluation_dir(reports_root)
    manifest_path = _manifest_path(reports_root)

    return {
        "manifest_found": False,
        "message": message,
        "reports_dir": str(evaluation_dir),
        "manifest_path": str(manifest_path),
        "section_order": SECTION_ORDER,
        "sections": {
            key: _empty_section(key)
            for key in SECTION_ORDER
        },
        "files": [],
        "file_count": 0,
        "row_count": 0,
        "rows": [],
        "quality_rows": [],
        "before_after_feature_rows": [],
        "before_after_rows": [],
        "spelling_correction_rows": [],
        "speed_rows": [],
        "extra_feature_rows": [],
        "clustering_evaluation_rows": [],
        "topic_detection_evaluation_rows": [],
        "errors": [],
    }


def list_evaluation_csv_files(
    reports_root: str | Path,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    evaluation_dir = _evaluation_dir(reports_root)
    manifest_path = _manifest_path(reports_root)

    if not evaluation_dir.is_dir():
        return {
            "reports_dir": str(evaluation_dir),
            "manifest_path": str(manifest_path),
            "manifest_found": False,
            "files": [],
            "file_count": 0,
        }

    csv_files = sorted(evaluation_dir.rglob("*.csv"))

    files = [
        _inspect_csv_file(
            path=csv_path,
            reports_root=reports_root,
        )
        for csv_path in csv_files
    ]

    return {
        "reports_dir": str(evaluation_dir),
        "manifest_path": str(manifest_path),
        "manifest_found": manifest_path.is_file(),
        "files": files,
        "file_count": len(files),
    }


def load_evaluation_dashboard(
    reports_root: str | Path,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    evaluation_dir = _evaluation_dir(reports_root)
    manifest_path = _manifest_path(reports_root)

    if not manifest_path.is_file():
        return _empty_dashboard(
            reports_root=reports_root,
            message=NO_REPORT_MANIFEST_MESSAGE,
        )

    manifest = _read_manifest(manifest_path)
    section_manifest = _normalize_section_manifest(manifest)
    sections = {
        key: _empty_section(key)
        for key in SECTION_ORDER
    }
    rows: List[Dict[str, Any]] = []
    errors = []
    unique_source_paths: Dict[str, Dict[str, Any]] = {}

    for section_key, section_config in section_manifest.items():
        section = sections.setdefault(
            section_key,
            _empty_section(section_key),
        )
        section["label"] = str(
            section_config.get("label")
            or section["label"]
        )
        section["description"] = str(
            section_config.get("description")
            or section["description"]
        )

        files = section_config.get("files", [])

        if isinstance(files, str):
            files = [files]

        if not isinstance(files, list):
            errors.append({
                "section": section_key,
                "error": "Section files must be a list.",
            })
            continue

        for csv_reference in files:
            try:
                csv_path = _resolve_manifest_csv_path(
                    evaluation_dir=evaluation_dir,
                    csv_reference=str(csv_reference),
                )

                if not csv_path.is_file():
                    raise FileNotFoundError(
                        f"Manifest CSV file was not found: {csv_reference}"
                    )

                csv_rows = _read_csv_rows(csv_path)
                normalized_rows = [
                    _normalize_evaluation_row(
                        row=row,
                        source_path=csv_path,
                        reports_root=reports_root,
                        section_key=section_key,
                    )
                    for row in csv_rows
                ]
                source = _source_metadata(
                    path=csv_path,
                    reports_root=reports_root,
                    row_count=len(normalized_rows),
                )

                section["sources"].append(source)
                section["source_csv_paths"].append(source["relative_path"])
                section["rows"].extend(normalized_rows)
                section["row_count"] = len(section["rows"])
                rows.extend(normalized_rows)
                unique_source_paths[source["relative_path"]] = source

            except Exception as error:
                errors.append({
                    "section": section_key,
                    "source_csv": str(csv_reference),
                    "error": str(error),
                })

    ordered_sections = {
        key: sections[key]
        for key in SECTION_ORDER
        if key in sections
    }

    for key, section in sections.items():
        if key not in ordered_sections:
            ordered_sections[key] = section

    return {
        "manifest_found": True,
        "message": "",
        "manifest": manifest,
        "reports_dir": str(evaluation_dir),
        "manifest_path": str(manifest_path),
        "section_order": list(ordered_sections.keys()),
        "sections": ordered_sections,
        "files": list(unique_source_paths.values()),
        "file_count": len(unique_source_paths),
        "row_count": len(rows),
        "rows": rows,
        "quality_rows": ordered_sections["model_comparison"]["rows"],
        "before_after_feature_rows": (
            ordered_sections["before_after_features"]["rows"]
        ),
        "before_after_rows": (
            ordered_sections["query_refinement_before_after"]["rows"]
        ),
        "spelling_correction_rows": (
            ordered_sections["spelling_correction"]["rows"]
        ),
        "speed_rows": ordered_sections["runtime_comparison"]["rows"],
        "extra_feature_rows": ordered_sections["extra_features"]["rows"],
        "clustering_evaluation_rows": (
            ordered_sections["clustering_evaluation"]["rows"]
        ),
        "topic_detection_evaluation_rows": (
            ordered_sections["topic_detection_evaluation"]["rows"]
        ),
        "errors": errors,
    }
