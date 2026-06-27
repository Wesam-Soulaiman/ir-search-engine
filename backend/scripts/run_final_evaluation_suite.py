import argparse
import csv
import json
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

for path in (PROJECT_ROOT, BACKEND_DIR):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

import django
from django.apps import apps
from django.conf import settings

if not apps.ready:
    django.setup()

from evaluation.evaluator import (
    DEFAULT_EVALUATION_QUERY_BATCH_SIZE,
    DEFAULT_LTR_EVALUATION_QUERY_BATCH_SIZE,
    EvaluationRunner,
)
from query_refinement.spelling_correction_service import (
    SpellingCorrectionService,
)


DEFAULT_DATASETS = [
    "quora",
    "clinical_trials",
]

BASELINE_MODELS = [
    "tfidf",
    "bm25",
    "embedding",
    "hybrid_serial",
    "hybrid_parallel",
]

FINAL_FILENAMES = {
    "model_comparison": "final_model_comparison.csv",
    "before_after_features": "final_before_after_features.csv",
    "query_refinement_before_after": (
        "final_query_refinement_before_after.csv"
    ),
    "spelling_correction": "final_spelling_correction.csv",
    "runtime_comparison": "final_runtime_comparison.csv",
    "extra_features": "final_extra_features.csv",
    "clustering_evaluation": "final_clustering_evaluation.csv",
    "topic_detection_evaluation": "final_topic_detection_evaluation.csv",
}

PREFERRED_EVALUATION_COLUMNS = [
    "section",
    "dataset",
    "model",
    "scenario",
    "run_name",
    "feature_group",
    "comparison_phase",
    "status",
    "notes",
    "retrieval_mode",
    "query_batch_size",
    "retrieval_depth",
    "precision_k",
    "recall_k",
    "ndcg_k",
    "loaded_queries",
    "qrels_queries",
    "evaluated_queries",
    "queries_with_positive_qrels",
    "queries_without_positive_qrels",
    "use_query_refinement",
    "use_spelling_correction",
    "misspelled_queries",
    "spelling_corrections_applied",
    "feedback_docs",
    "expansion_terms",
    "bm25_k1",
    "bm25_b",
    "candidate_count",
    "ltr_candidate_models",
    "include_biomedical",
    "ltr_model_path",
    "rrf_k",
    "distributed",
    "num_shards",
    "shard_top_k",
    "biomedical_weight",
    "MAP",
    "Precision@10",
    "Recall",
    "nDCG",
    "EvaluationWallTimeSeconds",
    "AverageQueryTimeMs",
    "QueriesPerSecond",
    "error",
]

CLUSTERING_COLUMNS = [
    "dataset",
    "model_used",
    "embedding_model",
    "number_of_clusters",
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "cluster_id",
    "cluster_size",
    "cluster_percentage",
    "top_terms",
    "example_doc_ids",
    "source_artifact",
    "notes",
]

TOPIC_COLUMNS = [
    "dataset",
    "method",
    "topic_id",
    "topic_size",
    "frequency",
    "top_terms",
    "topic_diversity",
    "topic_coherence",
    "example_doc_ids",
    "number_of_topics",
    "source_artifact",
    "notes",
]

MISSPELL_TOKEN_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z]{3,}\b")


@dataclass(frozen=True)
class EvaluationCase:
    dataset: str
    model: str
    scenario: str
    run_name: str
    section: str
    feature_group: str
    comparison_phase: str | None = None
    notes: str = ""
    use_query_refinement: bool = False
    use_spelling_correction: bool = False
    misspelled_queries: bool = False
    include_biomedical: bool = False
    biomedical_weight: float = 0.0
    ltr_candidate_models: Tuple[str, ...] | None = None
    ltr_model_path: str | None = None


class SpellingEvaluationRunner(EvaluationRunner):
    """
    Evaluate spelling correction with the same qrels and metric code used by
    the normal retrieval evaluator.
    """

    def __init__(
        self,
        *args,
        misspelled_queries: bool = False,
        use_spelling_correction: bool = False,
        **kwargs,
    ):
        self.misspelled_queries = bool(misspelled_queries)
        self.use_spelling_correction = bool(use_spelling_correction)
        self.spelling_corrections_applied = 0
        self.spelling_service = None
        super().__init__(*args, **kwargs)

        if self.use_spelling_correction:
            self.spelling_service = SpellingCorrectionService(
                dataset_key=self.dataset_key,
            )

    def _prepare_search_query(
        self,
        query_id: str,
    ) -> str:
        query = self.query_by_id[query_id]

        if self.misspelled_queries:
            query = make_misspelled_query(query)

        if self.use_spelling_correction and self.spelling_service:
            spelling_result = self.spelling_service.correct(query)
            self.spelling_corrections_applied += len(
                spelling_result.get(
                    "spelling_corrections",
                    [],
                )
            )
            query = str(
                spelling_result.get(
                    "corrected_query",
                    query,
                )
            )

        if self.use_query_refinement and self.query_refinement_service:
            return self.query_refinement_service.refine(query)

        return query


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the final report evaluation suite and write curated CSV "
            "outputs for the Analytics dashboard."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset keys to evaluate. Example: --datasets quora clinical_trials",
    )
    parser.add_argument(
        "--retrieval-depth",
        type=int,
        default=1000,
        help="Number of retrieved documents used for MAP and Recall.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=1000,
        help="Candidate pool size for hybrid and LTR models.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=64,
        help="Batch size for retrieval services that support batch search.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/evaluation/final",
        help=(
            "Output directory for final CSVs. Relative paths are resolved "
            "from the project root."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "baseline",
            "extra",
            "before_after",
            "spelling",
            "runtime",
        ],
        default="all",
        help="Subset of the final evaluation suite to run.",
    )
    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter.",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="BM25 b parameter.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k for hybrid_parallel and distributed_bm25.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Expected shard count for distributed BM25.",
    )
    parser.add_argument(
        "--shard-top-k",
        type=int,
        default=None,
        help="Per-shard top-k for distributed BM25.",
    )
    parser.add_argument(
        "--feedback-docs",
        type=int,
        default=3,
        help="Feedback documents used for PRF query refinement.",
    )
    parser.add_argument(
        "--expansion-terms",
        type=int,
        default=5,
        help="Expansion terms used for PRF query refinement.",
    )
    parser.add_argument(
        "--precision-k",
        type=int,
        default=10,
        help="Precision cutoff. The final report protocol uses 10.",
    )
    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=10,
        help="nDCG cutoff. The final report protocol uses 10.",
    )
    parser.add_argument(
        "--biomedical-weight",
        type=float,
        default=1.0,
        help=(
            "Biomedical embedding weight used for clinical_trials "
            "hybrid_parallel biomedical evaluation."
        ),
    )
    parser.add_argument(
        "--ltr-model-path",
        default=None,
        help="Optional explicit LTR model path.",
    )

    return parser.parse_args()


def validate_args(args):
    if not args.datasets:
        raise ValueError("At least one dataset is required.")

    if args.retrieval_depth <= 0:
        raise ValueError("retrieval-depth must be greater than zero.")

    if args.candidate_count < args.retrieval_depth:
        raise ValueError(
            "candidate-count must be at least retrieval-depth for final "
            "hybrid/LTR comparisons."
        )

    if args.query_batch_size <= 0:
        raise ValueError("query-batch-size must be greater than zero.")

    if args.precision_k <= 0 or args.ndcg_k <= 0:
        raise ValueError("precision-k and ndcg-k must be greater than zero.")

    if max(args.precision_k, args.ndcg_k) > args.retrieval_depth:
        raise ValueError(
            "retrieval-depth must be greater than or equal to Precision and "
            "nDCG cutoffs."
        )


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def resolve_ltr_model_path(
    dataset_key: str,
    explicit_path: str | None = None,
) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
    else:
        artifacts_dir = Path(
            getattr(
                settings,
                "ARTIFACTS_DIR",
                PROJECT_ROOT / "artifacts",
            )
        )
        path = (
            artifacts_dir
            / "models"
            / "ltr"
            / f"{dataset_key}_ltr.joblib"
        )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def resolve_query_batch_size(
    model: str,
    query_batch_size: int | None,
) -> int:
    if query_batch_size is not None:
        return int(query_batch_size)

    if model == "ltr":
        return DEFAULT_LTR_EVALUATION_QUERY_BATCH_SIZE

    return DEFAULT_EVALUATION_QUERY_BATCH_SIZE


def make_misspelled_query(query: str, max_terms: int = 2) -> str:
    replacements = 0

    def replace(match):
        nonlocal replacements

        token = match.group(0)

        if replacements >= max_terms:
            return token

        if len(token) >= 7:
            index = max(2, len(token) // 2)
            mutated = token[:index] + token[index + 1 :]
        elif len(token) >= 5:
            mutated = token[0] + token[2] + token[1] + token[3:]
        else:
            return token

        if mutated != token:
            replacements += 1

        return mutated

    return MISSPELL_TOKEN_PATTERN.sub(replace, str(query or ""))


def metric_key(prefix: str, cutoff: int) -> str:
    return f"{prefix}@{cutoff}"


def add_metric_aliases(
    row: Dict[str, Any],
    retrieval_depth: int,
    precision_k: int,
    recall_k: int,
    ndcg_k: int,
):
    map_key = metric_key("MAP", retrieval_depth)
    precision_key = metric_key("Precision", precision_k)
    recall_key = metric_key("Recall", recall_k)
    ndcg_key = metric_key("nDCG", ndcg_k)

    row["MAP"] = row.get(map_key)
    row["Recall"] = row.get(recall_key)
    row["nDCG"] = row.get(ndcg_key)

    if precision_k == 10:
        row["Precision@10"] = row.get(precision_key)
    else:
        row[precision_key] = row.get(precision_key)


def case_cache_key(
    case: EvaluationCase,
    args,
) -> Tuple[Any, ...]:
    return (
        case.dataset,
        case.model,
        args.retrieval_depth,
        args.precision_k,
        args.retrieval_depth,
        args.ndcg_k,
        args.candidate_count,
        args.bm25_k1,
        args.bm25_b,
        args.rrf_k,
        args.num_shards,
        args.shard_top_k,
        tuple(case.ltr_candidate_models or ()),
        case.include_biomedical,
        case.ltr_model_path,
        case.biomedical_weight,
        case.use_query_refinement,
        case.use_spelling_correction,
        case.misspelled_queries,
        args.feedback_docs,
        args.expansion_terms,
        args.query_batch_size,
    )


def base_result_row(
    case: EvaluationCase,
    args,
    status: str,
    error: str = "",
) -> Dict[str, Any]:
    recall_k = args.retrieval_depth

    row = {
        "section": case.section,
        "dataset": case.dataset,
        "model": case.model,
        "scenario": case.scenario,
        "run_name": case.run_name,
        "feature_group": case.feature_group,
        "comparison_phase": case.comparison_phase,
        "status": status,
        "notes": case.notes,
        "retrieval_depth": args.retrieval_depth,
        "precision_k": args.precision_k,
        "recall_k": recall_k,
        "ndcg_k": args.ndcg_k,
        "use_query_refinement": case.use_query_refinement,
        "use_spelling_correction": case.use_spelling_correction,
        "misspelled_queries": case.misspelled_queries,
        "feedback_docs": (
            args.feedback_docs
            if case.use_query_refinement
            else None
        ),
        "expansion_terms": (
            args.expansion_terms
            if case.use_query_refinement
            else None
        ),
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "candidate_count": (
            args.candidate_count
            if case.model in {"hybrid_serial", "hybrid_parallel", "ltr"}
            else None
        ),
        "ltr_candidate_models": (
            " ".join(case.ltr_candidate_models or ())
            if case.model == "ltr"
            else None
        ),
        "include_biomedical": (
            case.include_biomedical
            if case.model == "ltr"
            else None
        ),
        "ltr_model_path": (
            case.ltr_model_path
            if case.model == "ltr"
            else None
        ),
        "rrf_k": (
            args.rrf_k
            if case.model in {"hybrid_parallel", "distributed_bm25"}
            else None
        ),
        "distributed": case.model == "distributed_bm25",
        "num_shards": (
            args.num_shards
            if case.model == "distributed_bm25"
            else None
        ),
        "shard_top_k": (
            args.shard_top_k
            if case.model == "distributed_bm25"
            else None
        ),
        "biomedical_weight": (
            case.biomedical_weight
            if case.model == "hybrid_parallel"
            else None
        ),
        "error": error,
    }

    add_metric_aliases(
        row=row,
        retrieval_depth=args.retrieval_depth,
        precision_k=args.precision_k,
        recall_k=recall_k,
        ndcg_k=args.ndcg_k,
    )

    return row


def run_runner_for_case(
    case: EvaluationCase,
    args,
) -> Dict[str, Any]:
    runner_class = (
        SpellingEvaluationRunner
        if case.use_spelling_correction or case.misspelled_queries
        else EvaluationRunner
    )

    runner_kwargs = {
        "dataset_key": case.dataset,
        "model_name": case.model,
        "retrieval_depth": args.retrieval_depth,
        "precision_k": args.precision_k,
        "recall_k": args.retrieval_depth,
        "ndcg_k": args.ndcg_k,
        "candidate_count": args.candidate_count,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "rrf_k": args.rrf_k,
        "num_shards": args.num_shards,
        "shard_top_k": args.shard_top_k,
        "ltr_candidate_models": list(case.ltr_candidate_models or []),
        "include_biomedical": case.include_biomedical,
        "ltr_model_path": case.ltr_model_path,
        "biomedical_weight": case.biomedical_weight,
        "use_query_refinement": case.use_query_refinement,
        "feedback_docs": args.feedback_docs,
        "expansion_terms": args.expansion_terms,
        "query_batch_size": resolve_query_batch_size(
            model=case.model,
            query_batch_size=args.query_batch_size,
        ),
    }

    if runner_class is SpellingEvaluationRunner:
        runner_kwargs["misspelled_queries"] = case.misspelled_queries
        runner_kwargs["use_spelling_correction"] = (
            case.use_spelling_correction
        )

    runner = runner_class(**runner_kwargs)
    result = runner.evaluate()

    if isinstance(runner, SpellingEvaluationRunner):
        result["spelling_corrections_applied"] = (
            runner.spelling_corrections_applied
        )

    return result


def evaluate_case(
    case: EvaluationCase,
    args,
    cache: Dict[Tuple[Any, ...], Dict[str, Any]],
) -> Dict[str, Any]:
    if case.model == "ltr":
        ltr_path = resolve_ltr_model_path(
            dataset_key=case.dataset,
            explicit_path=case.ltr_model_path or args.ltr_model_path,
        )
        if not ltr_path.is_file():
            skipped = base_result_row(
                case=EvaluationCase(
                    **{
                        **case.__dict__,
                        "ltr_model_path": str(ltr_path),
                    }
                ),
                args=args,
                status="skipped",
                error=(
                    "LTR model was not found. Train it before running this "
                    f"comparison: {ltr_path}"
                ),
            )
            return skipped

        case = EvaluationCase(
            **{
                **case.__dict__,
                "ltr_model_path": str(ltr_path),
            }
        )

    key = case_cache_key(case, args)

    if key not in cache:
        try:
            print(
                f"Evaluating {case.dataset} / {case.model} / "
                f"{case.scenario}"
            )
            cache[key] = run_runner_for_case(case=case, args=args)
            cache[key]["status"] = "completed"
            cache[key]["error"] = ""
        except Exception as exc:
            cache[key] = base_result_row(
                case=case,
                args=args,
                status="failed",
                error=str(exc),
            )

    row = deepcopy(cache[key])
    row.update({
        "section": case.section,
        "scenario": case.scenario,
        "run_name": case.run_name,
        "feature_group": case.feature_group,
        "comparison_phase": case.comparison_phase,
        "notes": case.notes,
        "use_spelling_correction": case.use_spelling_correction,
        "misspelled_queries": case.misspelled_queries,
    })

    add_metric_aliases(
        row=row,
        retrieval_depth=args.retrieval_depth,
        precision_k=args.precision_k,
        recall_k=args.retrieval_depth,
        ndcg_k=args.ndcg_k,
    )

    return row


def build_baseline_cases(datasets: Iterable[str]) -> List[EvaluationCase]:
    cases = []

    for dataset in datasets:
        for model in BASELINE_MODELS:
            cases.append(
                EvaluationCase(
                    dataset=dataset,
                    model=model,
                    scenario=f"{model}_baseline",
                    run_name=f"{dataset}_{model}_baseline",
                    section="model_comparison",
                    feature_group="baseline_representation",
                    notes="Baseline representation comparison.",
                )
            )

    return cases


def build_query_refinement_cases(
    datasets: Iterable[str],
) -> List[EvaluationCase]:
    cases = []

    for dataset in datasets:
        cases.extend([
            EvaluationCase(
                dataset=dataset,
                model="bm25",
                scenario="before_refinement",
                run_name=f"{dataset}_bm25_before_refinement",
                section="query_refinement_before_after",
                feature_group="query_refinement",
                comparison_phase="before",
                notes="BM25 without pseudo relevance feedback.",
            ),
            EvaluationCase(
                dataset=dataset,
                model="bm25",
                scenario="after_refinement",
                run_name=f"{dataset}_bm25_after_prf_refinement",
                section="query_refinement_before_after",
                feature_group="query_refinement",
                comparison_phase="after",
                use_query_refinement=True,
                notes="BM25 with pseudo relevance feedback query refinement.",
            ),
        ])

    return cases


def build_spelling_cases(datasets: Iterable[str]) -> List[EvaluationCase]:
    cases = []

    for dataset in datasets:
        cases.extend([
            EvaluationCase(
                dataset=dataset,
                model="bm25",
                scenario="clean_queries",
                run_name=f"{dataset}_bm25_clean_queries",
                section="spelling_correction",
                feature_group="spelling_correction",
                comparison_phase="before",
                notes="Original clean evaluation queries.",
            ),
            EvaluationCase(
                dataset=dataset,
                model="bm25",
                scenario="misspelled_without_correction",
                run_name=f"{dataset}_bm25_misspelled_no_correction",
                section="spelling_correction",
                feature_group="spelling_correction",
                comparison_phase="before",
                misspelled_queries=True,
                notes="Deterministically misspelled queries without correction.",
            ),
            EvaluationCase(
                dataset=dataset,
                model="bm25",
                scenario="misspelled_with_correction",
                run_name=f"{dataset}_bm25_misspelled_with_correction",
                section="spelling_correction",
                feature_group="spelling_correction",
                comparison_phase="after",
                misspelled_queries=True,
                use_spelling_correction=True,
                notes="Misspelled queries corrected by the offline service.",
            ),
        ])

    return cases


def build_extra_feature_cases(
    datasets: Iterable[str],
    args,
) -> List[EvaluationCase]:
    cases = []

    for dataset in datasets:
        cases.extend([
            EvaluationCase(
                dataset=dataset,
                model="bm25",
                scenario="central_bm25",
                run_name=f"{dataset}_central_bm25",
                section="before_after_features",
                feature_group="distributed_bm25",
                comparison_phase="before",
                notes="Central BM25 baseline for distributed comparison.",
            ),
            EvaluationCase(
                dataset=dataset,
                model="distributed_bm25",
                scenario="distributed_bm25",
                run_name=f"{dataset}_distributed_bm25",
                section="before_after_features",
                feature_group="distributed_bm25",
                comparison_phase="after",
                notes="Distributed BM25 with shard-level retrieval and RRF.",
            ),
        ])

    if "clinical_trials" in set(datasets):
        clinical_ltr_path = resolve_ltr_model_path(
            dataset_key="clinical_trials",
            explicit_path=args.ltr_model_path,
        )
        cases.extend([
            EvaluationCase(
                dataset="clinical_trials",
                model="bm25",
                scenario="clinical_bm25_baseline",
                run_name="clinical_trials_bm25_for_biomedical",
                section="before_after_features",
                feature_group="biomedical_retrieval",
                comparison_phase="before",
                notes="Clinical Trials BM25 baseline for biomedical features.",
            ),
            EvaluationCase(
                dataset="clinical_trials",
                model="biomedical_embedding",
                scenario="biomedical_embedding",
                run_name="clinical_trials_biomedical_embedding",
                section="before_after_features",
                feature_group="biomedical_retrieval",
                comparison_phase="after",
                notes="Domain-specific biomedical embedding retrieval.",
            ),
            EvaluationCase(
                dataset="clinical_trials",
                model="hybrid_parallel",
                scenario="hybrid_parallel_biomedical",
                run_name="clinical_trials_hybrid_parallel_biomedical",
                section="before_after_features",
                feature_group="biomedical_retrieval",
                comparison_phase="after",
                biomedical_weight=args.biomedical_weight,
                notes="Hybrid parallel retrieval with biomedical weight enabled.",
            ),
        ])

        if clinical_ltr_path.is_file():
            cases.append(
                EvaluationCase(
                    dataset="clinical_trials",
                    model="ltr",
                    scenario="ltr_biomedical_features",
                    run_name="clinical_trials_ltr_biomedical_features",
                    section="before_after_features",
                    feature_group="biomedical_retrieval",
                    comparison_phase="after",
                    include_biomedical=True,
                    ltr_candidate_models=(
                        "bm25",
                        "tfidf",
                        "embedding",
                        # The public standalone search model is named
                        # biomedical_embedding, but LTR feature extraction
                        # expects the internal candidate key biomedical.
                        "biomedical",
                    ),
                    ltr_model_path=str(clinical_ltr_path),
                    notes="LTR with biomedical retrieval features when available.",
                )
            )

    return cases


def metric_value(row: Dict[str, Any], key: str = "MAP") -> float:
    value = row.get(key)

    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def best_non_ltr_baseline_model(
    dataset: str,
    args,
    cache: Dict[Tuple[Any, ...], Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
) -> str:
    rows = [
        row
        for row in baseline_rows
        if row.get("dataset") == dataset
        and row.get("model") in BASELINE_MODELS
        and row.get("status", "completed") != "failed"
    ]

    if not rows:
        rows = [
            evaluate_case(case, args, cache)
            for case in build_baseline_cases([dataset])
        ]

    if not rows:
        return "bm25"

    best_row = max(
        rows,
        key=lambda row: metric_value(row, "MAP"),
    )

    return str(best_row.get("model") or "bm25")


def build_ltr_comparison_cases(
    datasets: Iterable[str],
    args,
    cache: Dict[Tuple[Any, ...], Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
) -> List[EvaluationCase]:
    cases = []

    for dataset in datasets:
        ltr_path = resolve_ltr_model_path(
            dataset_key=dataset,
            explicit_path=args.ltr_model_path,
        )

        if not ltr_path.is_file():
            continue

        best_model = best_non_ltr_baseline_model(
            dataset=dataset,
            args=args,
            cache=cache,
            baseline_rows=baseline_rows,
        )

        cases.extend([
            EvaluationCase(
                dataset=dataset,
                model=best_model,
                scenario="best_non_ltr_baseline",
                run_name=f"{dataset}_{best_model}_best_non_ltr_baseline",
                section="before_after_features",
                feature_group="learning_to_rank",
                comparison_phase="before",
                notes=(
                    "Best measured non-LTR baseline selected by MAP from "
                    "the final baseline representation comparison."
                ),
            ),
            EvaluationCase(
                dataset=dataset,
                model="ltr",
                scenario="learning_to_rank",
                run_name=f"{dataset}_ltr",
                section="before_after_features",
                feature_group="learning_to_rank",
                comparison_phase="after",
                ltr_candidate_models=("bm25", "tfidf", "embedding"),
                ltr_model_path=str(ltr_path),
                notes="LTR reranking with trained model artifacts.",
            ),
        ])

    return cases


def runtime_cases_from(
    datasets: Iterable[str],
    args,
) -> List[EvaluationCase]:
    cases = []

    for dataset in datasets:
        for model in [
            "tfidf",
            "bm25",
            "embedding",
            "hybrid_serial",
            "hybrid_parallel",
            "distributed_bm25",
        ]:
            cases.append(
                EvaluationCase(
                    dataset=dataset,
                    model=model,
                    scenario=f"{model}_runtime",
                    run_name=f"{dataset}_{model}_runtime",
                    section="runtime_comparison",
                    feature_group="runtime",
                    notes="Runtime and throughput comparison.",
                )
            )

        ltr_path = resolve_ltr_model_path(
            dataset_key=dataset,
            explicit_path=args.ltr_model_path,
        )

        if ltr_path.is_file():
            cases.append(
                EvaluationCase(
                    dataset=dataset,
                    model="ltr",
                    scenario="ltr_runtime",
                    run_name=f"{dataset}_ltr_runtime",
                    section="runtime_comparison",
                    feature_group="runtime",
                    ltr_candidate_models=("bm25", "tfidf", "embedding"),
                    ltr_model_path=str(ltr_path),
                    notes="LTR runtime and throughput comparison.",
                )
            )

    if "clinical_trials" in set(datasets):
        cases.extend([
            EvaluationCase(
                dataset="clinical_trials",
                model="biomedical_embedding",
                scenario="biomedical_runtime",
                run_name="clinical_trials_biomedical_runtime",
                section="runtime_comparison",
                feature_group="runtime",
                notes="Biomedical embedding runtime.",
            ),
            EvaluationCase(
                dataset="clinical_trials",
                model="hybrid_parallel",
                scenario="hybrid_parallel_biomedical_runtime",
                run_name="clinical_trials_hybrid_parallel_biomedical_runtime",
                section="runtime_comparison",
                feature_group="runtime",
                biomedical_weight=args.biomedical_weight,
                notes="Hybrid parallel biomedical runtime.",
            ),
        ])

    return cases


def build_extra_feature_table_rows(
    before_after_rows: List[Dict[str, Any]],
    spelling_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for row in before_after_rows:
        extra_row = deepcopy(row)
        extra_row["section"] = "extra_features"
        rows.append(extra_row)

    for row in spelling_rows:
        extra_row = deepcopy(row)
        extra_row["section"] = "extra_features"
        rows.append(extra_row)

    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _topic_rows_by_cluster(dataset: str) -> Dict[str, Dict[str, str]]:
    topic_path = PROJECT_ROOT / "reports" / "topics" / f"{dataset}_cluster_topics.csv"

    if not topic_path.is_file():
        return {}

    return {
        str(row.get("cluster_id", "")).strip(): row
        for row in _read_csv(topic_path)
    }


def build_clustering_evaluation_rows(
    datasets: Iterable[str],
) -> List[Dict[str, Any]]:
    rows = []

    for dataset in datasets:
        manifest_path = (
            PROJECT_ROOT
            / "artifacts"
            / "clustering"
            / dataset
            / "manifest.json"
        )

        if not manifest_path.is_file():
            continue

        manifest = _read_json(manifest_path)
        summary_path = Path(
            manifest.get(
                "files",
                {},
            ).get(
                "summary_report",
                PROJECT_ROOT
                / "reports"
                / "clustering"
                / f"{dataset}_cluster_summary.csv",
            )
        )

        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path

        if not summary_path.is_file():
            continue

        topic_by_cluster = _topic_rows_by_cluster(dataset)

        for summary in _read_csv(summary_path):
            cluster_id = str(summary.get("cluster_id", "")).strip()
            topic = topic_by_cluster.get(cluster_id, {})
            example_doc_ids = (
                topic.get("representative_doc_ids")
                or summary.get("representative_doc_id")
                or ""
            )

            rows.append({
                "dataset": dataset,
                "model_used": manifest.get("algorithm", "MiniBatchKMeans"),
                "embedding_model": manifest.get("embedding_source", ""),
                "number_of_clusters": manifest.get("n_clusters", ""),
                "silhouette_score": manifest.get("silhouette_score", ""),
                "davies_bouldin_score": manifest.get(
                    "davies_bouldin_score",
                    "",
                ),
                "calinski_harabasz_score": manifest.get(
                    "calinski_harabasz_score",
                    "",
                ),
                "cluster_id": cluster_id,
                "cluster_size": summary.get("document_count", ""),
                "cluster_percentage": summary.get("percentage", ""),
                "top_terms": topic.get("top_terms", ""),
                "example_doc_ids": example_doc_ids,
                "source_artifact": _relative(manifest_path),
                "notes": (
                    "Intrinsic scores are blank when the clustering build "
                    "artifact did not compute them."
                ),
            })

    return rows


def calculate_topic_diversity(top_terms: str) -> float | str:
    terms = [
        term.strip().lower()
        for term in re.split(r"[\s,]+", str(top_terms or ""))
        if term.strip()
    ]

    if not terms:
        return ""

    return round(len(set(terms)) / len(terms), 6)


def build_topic_detection_evaluation_rows(
    datasets: Iterable[str],
) -> List[Dict[str, Any]]:
    rows = []

    for dataset in datasets:
        manifest_path = (
            PROJECT_ROOT
            / "artifacts"
            / "topics"
            / dataset
            / "manifest.json"
        )

        if not manifest_path.is_file():
            continue

        manifest = _read_json(manifest_path)
        report_path = Path(
            manifest.get(
                "report",
                PROJECT_ROOT
                / "reports"
                / "topics"
                / f"{dataset}_cluster_topics.csv",
            )
        )

        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path

        if not report_path.is_file():
            continue

        for topic in _read_csv(report_path):
            top_terms = topic.get("top_terms", "")

            rows.append({
                "dataset": dataset,
                "method": manifest.get("feature", "topic_detection"),
                "topic_id": topic.get("cluster_id", ""),
                "topic_size": topic.get("document_count", ""),
                "frequency": topic.get("document_count", ""),
                "top_terms": top_terms,
                "topic_diversity": calculate_topic_diversity(top_terms),
                "topic_coherence": topic.get("topic_coherence", ""),
                "example_doc_ids": topic.get("representative_doc_ids", ""),
                "number_of_topics": manifest.get("n_clusters", ""),
                "source_artifact": _relative(manifest_path),
                "notes": (
                    "Topic coherence is blank unless a separate topic "
                    "coherence artifact is available."
                ),
            })

    return rows


def ordered_fieldnames(
    rows: List[Dict[str, Any]],
    preferred_columns: List[str],
) -> List[str]:
    fieldnames = list(preferred_columns)

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    return fieldnames


def write_csv(
    output_path: Path,
    rows: List[Dict[str, Any]],
    preferred_columns: List[str],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(rows, preferred_columns)

    with output_path.open(
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

    print(f"Wrote {len(rows):,} rows to {_relative(output_path)}")


def suite_includes(
    requested_suite: str,
    section: str,
) -> bool:
    if requested_suite == "all":
        return True

    if requested_suite == "baseline":
        return section == "model_comparison"

    if requested_suite == "before_after":
        return section in {
            "before_after_features",
            "query_refinement_before_after",
        }

    if requested_suite == "spelling":
        return section == "spelling_correction"

    if requested_suite == "runtime":
        return section == "runtime_comparison"

    if requested_suite == "extra":
        return section in {
            "before_after_features",
            "spelling_correction",
            "extra_features",
            "clustering_evaluation",
            "topic_detection_evaluation",
        }

    return False


def write_outputs(
    args,
    output_dir: Path,
    section_rows: Dict[str, List[Dict[str, Any]]],
):
    for section, rows in section_rows.items():
        if not suite_includes(args.suite, section):
            continue

        filename = FINAL_FILENAMES[section]
        columns = (
            CLUSTERING_COLUMNS
            if section == "clustering_evaluation"
            else TOPIC_COLUMNS
            if section == "topic_detection_evaluation"
            else PREFERRED_EVALUATION_COLUMNS
        )
        write_csv(output_dir / filename, rows, columns)


def run_final_suite(args) -> int:
    validate_args(args)
    datasets = [
        str(dataset).strip()
        for dataset in args.datasets
        if str(dataset).strip()
    ]
    output_dir = resolve_project_path(args.output_dir)
    cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    failures = 0
    section_rows: Dict[str, List[Dict[str, Any]]] = {
        section: []
        for section in FINAL_FILENAMES
    }

    if suite_includes(args.suite, "model_comparison"):
        section_rows["model_comparison"] = [
            evaluate_case(case, args, cache)
            for case in build_baseline_cases(datasets)
        ]

    if suite_includes(args.suite, "query_refinement_before_after"):
        query_refinement_rows = [
            evaluate_case(case, args, cache)
            for case in build_query_refinement_cases(datasets)
        ]
        section_rows["query_refinement_before_after"] = (
            query_refinement_rows
        )
    else:
        query_refinement_rows = []

    if suite_includes(args.suite, "spelling_correction"):
        spelling_rows = [
            evaluate_case(case, args, cache)
            for case in build_spelling_cases(datasets)
        ]
        section_rows["spelling_correction"] = spelling_rows
    else:
        spelling_rows = []

    if suite_includes(args.suite, "before_after_features"):
        before_after_cases = (
            build_query_refinement_cases(datasets)
            + build_extra_feature_cases(datasets, args)
            + build_ltr_comparison_cases(
                datasets=datasets,
                args=args,
                cache=cache,
                baseline_rows=section_rows["model_comparison"],
            )
        )
        section_rows["before_after_features"] = [
            {
                **evaluate_case(case, args, cache),
                "section": "before_after_features",
            }
            for case in before_after_cases
        ]

    if suite_includes(args.suite, "runtime_comparison"):
        section_rows["runtime_comparison"] = [
            evaluate_case(case, args, cache)
            for case in runtime_cases_from(datasets, args)
        ]

    if suite_includes(args.suite, "extra_features"):
        if not section_rows["before_after_features"]:
            section_rows["before_after_features"] = [
                {
                    **evaluate_case(case, args, cache),
                    "section": "before_after_features",
                }
                for case in (
                    build_extra_feature_cases(datasets, args)
                    + build_ltr_comparison_cases(
                        datasets=datasets,
                        args=args,
                        cache=cache,
                        baseline_rows=section_rows["model_comparison"],
                    )
                )
            ]

        if not spelling_rows:
            spelling_rows = [
                evaluate_case(case, args, cache)
                for case in build_spelling_cases(datasets)
            ]

        section_rows["extra_features"] = build_extra_feature_table_rows(
            before_after_rows=section_rows["before_after_features"],
            spelling_rows=spelling_rows,
        )

    if suite_includes(args.suite, "clustering_evaluation"):
        section_rows["clustering_evaluation"] = (
            build_clustering_evaluation_rows(datasets)
        )

    if suite_includes(args.suite, "topic_detection_evaluation"):
        section_rows["topic_detection_evaluation"] = (
            build_topic_detection_evaluation_rows(datasets)
        )

    for rows in section_rows.values():
        failures += sum(
            1
            for row in rows
            if row.get("status") == "failed"
        )

    write_outputs(
        args=args,
        output_dir=output_dir,
        section_rows=section_rows,
    )

    if failures:
        print(f"Completed with {failures:,} failed evaluation row(s).")
        return 1

    print("Final evaluation suite completed.")
    return 0


def main():
    args = parse_args()

    try:
        exit_code = run_final_suite(args)
    except ValueError as exc:
        raise SystemExit(f"Invalid final evaluation arguments: {exc}") from exc

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
