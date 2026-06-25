import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = None

for candidate in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
    if candidate.name == "backend":
        BACKEND_DIR = candidate
        break

if BACKEND_DIR is None:
    # Fallback for unusual copies of this file.
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

django.setup()

from django.conf import settings

from evaluation.evaluator import (
    EvaluationRunner,
    SUPPORTED_MODELS,
)


DEFAULT_MODELS = [
    "tfidf",
    "bm25",
    "embedding",
    "hybrid_serial",
    "hybrid_parallel",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval models using every "
            "query represented in dataset qrels."
        )
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Single dataset key from dataset registry. "
            "Kept for backward compatibility."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "One or more dataset keys from dataset registry. "
            "Example: --datasets quora clinical_trials"
        ),
    )

    parser.add_argument(
        "--retrieval-depth",
        "--top-k",
        dest="retrieval_depth",
        type=int,
        default=1000,
        help=(
            "Number of documents retrieved per query. "
            "--top-k remains as a backward-compatible alias."
        ),
    )

    parser.add_argument(
        "--precision-k",
        type=int,
        default=10,
        help="Cutoff used for Precision@K.",
    )

    parser.add_argument(
        "--recall-k",
        type=int,
        default=None,
        help=(
            "Cutoff used for Recall@K. "
            "Defaults to retrieval depth."
        ),
    )

    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=10,
        help="Cutoff used for nDCG@K.",
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=1000,
        help=(
            "Candidate count for hybrid models. "
            "Must be at least retrieval depth."
        ),
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
        help="RRF k parameter for hybrid parallel.",
    )

    parser.add_argument(
        "--biomedical-weight",
        type=float,
        default=0.0,
        help=(
            "Optional biomedical PubMedBERT embedding weight for "
            "hybrid_parallel. Defaults to 0.0."
        ),
    )

    parser.add_argument(
        "--use-query-refinement",
        action="store_true",
        help="Enable pseudo-relevance feedback.",
    )

    parser.add_argument(
        "--feedback-docs",
        type=int,
        default=3,
        help=(
            "Number of feedback documents used "
            "for query refinement."
        ),
    )

    parser.add_argument(
        "--expansion-terms",
        type=int,
        default=5,
        help="Number of query expansion terms.",
    )

    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=64,
        help=(
            "Number of queries evaluated together for models "
            "that support batch retrieval."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(
            SUPPORTED_MODELS
        ),
        default=DEFAULT_MODELS,
        help=(
            "Models to evaluate. By default all "
            "retrieval models are evaluated."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV path. Only allowed when "
            "evaluating one dataset."
        ),
    )

    return parser.parse_args()


def resolve_dataset_keys(args) -> List[str]:
    if args.dataset and args.datasets:
        raise ValueError(
            "Use either --dataset or --datasets, not both."
        )

    if args.datasets:
        dataset_keys = args.datasets

    elif args.dataset:
        dataset_keys = [args.dataset]

    else:
        dataset_keys = ["sample_dataset"]

    normalized_keys = []

    for dataset_key in dataset_keys:
        cleaned_key = str(dataset_key).strip()

        if not cleaned_key:
            raise ValueError(
                "dataset key cannot be empty."
            )

        normalized_keys.append(cleaned_key)

    return normalized_keys


def validate_args(args, dataset_keys: List[str]):
    if len(dataset_keys) > 1 and args.output:
        raise ValueError(
            "--output can only be used with one dataset. "
            "When multiple datasets are requested, default "
            "per-dataset output paths are used."
        )

    if args.retrieval_depth <= 0:
        raise ValueError(
            "retrieval-depth must be greater "
            "than zero."
        )

    if args.precision_k <= 0:
        raise ValueError(
            "precision-k must be greater "
            "than zero."
        )

    if (
        args.recall_k is not None
        and args.recall_k <= 0
    ):
        raise ValueError(
            "recall-k must be greater "
            "than zero."
        )

    if args.ndcg_k <= 0:
        raise ValueError(
            "ndcg-k must be greater than zero."
        )

    recall_k = (
        args.recall_k
        if args.recall_k is not None
        else args.retrieval_depth
    )

    largest_cutoff = max(
        args.precision_k,
        recall_k,
        args.ndcg_k,
    )

    if (
        largest_cutoff
        > args.retrieval_depth
    ):
        raise ValueError(
            "retrieval-depth must be at least "
            "as large as every metric cutoff. "
            f"retrieval-depth="
            f"{args.retrieval_depth}, "
            f"largest cutoff="
            f"{largest_cutoff}."
        )

    if args.candidate_count <= 0:
        raise ValueError(
            "candidate-count must be greater "
            "than zero."
        )

    hybrid_models_requested = any(
        model
        in {
            "hybrid_serial",
            "hybrid_parallel",
        }
        for model in args.models
    )

    if (
        hybrid_models_requested
        and args.candidate_count
        < args.retrieval_depth
    ):
        raise ValueError(
            "candidate-count must be at least "
            "retrieval-depth when evaluating "
            "hybrid models."
        )

    if args.bm25_k1 <= 0:
        raise ValueError(
            "bm25-k1 must be greater than zero."
        )

    if not 0.0 <= args.bm25_b <= 1.0:
        raise ValueError(
            "bm25-b must be between 0 and 1."
        )

    if args.rrf_k <= 0:
        raise ValueError(
            "rrf-k must be greater than zero."
        )

    if args.biomedical_weight < 0:
        raise ValueError(
            "biomedical-weight must be greater than or equal to zero."
        )

    if (
        args.biomedical_weight > 0
        and "hybrid_parallel" in args.models
        and any(
            dataset_key != "clinical_trials"
            for dataset_key in dataset_keys
        )
    ):
        raise ValueError(
            "biomedical-weight can only be used with hybrid_parallel "
            "on the clinical_trials dataset."
        )

    if args.query_batch_size <= 0:
        raise ValueError(
            "query-batch-size must be greater "
            "than zero."
        )

    if args.use_query_refinement:
        if args.feedback_docs <= 0:
            raise ValueError(
                "feedback-docs must be greater "
                "than zero."
            )

        if args.expansion_terms <= 0:
            raise ValueError(
                "expansion-terms must be greater "
                "than zero."
            )


def get_default_output_path(
    dataset_key: str,
    retrieval_depth: int,
    use_query_refinement: bool = False,
) -> str:
    project_root = settings.BASE_DIR.parent

    output_dir = os.path.join(
        project_root,
        "reports",
        "evaluation",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    suffix = (
        "with_refinement"
        if use_query_refinement
        else "baseline"
    )

    filename = (
        f"{dataset_key}_"
        f"{suffix}_"
        f"depth_{retrieval_depth}.csv"
    )

    return os.path.join(
        output_dir,
        filename,
    )


def run_model_evaluation(
    dataset_key: str,
    model_name: str,
    retrieval_depth: int,
    precision_k: int,
    recall_k: int | None,
    ndcg_k: int,
    candidate_count: int,
    bm25_k1: float,
    bm25_b: float,
    rrf_k: int,
    biomedical_weight: float,
    use_query_refinement: bool,
    feedback_docs: int,
    expansion_terms: int,
    query_batch_size: int,
) -> Dict:
    runner = EvaluationRunner(
        dataset_key=dataset_key,
        model_name=model_name,
        retrieval_depth=retrieval_depth,
        precision_k=precision_k,
        recall_k=recall_k,
        ndcg_k=ndcg_k,
        candidate_count=candidate_count,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        rrf_k=rrf_k,
        biomedical_weight=biomedical_weight,
        use_query_refinement=(
            use_query_refinement
        ),
        feedback_docs=feedback_docs,
        expansion_terms=expansion_terms,
        query_batch_size=query_batch_size,
    )

    return runner.evaluate()


def save_results_to_csv(
    results: List[Dict],
    output_path: str,
):
    if not results:
        raise ValueError(
            "No evaluation results to save."
        )

    output_path = os.path.abspath(
        output_path
    )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    fieldnames: List[str] = []

    for result in results:
        for key in result:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def build_failure_result(
    args,
    dataset_key: str,
    model_name: str,
    error: Exception,
) -> Dict:
    recall_k = (
        args.recall_k
        if args.recall_k is not None
        else args.retrieval_depth
    )

    return {
        "dataset": dataset_key,
        "model": model_name,
        "retrieval_mode": None,
        "query_batch_size": args.query_batch_size,
        "retrieval_depth": (
            args.retrieval_depth
        ),
        "precision_k": (
            args.precision_k
        ),
        "recall_k": recall_k,
        "ndcg_k": args.ndcg_k,
        "loaded_queries": None,
        "qrels_queries": None,
        "evaluated_queries": 0,
        "queries_with_positive_qrels": None,
        "queries_without_positive_qrels": None,
        "use_query_refinement": (
            args.use_query_refinement
        ),
        "feedback_docs": (
            args.feedback_docs
            if args.use_query_refinement
            else None
        ),
        "expansion_terms": (
            args.expansion_terms
            if args.use_query_refinement
            else None
        ),
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "candidate_count": (
            args.candidate_count
            if model_name
            in {
                "hybrid_serial",
                "hybrid_parallel",
            }
            else None
        ),
        "rrf_k": (
            args.rrf_k
            if model_name
            == "hybrid_parallel"
            else None
        ),
        "biomedical_weight": (
            args.biomedical_weight
            if model_name
            == "hybrid_parallel"
            else None
        ),
        f"MAP@{args.retrieval_depth}": None,
        f"Precision@{args.precision_k}": None,
        f"Recall@{recall_k}": None,
        f"nDCG@{args.ndcg_k}": None,
        "EvaluationWallTimeSeconds": None,
        "AverageQueryTimeMs": None,
        "QueriesPerSecond": None,
        "error": str(error),
    }


def print_configuration(
    args,
    dataset_key: str,
    output_path: str,
):
    resolved_recall_k = (
        args.recall_k
        if args.recall_k is not None
        else args.retrieval_depth
    )

    print("=" * 70)
    print("Evaluation configuration")
    print("=" * 70)
    print(f"Dataset: {dataset_key}")
    print(
        "Models: "
        + ", ".join(args.models)
    )
    print(
        "Retrieval depth: "
        f"{args.retrieval_depth:,}"
    )
    print(
        "Precision cutoff: "
        f"{args.precision_k:,}"
    )
    print(
        "Recall cutoff: "
        f"{resolved_recall_k:,}"
    )
    print(
        "nDCG cutoff: "
        f"{args.ndcg_k:,}"
    )
    print(
        "Candidate count: "
        f"{args.candidate_count:,}"
    )
    print(
        "Biomedical weight: "
        f"{args.biomedical_weight:g}"
    )
    print(
        "Query batch size: "
        f"{args.query_batch_size:,}"
    )
    print(
        "Query refinement: "
        f"{args.use_query_refinement}"
    )
    print(
        "Output: "
        f"{os.path.abspath(output_path)}"
    )
    print("=" * 70)


def evaluate_dataset(
    args,
    dataset_key: str,
    output_path: str,
) -> int:
    print_configuration(
        args=args,
        dataset_key=dataset_key,
        output_path=output_path,
    )

    all_results: List[Dict] = []
    failure_count = 0

    for model_name in args.models:
        print()
        print("=" * 70)
        print(
            f"Evaluating model: "
            f"{model_name}"
        )
        print("=" * 70)

        try:
            result = run_model_evaluation(
                dataset_key=dataset_key,
                model_name=model_name,
                retrieval_depth=(
                    args.retrieval_depth
                ),
                precision_k=(
                    args.precision_k
                ),
                recall_k=args.recall_k,
                ndcg_k=args.ndcg_k,
                candidate_count=(
                    args.candidate_count
                ),
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
                rrf_k=args.rrf_k,
                biomedical_weight=args.biomedical_weight,
                use_query_refinement=(
                    args.use_query_refinement
                ),
                feedback_docs=(
                    args.feedback_docs
                ),
                expansion_terms=(
                    args.expansion_terms
                ),
                query_batch_size=(
                    args.query_batch_size
                ),
            )

            all_results.append(
                result
            )

            print(
                f"Completed model: "
                f"{model_name}"
            )

            print(result)

        except KeyboardInterrupt:
            print()
            print(
                "Evaluation interrupted by user."
            )

            if all_results:
                save_results_to_csv(
                    results=all_results,
                    output_path=output_path,
                )

                print(
                    "Partial results saved to: "
                    f"{os.path.abspath(output_path)}"
                )

            raise SystemExit(130)

        except Exception as error:
            failure_count += 1

            print(
                f"Failed model: "
                f"{model_name}"
            )

            print(
                f"Error: {error}"
            )

            all_results.append(
                build_failure_result(
                    args=args,
                    dataset_key=dataset_key,
                    model_name=model_name,
                    error=error,
                )
            )

        # Save after every model so completed results survive a later
        # interruption or failure.
        save_results_to_csv(
            results=all_results,
            output_path=output_path,
        )

    print()
    print("=" * 70)
    print(
        "Evaluation results saved to: "
        f"{os.path.abspath(output_path)}"
    )
    print(
        "Models requested: "
        f"{len(args.models):,}"
    )
    print(
        "Models failed: "
        f"{failure_count:,}"
    )
    print("=" * 70)

    return failure_count


def main():
    args = parse_args()

    try:
        dataset_keys = resolve_dataset_keys(args)
        validate_args(args, dataset_keys)

    except ValueError as error:
        raise SystemExit(
            f"Invalid evaluation arguments: "
            f"{error}"
        ) from error

    total_failures = 0

    for dataset_key in dataset_keys:
        output_path = (
            args.output
            or get_default_output_path(
                dataset_key=dataset_key,
                retrieval_depth=(
                    args.retrieval_depth
                ),
                use_query_refinement=(
                    args.use_query_refinement
                ),
            )
        )

        total_failures += evaluate_dataset(
            args=args,
            dataset_key=dataset_key,
            output_path=output_path,
        )

    if total_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
