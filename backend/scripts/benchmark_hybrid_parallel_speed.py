import argparse
import os
import sys
from pathlib import Path
from time import perf_counter


SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

for path in (BACKEND_DIR, PROJECT_ROOT):
    path_string = str(path)

    if path_string not in sys.path:
        sys.path.insert(0, path_string)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

import django

django.setup()

from datasets.dataset_loader import DatasetLoader
from retrieval.hybrid_parallel_service import (
    HybridParallelRetrievalService,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Hybrid Parallel single-query search "
            "against batch search."
        )
    )

    parser.add_argument(
        "--dataset",
        default="quora",
        help="Dataset key from the dataset registry.",
    )

    parser.add_argument(
        "--query-count",
        type=int,
        default=128,
        help="Number of queries to benchmark.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Returned result count per query.",
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=100,
        help="Hybrid Parallel candidate count per sub-model.",
    )

    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=64,
        help="Number of queries processed per batch.",
    )

    return parser.parse_args()


def validate_args(args):
    if args.query_count <= 0:
        raise ValueError(
            "query-count must be greater than zero."
        )

    if args.top_k <= 0:
        raise ValueError(
            "top-k must be greater than zero."
        )

    if args.candidate_count <= 0:
        raise ValueError(
            "candidate-count must be greater than zero."
        )

    if args.query_batch_size <= 0:
        raise ValueError(
            "query-batch-size must be greater than zero."
        )


def load_queries(
    dataset_key: str,
    query_count: int,
):
    query_rows = DatasetLoader.load_queries(
        dataset_key
    )

    queries = [
        str(row.get("query", "")).strip()
        for row in query_rows
        if str(row.get("query", "")).strip()
    ]

    return queries[:query_count]


def run_single_query_loop(
    service: HybridParallelRetrievalService,
    queries,
    top_k: int,
):
    start = perf_counter()

    results = [
        service.search(
            query=query,
            top_k=top_k,
        )
        for query in queries
    ]

    return perf_counter() - start, results


def run_batch_search(
    service: HybridParallelRetrievalService,
    queries,
    top_k: int,
    query_batch_size: int,
):
    start = perf_counter()

    results = service.search_batch(
        queries=queries,
        top_k=top_k,
        query_batch_size=query_batch_size,
    )

    return perf_counter() - start, results


def main():
    args = parse_args()
    validate_args(args)

    queries = load_queries(
        dataset_key=args.dataset,
        query_count=args.query_count,
    )

    if not queries:
        raise ValueError(
            f"No benchmark queries found for dataset '{args.dataset}'."
        )

    service = HybridParallelRetrievalService(
        dataset_key=args.dataset,
        candidate_count=args.candidate_count,
    )

    single_seconds, single_results = run_single_query_loop(
        service=service,
        queries=queries,
        top_k=args.top_k,
    )

    batch_seconds, batch_results = run_batch_search(
        service=service,
        queries=queries,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size,
    )

    speedup = (
        single_seconds / batch_seconds
        if batch_seconds > 0
        else float("inf")
    )

    print(f"Dataset: {args.dataset}")
    print(f"Queries: {len(queries)}")
    print(f"top_k: {args.top_k}")
    print(f"candidate_count: {args.candidate_count}")
    print(f"query_batch_size: {args.query_batch_size}")
    print(f"single_query_seconds: {single_seconds:.3f}")
    print(f"batch_seconds: {batch_seconds:.3f}")
    print(f"speedup: {speedup:.2f}x")
    print(f"single_query_result_lists: {len(single_results)}")
    print(f"batch_result_lists: {len(batch_results)}")


if __name__ == "__main__":
    main()
