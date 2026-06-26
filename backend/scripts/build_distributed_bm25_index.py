import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = None

for candidate in [
    SCRIPT_PATH.parent,
    *SCRIPT_PATH.parents,
]:
    if candidate.name == "backend":
        BACKEND_DIR = candidate
        break

if BACKEND_DIR is None:
    BACKEND_DIR = SCRIPT_PATH.parents[1]

PROJECT_ROOT = BACKEND_DIR.parent

for path in (
    PROJECT_ROOT,
    BACKEND_DIR,
):
    path_string = str(path)

    if path_string not in sys.path:
        sys.path.insert(
            0,
            path_string,
        )


os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

import django

django.setup()

from django.conf import settings

from document_store.repository import (
    DocumentStoreRepository,
)
from indexing.distributed_bm25_index import (
    DEFAULT_DISTRIBUTED_NUM_SHARDS,
    DEFAULT_DISTRIBUTED_RRF_K,
    DistributedBm25IndexBuilder,
    validate_num_shards,
)


SUPPORTED_DATASETS = [
    "quora",
    "clinical_trials",
]


def parse_optional_positive_integer(
    value: str,
):
    normalized = str(value).strip().lower()

    if normalized in {
        "none",
        "null",
        "unlimited",
        "all",
    }:
        return None

    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "The value must be greater than zero or 'none'."
        )

    return parsed


def parse_num_shards(
    value: str,
) -> int:
    try:
        return validate_num_shards(
            int(value)
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            str(error)
        ) from error


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a local sharded distributed BM25 index "
            "from the SQLite document store."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
    )

    parser.add_argument(
        "--num-shards",
        type=parse_num_shards,
        default=DEFAULT_DISTRIBUTED_NUM_SHARDS,
        help=(
            "Number of deterministic local shards to build. "
            f"Default: {DEFAULT_DISTRIBUTED_NUM_SHARDS}."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing distributed BM25 index."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help=(
            "Documents processed per shard statistics or postings batch."
        ),
    )

    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Minimum document frequency retained in each shard vocabulary.",
    )

    parser.add_argument(
        "--max-df",
        type=float,
        default=0.98,
        help=(
            "Maximum document frequency inside each shard. "
            "Values <= 1 are fractions; values > 1 are absolute counts."
        ),
    )

    parser.add_argument(
        "--max-features",
        type=parse_optional_positive_integer,
        default=None,
        help=(
            "Maximum vocabulary size per shard, or 'none' for no limit."
        ),
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.25,
        help=(
            "rank_bm25-compatible floor multiplier for negative IDF."
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_DISTRIBUTED_RRF_K,
        help=(
            "Default RRF k value recorded in the distributed manifest."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        repository = DocumentStoreRepository(
            settings.CORPUS_DATABASE_PATH
        )

        builder = DistributedBm25IndexBuilder(
            repository=repository,
            dataset_key=args.dataset,
            indexes_root=settings.INDEXES_DIR,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            epsilon=args.epsilon,
            rrf_k=args.rrf_k,
        )

        print("=" * 70)
        print("Distributed BM25 build")
        print("=" * 70)
        print(f"Dataset: {args.dataset}")
        print(
            "Index directory: "
            f"{builder.index_dir}"
        )
        print(
            "Number of shards: "
            f"{args.num_shards}"
        )
        print(
            "Batch size: "
            f"{args.batch_size:,}"
        )
        print(f"min_df: {args.min_df}")
        print(f"max_df: {args.max_df}")
        print(
            "max_features: "
            f"{args.max_features}"
        )
        print(f"epsilon: {args.epsilon}")
        print(f"rrf_k: {args.rrf_k}")
        print(f"Force rebuild: {args.force}")
        print("=" * 70)

        summary = builder.build(
            force=args.force
        )

        print()
        print("=" * 70)
        print(
            "Distributed BM25 index built successfully."
        )
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )
        print("=" * 70)

    except KeyboardInterrupt:
        print()
        print(
            "Distributed BM25 build interrupted. "
            "Rerun with --force to rebuild from scratch."
        )
        sys.exit(130)

    except Exception as error:
        print()
        print(
            "Distributed BM25 build failed: "
            f"{error}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
