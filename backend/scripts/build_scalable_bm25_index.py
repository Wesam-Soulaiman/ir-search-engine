import argparse
import json
import os
import sys

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
from indexing.scalable_bm25_index import (
    ScalableBm25IndexBuilder,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete disk-backed BM25 inverted "
            "index from the SQLite document store."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help=(
            "Documents processed per statistics or postings batch."
        ),
    )

    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Minimum document frequency retained in the vocabulary.",
    )

    parser.add_argument(
        "--max-df",
        type=float,
        default=0.98,
        help=(
            "Maximum document frequency. Values <= 1 are corpus "
            "fractions; values > 1 are absolute counts."
        ),
    )

    parser.add_argument(
        "--max-features",
        type=parse_optional_positive_integer,
        default=None,
        help=(
            "Maximum vocabulary size, or 'none' for no limit."
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

    operation = parser.add_mutually_exclusive_group()

    operation.add_argument(
        "--overwrite",
        action="store_true",
    )

    operation.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        repository = DocumentStoreRepository(
            settings.CORPUS_DATABASE_PATH
        )

        builder = ScalableBm25IndexBuilder(
            repository=repository,
            dataset_key=args.dataset,
            indexes_root=settings.INDEXES_DIR,
            batch_size=args.batch_size,
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            epsilon=args.epsilon,
        )

        print("=" * 70)
        print(f"Dataset: {args.dataset}")
        print(f"Index directory: {builder.index_dir}")
        print(f"Batch size: {args.batch_size:,}")
        print(f"min_df: {args.min_df}")
        print(f"max_df: {args.max_df}")
        print(f"max_features: {args.max_features}")
        print(f"epsilon: {args.epsilon}")
        print(f"Resume: {args.resume}")
        print(f"Overwrite: {args.overwrite}")
        print("=" * 70)

        summary = builder.build(
            overwrite=args.overwrite,
            resume=args.resume,
        )

        print()
        print("=" * 70)
        print(
            "Scalable BM25 index built successfully."
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
            "BM25 build interrupted. Run the same command "
            "again with --resume."
        )
        sys.exit(130)

    except Exception as error:
        print()
        print(f"BM25 build failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
