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
from indexing.scalable_tfidf_index import (
    ScalableTfidfIndexBuilder,
)


SUPPORTED_DATASETS = [
    "quora",
    "clinical_trials",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete sharded TF-IDF index "
            "from the SQLite document store."
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
            "Document-frequency batch size."
        ),
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help=(
            "Documents stored in each sparse matrix shard."
        ),
    )

    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-df",
        type=float,
        default=0.98,
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--no-sublinear-tf",
        action="store_true",
    )

    operation = (
        parser.add_mutually_exclusive_group()
    )

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

        builder = ScalableTfidfIndexBuilder(
            repository=repository,
            dataset_key=args.dataset,
            indexes_root=settings.INDEXES_DIR,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            sublinear_tf=(
                not args.no_sublinear_tf
            ),
        )

        print("=" * 70)
        print(f"Dataset: {args.dataset}")
        print(f"Index directory: {builder.index_dir}")
        print(f"Batch size: {args.batch_size:,}")
        print(f"Shard size: {args.shard_size:,}")
        print(f"min_df: {args.min_df}")
        print(f"max_df: {args.max_df}")
        print(
            f"max_features: "
            f"{args.max_features:,}"
        )
        print(
            "Sublinear TF: "
            f"{not args.no_sublinear_tf}"
        )
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
            "Scalable TF-IDF index built successfully."
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
            "TF-IDF build interrupted. "
            "Run the command again with --resume."
        )

        sys.exit(130)

    except Exception as error:
        print()
        print(
            f"TF-IDF build failed: {error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()