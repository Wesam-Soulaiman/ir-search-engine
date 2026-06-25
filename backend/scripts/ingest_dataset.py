import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

import django

django.setup()

from django.conf import settings

from document_store.importer import (
    DatasetIngestionService,
)
from document_store.repository import (
    DocumentStoreRepository,
)


SUPPORTED_DATASETS = [
    "quora",
    "clinical_trials",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream a prepared IR dataset into the "
            "offline SQLite document store."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset to import.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help=(
            "Records committed per transaction. "
            "Default: 1000."
        ),
    )

    parser.add_argument(
        "--database-path",
        default=None,
        help=(
            "Optional SQLite path. Defaults to "
            "settings.CORPUS_DATABASE_PATH."
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Optional checkpoint directory. By default, "
            "a checkpoints directory is created beside "
            "the corpus database."
        ),
    )

    parser.add_argument(
        "--allow-development-sample",
        action="store_true",
        help=(
            "Allow importing a dataset created with limits "
            "or --evaluation-sample. Use only for testing."
        ),
    )

    operation_group = (
        parser.add_mutually_exclusive_group()
    )

    operation_group.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete the existing dataset from the store "
            "and begin again."
        ),
    )

    operation_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted import using its "
            "saved byte offsets."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size <= 0:
        print(
            "Dataset ingestion failed: "
            "--batch-size must be greater than zero."
        )
        sys.exit(1)

    database_path = (
        Path(args.database_path)
        .expanduser()
        .resolve()
        if args.database_path
        else settings.CORPUS_DATABASE_PATH
    )

    try:
        repository = DocumentStoreRepository(
            database_path
        )

        service = DatasetIngestionService(
            repository=repository,
            dataset_key=args.dataset,
            batch_size=args.batch_size,
            checkpoint_dir=(
                args.checkpoint_dir
            ),
            allow_development_sample=(
                args.allow_development_sample
            ),
        )

        print("=" * 70)
        print(f"Dataset: {args.dataset}")
        print(f"Database: {database_path}")
        print(f"Batch size: {args.batch_size:,}")
        print(f"Resume: {args.resume}")
        print(f"Overwrite: {args.overwrite}")
        print(
            "Allow development sample: "
            f"{args.allow_development_sample}"
        )
        print("=" * 70)

        summary = service.ingest(
            overwrite=args.overwrite,
            resume=args.resume,
        )

        print()
        print("=" * 70)
        print("Dataset ingestion completed successfully.")
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
            "Import interrupted. Progress was saved. "
            "Run the same command with --resume."
        )
        sys.exit(130)

    except Exception as error:
        print()
        print(f"Dataset ingestion failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
