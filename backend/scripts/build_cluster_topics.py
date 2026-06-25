import argparse
import json
import os
import sys
from pathlib import Path

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

from topic_detection.topic_detection_service import (
    ClusterTopicDetectionService,
)

SUPPORTED_DATASETS = [
    "quora",
    "clinical_trials",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build topic labels for previously-built document clusters."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
    )

    parser.add_argument(
        "--top-terms",
        type=int,
        default=10,
        help="Number of terms saved for each cluster.",
    )

    parser.add_argument(
        "--sample-docs-per-cluster",
        type=int,
        default=500,
        help=(
            "Maximum number of documents sampled from each cluster "
            "to infer topic terms."
        ),
    )

    parser.add_argument(
        "--fetch-batch-size",
        type=int,
        default=500,
        help="SQLite document fetch batch size.",
    )

    parser.add_argument(
        "--min-term-length",
        type=int,
        default=3,
        help="Ignore candidate topic terms shorter than this length.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing topic reports.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    service = ClusterTopicDetectionService(
        dataset_key=args.dataset,
        project_root=PROJECT_ROOT,
    )

    result = service.build(
        top_terms=args.top_terms,
        sample_docs_per_cluster=args.sample_docs_per_cluster,
        fetch_batch_size=args.fetch_batch_size,
        min_term_length=args.min_term_length,
        overwrite=args.overwrite,
    )

    print()
    print("=" * 70)
    print("Cluster topic detection built successfully.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
