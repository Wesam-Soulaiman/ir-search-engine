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

from clustering.clustering_service import DocumentClusteringService

SUPPORTED_DATASETS = [
    "quora",
    "clinical_trials",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build document clusters from the saved SentenceTransformer "
            "embedding memmap."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "Number of clusters. Defaults to 30 for Quora and "
            "20 for Clinical Trials."
        ),
    )

    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=100000,
        help="Number of vectors sampled to train MiniBatchKMeans.",
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing clustering artifacts.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    service = DocumentClusteringService(
        dataset_key=args.dataset,
        project_root=PROJECT_ROOT,
    )

    result = service.build(
        n_clusters=args.n_clusters,
        train_sample_size=args.train_sample_size,
        train_batch_size=args.train_batch_size,
        predict_batch_size=args.predict_batch_size,
        random_seed=args.random_seed,
        max_iter=args.max_iter,
        n_init=args.n_init,
        overwrite=args.overwrite,
    )

    print()
    print("=" * 70)
    print("Document clustering built successfully.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
