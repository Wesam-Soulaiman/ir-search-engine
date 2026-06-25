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

from django.conf import settings

from document_store.repository import (
    DocumentStoreRepository,
)
from indexing.scalable_embedding_index import (
    ScalableEmbeddingIndexBuilder,
)


SUPPORTED_DATASETS = [
    "quora",
    "clinical_trials",
]


def parse_optional_positive_integer(value: str):
    normalized = str(value).strip().lower()

    if normalized in {
        "none",
        "null",
        "default",
        "auto",
    }:
        return None

    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "The value must be greater than zero or 'auto'."
        )

    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete SentenceTransformer + FAISS "
            "embedding index from the SQLite document store."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
    )

    parser.add_argument(
        "--model-path",
        default=str(
            (
                Path("..")
                / "artifacts"
                / "models"
                / "multi-qa-MiniLM-L6-cos-v1"
            ).resolve()
        ),
        help=(
            "Local SentenceTransformer model directory. "
            "The default points to artifacts/models/"
            "multi-qa-MiniLM-L6-cos-v1."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for encoding documents.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Texts encoded per SentenceTransformer batch.",
    )

    parser.add_argument(
        "--add-batch-size",
        type=int,
        default=8192,
        help="Vectors added to FAISS per batch.",
    )

    parser.add_argument(
        "--text-max-characters",
        type=parse_optional_positive_integer,
        default=None,
        help=(
            "Maximum characters passed to the embedding model "
            "per document, or 'auto' for dataset defaults."
        ),
    )

    parser.add_argument(
        "--index-type",
        choices=["ivfpq", "hnsw", "flat"],
        default="ivfpq",
        help=(
            "FAISS index type. Use ivfpq for low RAM, hnsw "
            "for higher accuracy when RAM is enough, flat for exact search."
        ),
    )

    parser.add_argument(
        "--nlist",
        type=parse_optional_positive_integer,
        default=None,
        help="IVF cluster count, or 'auto' for dataset defaults.",
    )

    parser.add_argument(
        "--pq-m",
        type=int,
        default=48,
        help="Number of PQ subquantizers. Must divide embedding dimension.",
    )

    parser.add_argument(
        "--pq-nbits",
        type=int,
        default=8,
        help="Bits per PQ code.",
    )

    parser.add_argument(
        "--nprobe",
        type=int,
        default=32,
        help="Number of IVF clusters searched at query time.",
    )

    parser.add_argument(
        "--train-sample-size",
        type=parse_optional_positive_integer,
        default=None,
        help="Training vectors sampled for IVF-PQ, or 'auto'.",
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

        builder = ScalableEmbeddingIndexBuilder(
            repository=repository,
            dataset_key=args.dataset,
            indexes_root=settings.INDEXES_DIR,
            model_path=args.model_path,
            batch_size=args.batch_size,
            add_batch_size=args.add_batch_size,
            device=args.device,
            text_max_characters=args.text_max_characters,
            index_type=args.index_type,
            nlist=args.nlist,
            pq_m=args.pq_m,
            pq_nbits=args.pq_nbits,
            nprobe=args.nprobe,
            train_sample_size=args.train_sample_size,
        )

        print("=" * 70)
        print(f"Dataset: {args.dataset}")
        print(f"Index directory: {builder.index_dir}")
        print(f"Model path: {builder.model_path}")
        print(f"Device: {builder.device}")
        print(f"Encoding batch size: {builder.batch_size}")
        print(f"FAISS add batch size: {builder.add_batch_size}")
        print(f"Text max characters: {builder.text_max_characters}")
        print(f"Index type: {builder.index_type}")
        print(f"nlist: {builder.nlist}")
        print(f"pq_m: {builder.pq_m}")
        print(f"pq_nbits: {builder.pq_nbits}")
        print(f"nprobe: {builder.nprobe}")
        print(f"Train sample size: {builder.train_sample_size}")
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
            "Scalable embedding index built successfully."
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
            "Embedding build interrupted. Run the same command "
            "again with --resume."
        )
        sys.exit(130)

    except Exception as error:
        print()
        print(f"Embedding build failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
