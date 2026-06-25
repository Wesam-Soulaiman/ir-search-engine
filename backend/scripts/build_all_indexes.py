import argparse
import os
import time

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

import django
django.setup()

from scripts.build_bm25_index import build_bm25_index
from scripts.build_embedding_index import build_embedding_index
from scripts.build_tfidf_index import build_tfidf_index


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build all retrieval indexes for a dataset."
    )

    parser.add_argument(
        "--dataset",
        default="sample_dataset",
        help="Dataset key from dataset registry."
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model used for the embedding index."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size."
    )

    parser.add_argument(
        "--skip-tfidf",
        action="store_true",
        help="Skip TF-IDF index generation."
    )

    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip BM25 index generation."
    )

    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding/FAISS index generation."
    )

    return parser.parse_args()


def run_step(step_name: str, build_function, *args, **kwargs):
    print()
    print("=" * 70)
    print(f"Starting: {step_name}")
    print("=" * 70)

    start_time = time.perf_counter()

    try:
        build_function(*args, **kwargs)
    except Exception as error:
        elapsed = time.perf_counter() - start_time

        print(f"Failed: {step_name}")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Error: {error}")

        raise

    elapsed = time.perf_counter() - start_time

    print(f"Completed: {step_name}")
    print(f"Elapsed time: {elapsed:.2f} seconds")


def main():
    args = parse_args()

    total_start_time = time.perf_counter()

    print(f"Building indexes for dataset: {args.dataset}")

    if not args.skip_tfidf:
        run_step(
            step_name="TF-IDF index",
            build_function=build_tfidf_index,
            dataset_key=args.dataset,
        )
    else:
        print("Skipping TF-IDF index.")

    if not args.skip_bm25:
        run_step(
            step_name="BM25 index",
            build_function=build_bm25_index,
            dataset_key=args.dataset,
        )
    else:
        print("Skipping BM25 index.")

    if not args.skip_embedding:
        run_step(
            step_name="Embedding / FAISS index",
            build_function=build_embedding_index,
            dataset_key=args.dataset,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
    else:
        print("Skipping embedding / FAISS index.")

    total_elapsed = time.perf_counter() - total_start_time

    print()
    print("=" * 70)
    print("All requested indexes were built successfully.")
    print(f"Dataset: {args.dataset}")
    print(f"Total elapsed time: {total_elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()