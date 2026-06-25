import argparse
import json
import os

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

import django

django.setup()

import joblib
from django.conf import settings

from datasets.dataset_loader import DatasetLoader
from preprocessing.preprocessing_service import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build and save dataset-aware BM25 index data."
        )
    )

    parser.add_argument(
        "--dataset",
        default="sample_dataset",
        help="Dataset key from dataset registry.",
    )

    return parser.parse_args()


def get_index_dir(dataset_key: str) -> str:
    project_root = settings.BASE_DIR.parent

    index_dir = os.path.join(
        project_root,
        "indexes",
        dataset_key,
        "bm25",
    )

    os.makedirs(
        index_dir,
        exist_ok=True,
    )

    return index_dir


def build_bm25_index(dataset_key: str):
    print(
        f"Loading documents for dataset: "
        f"{dataset_key}"
    )

    documents = DatasetLoader.load_documents(
        dataset_key
    )

    if not documents:
        raise ValueError(
            f"No documents found for dataset "
            f"'{dataset_key}'."
        )

    preprocessor = TextPreprocessor(
        dataset_key=dataset_key
    )

    print(f"Loaded {len(documents):,} documents.")

    print(
        "Active preprocessing configuration:"
    )
    print(
        json.dumps(
            preprocessor.get_configuration(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("Tokenizing documents for BM25...")

    tokenized_corpus = []

    for document_index, document in enumerate(
        documents,
        start=1,
    ):
        search_text = (
            f"{document.get('title', '')} "
            f"{document.get('text', '')}"
        ).strip()

        tokenized_corpus.append(
            preprocessor.preprocess_tokens(
                search_text
            )
        )

        if document_index % 10_000 == 0:
            print(
                "Tokenized "
                f"{document_index:,} documents...",
                flush=True,
            )

    index_dir = get_index_dir(dataset_key)

    documents_path = os.path.join(
        index_dir,
        "documents.joblib",
    )

    tokenized_corpus_path = os.path.join(
        index_dir,
        "tokenized_corpus.joblib",
    )

    preprocessing_config_path = os.path.join(
        index_dir,
        "preprocessing_config.json",
    )

    metadata_path = os.path.join(
        index_dir,
        "metadata.json",
    )

    print("Saving BM25 index data...")

    joblib.dump(
        documents,
        documents_path,
    )

    joblib.dump(
        tokenized_corpus,
        tokenized_corpus_path,
    )

    with open(
        preprocessing_config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            preprocessor.get_configuration(),
            file,
            ensure_ascii=False,
            indent=2,
        )

    token_count = sum(
        len(tokens)
        for tokens in tokenized_corpus
    )

    empty_document_count = sum(
        1
        for tokens in tokenized_corpus
        if not tokens
    )

    average_document_length = (
        token_count / len(tokenized_corpus)
        if tokenized_corpus
        else 0.0
    )

    metadata = {
        "dataset": dataset_key,
        "document_count": len(documents),
        "tokenized_document_count": len(
            tokenized_corpus
        ),
        "token_count": token_count,
        "empty_document_count": (
            empty_document_count
        ),
        "average_document_length": round(
            average_document_length,
            6,
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"BM25 index data saved to: {index_dir}"
    )

    print(
        "Average processed document length: "
        f"{average_document_length:.2f} tokens"
    )

    print(
        "Empty processed documents: "
        f"{empty_document_count:,}"
    )


def main():
    args = parse_args()

    build_bm25_index(
        dataset_key=args.dataset
    )


if __name__ == "__main__":
    main()