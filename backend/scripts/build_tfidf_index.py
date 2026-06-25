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
import numpy as np
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets.dataset_loader import DatasetLoader
from preprocessing.preprocessing_service import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build and save a dataset-aware TF-IDF index."
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
        "tfidf",
    )

    os.makedirs(
        index_dir,
        exist_ok=True,
    )

    return index_dir


def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        dtype=np.float32,
    )


def build_tfidf_index(dataset_key: str):
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

    print("Preprocessing documents for TF-IDF...")

    processed_texts = []

    for document_index, document in enumerate(
        documents,
        start=1,
    ):
        search_text = (
            f"{document.get('title', '')} "
            f"{document.get('text', '')}"
        ).strip()

        processed_texts.append(
            preprocessor.preprocess(search_text)
        )

        if document_index % 10_000 == 0:
            print(
                "Preprocessed "
                f"{document_index:,} documents...",
                flush=True,
            )

    print(
        "Building TF-IDF vectorizer and matrix..."
    )

    vectorizer = create_vectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        processed_texts
    )

    index_dir = get_index_dir(dataset_key)

    vectorizer_path = os.path.join(
        index_dir,
        "vectorizer.joblib",
    )

    matrix_path = os.path.join(
        index_dir,
        "matrix.joblib",
    )

    documents_path = os.path.join(
        index_dir,
        "documents.joblib",
    )

    preprocessing_config_path = os.path.join(
        index_dir,
        "preprocessing_config.json",
    )

    metadata_path = os.path.join(
        index_dir,
        "metadata.json",
    )

    print("Saving TF-IDF index files...")

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    joblib.dump(
        tfidf_matrix,
        matrix_path,
    )

    joblib.dump(
        documents,
        documents_path,
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

    metadata = {
        "dataset": dataset_key,
        "document_count": len(documents),
        "matrix_rows": int(
            tfidf_matrix.shape[0]
        ),
        "matrix_columns": int(
            tfidf_matrix.shape[1]
        ),
        "vocabulary_size": len(
            vectorizer.vocabulary_
        ),
        "dtype": str(tfidf_matrix.dtype),
        "vectorizer": {
            "tokenizer": "str.split",
            "lowercase": False,
            "token_pattern": None,
        },
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
        f"TF-IDF index saved to: {index_dir}"
    )

    print(
        "TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    print(
        "Vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )


def main():
    args = parse_args()

    build_tfidf_index(
        dataset_key=args.dataset
    )


if __name__ == "__main__":
    main()