import argparse
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

import django
django.setup()

import faiss
import joblib
from django.conf import settings
from sentence_transformers import SentenceTransformer

from datasets.dataset_loader import DatasetLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and save FAISS embedding index for a dataset."
    )

    parser.add_argument(
        "--dataset",
        default="sample_dataset",
        help="Dataset key from dataset registry."
    )

    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size."
    )

    return parser.parse_args()


def get_index_dir(dataset_key: str) -> str:
    project_root = settings.BASE_DIR.parent

    index_dir = os.path.join(
        project_root,
        "indexes",
        dataset_key,
        "embedding"
    )

    os.makedirs(index_dir, exist_ok=True)

    return index_dir


def build_embedding_index(
    dataset_key: str,
    model_name: str,
    batch_size: int,
):
    print(f"Loading documents for dataset: {dataset_key}")

    documents = DatasetLoader.load_documents(dataset_key)

    texts = [
        f"{doc.get('title', '')}. {doc.get('text', '')}"
        for doc in documents
    ]

    print(f"Loaded {len(documents)} documents.")
    print(f"Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    print("Encoding documents...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
    ).astype("float32")

    dimension = embeddings.shape[1]

    print(f"Building FAISS IndexFlatIP with dimension: {dimension}")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_dir = get_index_dir(dataset_key)

    faiss_index_path = os.path.join(index_dir, "faiss.index")
    documents_path = os.path.join(index_dir, "documents.joblib")
    model_name_path = os.path.join(index_dir, "model_name.txt")

    print("Saving FAISS index and metadata...")

    faiss.write_index(index, faiss_index_path)
    joblib.dump(documents, documents_path)

    with open(model_name_path, "w", encoding="utf-8") as file:
        file.write(model_name)

    print(f"Embedding index saved to: {index_dir}")


def main():
    args = parse_args()

    build_embedding_index(
        dataset_key=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()