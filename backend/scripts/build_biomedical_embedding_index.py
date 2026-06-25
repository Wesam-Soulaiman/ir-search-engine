import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import faiss
import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


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


SUPPORTED_DATASETS = [
    "clinical_trials",
]

DISPLAY_NAME = "Biomedical PubMedBERT"
DEFAULT_TEXT_MAX_CHARACTERS = 2000


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
            "Build the Clinical Trials biomedical PubMedBERT "
            "SentenceTransformer + FAISS index."
        )
    )

    parser.add_argument(
        "--dataset",
        default="clinical_trials",
        choices=SUPPORTED_DATASETS,
    )

    parser.add_argument(
        "--model-path",
        default=str(
            settings.BIOMEDICAL_EMBEDDING_MODEL_PATH
        ),
        help=(
            "Local biomedical SentenceTransformer directory. "
            "Download it first with download_biomedical_embedding_model.py."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            settings.BIOMEDICAL_EMBEDDING_INDEX_DIR
        ),
        help=(
            "Directory where faiss.index, doc_ids.joblib, "
            "embedding_config.json, and manifest.json are written."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for document encoding.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Texts encoded per SentenceTransformer batch.",
    )

    parser.add_argument(
        "--text-max-characters",
        type=parse_optional_positive_integer,
        default=DEFAULT_TEXT_MAX_CHARACTERS,
        help=(
            "Maximum characters passed to the embedding model "
            "per document, or 'auto' for no script-level cap."
        ),
    )

    parser.add_argument(
        "--limit",
        type=parse_optional_positive_integer,
        default=None,
        help="Optional document limit for smoke-test indexes.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing biomedical index directory before building.",
    )

    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    normalized = str(requested_device).strip().lower()

    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if normalized in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() "
                "returned False."
            )
        return "cuda"

    if normalized == "cpu":
        return "cpu"

    raise ValueError(
        "device must be one of: auto, cpu, cuda."
    )


def get_model_dimension(model: SentenceTransformer) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())

    return int(model.get_sentence_embedding_dimension())


def document_to_text(
    row: Any,
    text_max_characters: Optional[int],
) -> str:
    title = str(row["title"] or "").strip()
    raw_text = str(row["raw_text"] or "").strip()

    if title and raw_text:
        if raw_text.startswith(title):
            text = raw_text
        else:
            text = f"{title}. {raw_text}"
    else:
        text = title or raw_text

    text = " ".join(
        text.split()
    )

    if (
        text_max_characters is not None
        and len(text) > text_max_characters
    ):
        text = text[:text_max_characters]

    return text


def validate_local_model_path(
    model_path: Path,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Biomedical embedding model path does not exist: {model_path}"
        )

    safetensors_path = model_path / "model.safetensors"

    if not safetensors_path.is_file():
        raise FileNotFoundError(
            "Biomedical embedding model is incomplete: "
            f"model.safetensors was not found at {safetensors_path}."
        )


def build_index(args) -> dict:
    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be greater than zero."
        )

    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    validate_local_model_path(
        model_path
    )

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                "Biomedical index directory already exists and is not empty: "
                f"{output_dir}. Use --overwrite to rebuild it."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository = DocumentStoreRepository(
        settings.CORPUS_DATABASE_PATH
    )
    repository.initialize()

    dataset = repository.get_dataset(
        args.dataset
    )

    if dataset is None:
        raise ValueError(
            f"Dataset '{args.dataset}' is not present in the document store. "
            "Run the document-store ingestion script first."
        )

    counts = repository.get_dataset_counts(
        args.dataset
    )
    available_documents = int(
        counts["documents"]
    )
    document_count = (
        min(
            available_documents,
            int(args.limit),
        )
        if args.limit is not None
        else available_documents
    )

    if document_count <= 0:
        raise ValueError(
            f"Dataset '{args.dataset}' contains no documents."
        )

    device = resolve_device(
        args.device
    )

    print(
        "Loading biomedical embedding model from "
        f"{model_path} on {device}..."
    )

    model = SentenceTransformer(
        str(model_path),
        device=device,
        local_files_only=True,
    )

    dimension = get_model_dimension(
        model
    )
    index = faiss.IndexFlatIP(
        dimension
    )
    doc_ids: List[str] = []

    start_time = time.perf_counter()
    processed = 0

    with repository.connection() as connection:
        cursor = connection.execute(
            """
            SELECT
                doc_id,
                title,
                raw_text
            FROM documents
            WHERE dataset_key = ?
            ORDER BY rowid
            """,
            (args.dataset,),
        )

        while processed < document_count:
            fetch_size = min(
                args.batch_size,
                document_count - processed,
            )
            rows = cursor.fetchmany(
                fetch_size
            )

            if not rows:
                break

            texts = [
                document_to_text(
                    row,
                    args.text_max_characters,
                )
                for row in rows
            ]

            embeddings = model.encode(
                texts,
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(
                "float32",
                copy=False,
            )

            if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
                raise RuntimeError(
                    "Encoded document vectors have an unexpected shape. "
                    f"Expected dimension {dimension}, got {embeddings.shape}."
                )

            index.add(
                np.ascontiguousarray(
                    embeddings,
                    dtype="float32",
                )
            )

            doc_ids.extend(
                str(row["doc_id"])
                for row in rows
            )

            processed += len(rows)

            elapsed = max(
                time.perf_counter() - start_time,
                1e-9,
            )

            print(
                f"Embedded {processed:,}/{document_count:,} "
                f"documents ({processed / elapsed:.2f} docs/s)."
            )

    if processed != document_count:
        raise RuntimeError(
            "Biomedical embedding build ended before all expected "
            "documents were processed."
        )

    if int(index.ntotal) != document_count:
        raise RuntimeError(
            "FAISS vector count does not match document count."
        )

    faiss_index_path = output_dir / "faiss.index"
    doc_ids_path = output_dir / "doc_ids.joblib"
    embedding_config_path = output_dir / "embedding_config.json"
    manifest_path = output_dir / "manifest.json"
    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    faiss.write_index(
        index,
        str(faiss_index_path),
    )

    joblib.dump(
        doc_ids,
        doc_ids_path,
    )

    embedding_config = {
        "index_version": 1,
        "dataset": args.dataset,
        "document_count": document_count,
        "embedding_dimension": dimension,
        "model_path": str(model_path),
        "model_name": settings.BIOMEDICAL_EMBEDDING_MODEL_NAME,
        "display_name": DISPLAY_NAME,
        "text_strategy": "title_plus_raw_text_prefix",
        "text_max_characters": args.text_max_characters,
        "normalized_embeddings": True,
        "vectors_are_normalized": True,
        "similarity": "cosine_via_inner_product",
        "created_at": created_at,
    }

    embedding_config_path.write_text(
        json.dumps(
            embedding_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "index_version": 1,
        "index_type": "sentence_transformer_faiss",
        "faiss_index_type": "flat",
        "dataset": args.dataset,
        "dataset_metadata": dataset,
        "document_count": document_count,
        "embedding_dimension": dimension,
        "model_name": settings.BIOMEDICAL_EMBEDDING_MODEL_NAME,
        "model_path": str(model_path),
        "display_name": DISPLAY_NAME,
        "device_used_for_build": device,
        "text_max_characters": args.text_max_characters,
        "normalized_embeddings": True,
        "vectors_are_normalized": True,
        "similarity": "cosine_via_inner_product",
        "faiss_ntotal": int(index.ntotal),
        "created_at": created_at,
        "created_at_unix": time.time(),
        "files": {
            "faiss_index": faiss_index_path.name,
            "doc_ids": doc_ids_path.name,
            "embedding_config": embedding_config_path.name,
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "dataset": args.dataset,
        "index_dir": str(output_dir),
        "document_count": document_count,
        "embedding_dimension": dimension,
        "model_name": settings.BIOMEDICAL_EMBEDDING_MODEL_NAME,
        "faiss_ntotal": int(index.ntotal),
        "model_path": str(model_path),
        "created_at": created_at,
        "index_size_mb": round(
            faiss_index_path.stat().st_size / (1024 * 1024),
            2,
        ),
        "doc_ids_size_mb": round(
            doc_ids_path.stat().st_size / (1024 * 1024),
            2,
        ),
    }


def main():
    args = parse_args()

    try:
        summary = build_index(
            args
        )

    except KeyboardInterrupt:
        print()
        print(
            "Biomedical embedding build interrupted."
        )
        raise SystemExit(130)

    except Exception as error:
        print()
        print(
            f"Biomedical embedding build failed: {error}"
        )
        raise SystemExit(1) from error

    print()
    print(
        "Biomedical PubMedBERT embedding index built successfully."
    )
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
