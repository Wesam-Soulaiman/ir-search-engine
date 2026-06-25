import argparse
import json
import os
import shutil
import sys
import time
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
from huggingface_hub import snapshot_download


MODEL_SAFETENSORS_FILENAME = "model.safetensors"
IGNORED_DOWNLOAD_PATTERNS = ["*.bin"]
DISPLAY_NAME = "Biomedical PubMedBERT"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download and save the biomedical SentenceTransformer "
            "model used by biomedical embedding retrieval."
        )
    )

    parser.add_argument(
        "--model-name",
        default=settings.BIOMEDICAL_EMBEDDING_MODEL_NAME,
        help="SentenceTransformer model name on Hugging Face.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(settings.BIOMEDICAL_EMBEDDING_MODEL_PATH),
        help="Local directory where the model will be saved.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the local model directory if it already exists.",
    )

    return parser.parse_args()


def find_bin_files(
    directory: Path,
) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*.bin")
        if path.is_file()
    )


def validate_local_model_directory(
    model_dir: Path,
):
    safetensors_path = (
        model_dir
        / MODEL_SAFETENSORS_FILENAME
    )

    if not safetensors_path.is_file():
        raise RuntimeError(
            "Incomplete biomedical model directory: "
            f"{MODEL_SAFETENSORS_FILENAME} was not found at "
            f"{safetensors_path}."
        )

    bin_files = find_bin_files(
        model_dir
    )

    if bin_files:
        examples = [
            str(path)
            for path in bin_files[:5]
        ]
        raise RuntimeError(
            "Unsafe biomedical model directory: .bin model files are "
            f"not allowed. Examples: {examples}"
        )


def prepare_output_directory(
    output_dir: Path,
    force: bool,
) -> bool:
    if output_dir.exists() and any(output_dir.iterdir()):
        if force:
            shutil.rmtree(
                output_dir
            )
        else:
            validate_local_model_directory(
                output_dir
            )
            return False

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return True


def download_model_snapshot(
    model_name: str,
    output_dir: Path,
) -> str:
    return snapshot_download(
        repo_id=model_name,
        local_dir=str(output_dir),
        ignore_patterns=IGNORED_DOWNLOAD_PATTERNS,
    )


def write_download_manifest(
    model_name: str,
    output_dir: Path,
):
    manifest = {
        "model_name": model_name,
        "display_name": DISPLAY_NAME,
        "output_dir": str(output_dir),
        "safetensors_required": True,
        "ignored_download_patterns": IGNORED_DOWNLOAD_PATTERNS,
        "saved_at_unix": time.time(),
    }

    (
        output_dir
        / "ir_model_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    should_download = prepare_output_directory(
        output_dir=output_dir,
        force=args.force,
    )

    if not should_download:
        print(
            "Biomedical embedding model already exists and is complete: "
            f"{output_dir}"
        )
        print(
            f"Final local path: {output_dir}"
        )
        return

    print(
        "Downloading biomedical embedding model: "
        f"{args.model_name}"
    )

    download_model_snapshot(
        model_name=args.model_name,
        output_dir=output_dir,
    )

    validate_local_model_directory(
        output_dir
    )

    write_download_manifest(
        model_name=args.model_name,
        output_dir=output_dir,
    )

    print(
        "Biomedical embedding model saved to: "
        f"{output_dir}"
    )
    print(
        f"Final local path: {output_dir}"
    )


if __name__ == "__main__":
    main()
