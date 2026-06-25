from pathlib import Path
from typing import Optional

from django.conf import settings

from retrieval.embedding_service import (
    EmbeddingIndexError,
    EmbeddingRetrievalService,
)


class BiomedicalEmbeddingService(EmbeddingRetrievalService):
    """
    Biomedical PubMedBERT SentenceTransformer + FAISS retrieval for
    Clinical Trials.

    This service uses a separate saved index directory from the existing
    MiniLM embedding service so old embedding behavior and artifacts remain
    untouched.
    """

    SUPPORTED_DATASET = "clinical_trials"
    INDEX_NAME = "biomedical_embedding"
    DISPLAY_NAME = "Biomedical PubMedBERT"

    def __init__(
        self,
        dataset_key: str = SUPPORTED_DATASET,
        use_saved_index: bool = True,
        device: str = "auto",
        model_path: Optional[str | Path] = None,
    ):
        normalized_dataset = str(dataset_key).strip()

        if normalized_dataset != self.SUPPORTED_DATASET:
            raise ValueError(
                "Biomedical embedding retrieval is only supported "
                "for the clinical_trials dataset."
            )

        resolved_model_path = (
            model_path
            if model_path is not None
            else getattr(
                settings,
                "BIOMEDICAL_EMBEDDING_MODEL_PATH",
                (
                    settings.BASE_DIR.parent
                    / "artifacts"
                    / "models"
                    / "biomedical-embedding"
                ),
            )
        )

        index_dir = getattr(
            settings,
            "BIOMEDICAL_EMBEDDING_INDEX_DIR",
            (
                settings.BASE_DIR.parent
                / "indexes"
                / self.SUPPORTED_DATASET
                / self.INDEX_NAME
            ),
        )

        super().__init__(
            dataset_key=normalized_dataset,
            use_saved_index=use_saved_index,
            device=device,
            model_path=resolved_model_path,
            index_name=self.INDEX_NAME,
            index_dir=index_dir,
        )

    def _validate_manifest(self):
        super()._validate_manifest()

        model_name = str(
            self.manifest.get("model_name", "")
        ).strip()
        expected_model_name = str(
            settings.BIOMEDICAL_EMBEDDING_MODEL_NAME
        ).strip()

        if model_name != expected_model_name:
            raise EmbeddingIndexError(
                "Biomedical embedding manifest model_name does not "
                "match the configured biomedical model. "
                f"Expected {expected_model_name}, got {model_name or 'missing'}."
            )

    def _load_model(self, model_path: Path):
        safetensors_path = model_path / "model.safetensors"

        if not safetensors_path.is_file():
            raise FileNotFoundError(
                "Biomedical embedding model is incomplete: "
                f"model.safetensors was not found at {safetensors_path}."
            )

        return super()._load_model(
            model_path
        )

    def _hydrate_faiss_results(self, ranked_documents):
        results = super()._hydrate_faiss_results(
            ranked_documents
        )

        for result in results:
            result["document_source"] = (
                "sqlite_document_store"
            )

        return results
