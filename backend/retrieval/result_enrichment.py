from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from django.conf import settings

from document_store.repository import (
    DocumentStoreError,
    DocumentStoreRepository,
)


DOCUMENT_STORE_DATASETS = {
    "quora",
    "clinical_trials",
}


@lru_cache(maxsize=1)
def get_document_store_repository(
) -> DocumentStoreRepository:
    """
    Return the shared offline document-store repository.

    The connection itself is not cached. The repository opens a short-lived
    SQLite connection for each operation, which is safer for Django requests.
    """
    repository = DocumentStoreRepository(
        settings.CORPUS_DATABASE_PATH
    )

    repository.initialize()

    return repository


def document_store_is_required(
    dataset_key: str,
) -> bool:
    return dataset_key in DOCUMENT_STORE_DATASETS


def ensure_dataset_is_imported(
    dataset_key: str,
):
    repository = get_document_store_repository()

    dataset = repository.get_dataset(
        dataset_key
    )

    if dataset is None:
        raise DocumentStoreError(
            f"Dataset '{dataset_key}' is not available in "
            "the offline document store. Run "
            f"'python -m scripts.ingest_dataset "
            f"--dataset {dataset_key}' first."
        )


def enrich_search_results(
    dataset_key: str,
    results: Iterable[Dict[str, Any]],
    snippet_length: int = 500,
    include_raw_text: bool = False,
    include_metadata: bool = False,
    strict: bool = True,
) -> List[Dict[str, Any]]:
    """
    Replace index-provided titles/snippets with original records from SQLite.

    Ranking, scores, and model-specific fields are preserved.

    Args:
        dataset_key:
            Dataset associated with the result list.

        results:
            Ranked result dictionaries containing doc_id.

        snippet_length:
            Maximum number of characters copied from raw_text into snippet.

        include_raw_text:
            Include the entire original text in each result. This should
            normally remain False for search-result lists.

        include_metadata:
            Include document metadata stored in SQLite.

        strict:
            Raise an error when a retrieved doc_id cannot be found in the
            document store. This prevents silently returning index metadata
            when the database is incomplete.
    """
    if snippet_length <= 0:
        raise ValueError(
            "snippet_length must be greater than zero."
        )

    ranked_results = [
        dict(result)
        for result in results
    ]

    if not ranked_results:
        return []

    # sample_dataset is retained as a small internal test collection and
    # is not required to be imported into the production document store.
    if not document_store_is_required(
        dataset_key
    ):
        for result in ranked_results:
            result["document_source"] = (
                "retrieval_index"
            )

        return ranked_results

    ensure_dataset_is_imported(
        dataset_key
    )

    ordered_doc_ids = []

    for result in ranked_results:
        doc_id = result.get("doc_id")

        if doc_id is None:
            if strict:
                raise DocumentStoreError(
                    "A retrieval result is missing doc_id."
                )

            continue

        normalized_doc_id = str(
            doc_id
        ).strip()

        if not normalized_doc_id:
            if strict:
                raise DocumentStoreError(
                    "A retrieval result contains an empty doc_id."
                )

            continue

        result["doc_id"] = normalized_doc_id
        ordered_doc_ids.append(
            normalized_doc_id
        )

    repository = get_document_store_repository()

    documents = repository.get_documents(
        dataset_key=dataset_key,
        doc_ids=ordered_doc_ids,
    )

    documents_by_id = {
        document["doc_id"]: document
        for document in documents
    }

    missing_doc_ids = [
        doc_id
        for doc_id in ordered_doc_ids
        if doc_id not in documents_by_id
    ]

    if missing_doc_ids and strict:
        unique_missing_ids = list(
            dict.fromkeys(missing_doc_ids)
        )

        raise DocumentStoreError(
            "Retrieved document IDs are missing from the "
            f"document store for dataset '{dataset_key}'. "
            f"Missing count: {len(unique_missing_ids)}. "
            f"Examples: {unique_missing_ids[:10]}"
        )

    enriched_results = []

    for result in ranked_results:
        doc_id = result.get("doc_id")
        document = documents_by_id.get(
            doc_id
        )

        if document is None:
            if strict:
                continue

            result["document_source"] = (
                "retrieval_index"
            )

            enriched_results.append(
                result
            )

            continue

        raw_text = str(
            document.get(
                "raw_text",
                "",
            )
            or ""
        )

        result["title"] = str(
            document.get(
                "title",
                "",
            )
            or ""
        )

        result["snippet"] = raw_text[
            :snippet_length
        ]

        result["document_source"] = (
            "sqlite_document_store"
        )

        if include_raw_text:
            result["raw_text"] = raw_text

        if include_metadata:
            result["document_metadata"] = (
                document.get(
                    "metadata",
                    {},
                )
            )

        enriched_results.append(
            result
        )

    return enriched_results


def get_raw_document(
    dataset_key: str,
    doc_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve one full original document from SQLite.
    """
    if not document_store_is_required(
        dataset_key
    ):
        return None

    ensure_dataset_is_imported(
        dataset_key
    )

    repository = get_document_store_repository()

    return repository.get_document(
        dataset_key=dataset_key,
        doc_id=str(doc_id),
    )