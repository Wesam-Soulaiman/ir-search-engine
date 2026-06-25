import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import ir_datasets
from ftfy import fix_text


SUPPORTED_DATASETS = {
    "quora": {
        "dataset_id": "beir/quora/test",
        "output_name": "quora",
    },
    "clinical_trials": {
        "dataset_id": "clinicaltrials/2017/trec-pm-2018",
        "output_name": "clinical_trials",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download and convert an ir_datasets collection "
            "to the project's unified format."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS.keys(),
        help="Dataset alias to prepare.",
    )

    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help=(
            "Optional maximum number of documents to export. "
            "Required when --evaluation-sample is enabled."
        ),
    )

    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help=(
            "Optional maximum number of queries to export. "
            "Required when --evaluation-sample is enabled."
        ),
    )

    parser.add_argument(
        "--evaluation-sample",
        action="store_true",
        help=(
            "Build a relevance-aware evaluation sample. "
            "Only queries with positive qrels are selected, "
            "and all positively relevant documents are guaranteed "
            "to be included in the exported document collection."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. By default, "
            "data/<dataset-name> under the project root is used."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    return parser.parse_args()


def get_project_root() -> Path:
    """
    Current file:
    project/backend/scripts/prepare_ir_dataset.py

    parents[0] -> scripts
    parents[1] -> backend
    parents[2] -> project root
    """
    return Path(__file__).resolve().parents[2]


def get_output_dir(
    dataset_alias: str,
    custom_output_dir: Optional[str],
) -> Path:
    if custom_output_dir:
        return Path(custom_output_dir).expanduser().resolve()

    output_name = SUPPORTED_DATASETS[dataset_alias]["output_name"]

    return get_project_root() / "data" / output_name


def record_to_dict(item: Any) -> Dict[str, Any]:
    """
    Convert an ir_datasets record object into a dictionary.
    """
    if hasattr(item, "_asdict"):
        return dict(item._asdict())

    if hasattr(item, "__dict__"):
        return vars(item)

    raise TypeError(
        "Unsupported ir_datasets record type: "
        f"{type(item).__name__}"
    )


def first_available(
    values: Dict[str, Any],
    possible_keys: list[str],
    default: Any = None,
) -> Any:
    """
    Return the first available non-None value.
    """
    for key in possible_keys:
        value = values.get(key)

        if value is not None:
            return value

    return default


def clean_text(value: Any) -> str:
    """
    Fix broken Unicode text and normalize line endings.
    """
    if value is None:
        return ""

    text = fix_text(str(value))

    return (
        text
        .replace("\u00a0", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def sanitize_tsv_text(value: Any) -> str:
    """
    Remove characters that would corrupt TSV output.
    """
    return (
        clean_text(value)
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def normalize_document(document: Any) -> Dict[str, str]:
    """
    Convert a document into the project's unified format:

    {
        "doc_id": "...",
        "title": "...",
        "text": "..."
    }
    """
    raw = record_to_dict(document)

    doc_id = first_available(
        raw,
        [
            "doc_id",
            "docid",
            "id",
            "_id",
            "docno",
        ],
    )

    if doc_id is None:
        raise ValueError(
            "Could not find document ID in fields: "
            f"{list(raw.keys())}"
        )

    title = first_available(
        raw,
        [
            "title",
            "headline",
            "subject",
        ],
        default="",
    )

    # Clinical Trials documents contain several important fields.
    if {
        "condition",
        "summary",
        "detailed_description",
        "eligibility",
    }.intersection(raw.keys()):
        text_parts = [
            raw.get("condition", ""),
            raw.get("summary", ""),
            raw.get("detailed_description", ""),
            raw.get("eligibility", ""),
        ]

        cleaned_parts = []

        for part in text_parts:
            cleaned_part = clean_text(part)

            if cleaned_part:
                cleaned_parts.append(cleaned_part)

        return {
            "doc_id": clean_text(doc_id),
            "title": clean_text(title),
            "text": " ".join(cleaned_parts),
        }

    text = first_available(
        raw,
        [
            "text",
            "body",
            "contents",
            "content",
            "answer",
            "passage",
            "argument",
        ],
        default=None,
    )

    if text is None:
        excluded_fields = {
            "doc_id",
            "docid",
            "id",
            "_id",
            "docno",
            "title",
            "headline",
            "subject",
        }

        text_parts = []

        for key, value in raw.items():
            if key in excluded_fields:
                continue

            if isinstance(value, str):
                cleaned_value = clean_text(value)

                if cleaned_value:
                    text_parts.append(cleaned_value)

            elif isinstance(value, (list, tuple)):
                cleaned_items = []

                for item in value:
                    cleaned_item = clean_text(item)

                    if cleaned_item:
                        cleaned_items.append(cleaned_item)

                if cleaned_items:
                    text_parts.append(
                        " ".join(cleaned_items)
                    )

        text = " ".join(text_parts)

    return {
        "doc_id": clean_text(doc_id),
        "title": clean_text(title),
        "text": clean_text(text),
    }


def normalize_query(query: Any) -> Dict[str, str]:
    """
    Convert a query into the project's unified format:

    {
        "query_id": "...",
        "query": "..."
    }
    """
    raw = record_to_dict(query)

    query_id = first_available(
        raw,
        [
            "query_id",
            "qid",
            "id",
        ],
    )

    if query_id is None:
        raise ValueError(
            "Could not find query ID in fields: "
            f"{list(raw.keys())}"
        )

    # TREC Precision Medicine query format.
    if "disease" in raw:
        query_parts = [
            raw.get("disease", ""),
            raw.get("gene", ""),
            raw.get("demographic", ""),
            raw.get("other", ""),
        ]

        cleaned_parts = []

        for part in query_parts:
            cleaned_part = clean_text(part)

            if cleaned_part:
                cleaned_parts.append(cleaned_part)

        query_text = " ".join(cleaned_parts)

        if not query_text:
            raise ValueError(
                "Could not build query text from fields: "
                f"{list(raw.keys())}"
            )

        return {
            "query_id": clean_text(query_id),
            "query": query_text,
        }

    query_text = first_available(
        raw,
        [
            "text",
            "query",
            "title",
            "description",
        ],
    )

    if query_text is None:
        raise ValueError(
            "Could not find query text in fields: "
            f"{list(raw.keys())}"
        )

    return {
        "query_id": clean_text(query_id),
        "query": clean_text(query_text),
    }


def normalize_qrel(qrel: Any) -> Dict[str, Any]:
    """
    Convert a qrel into the project's unified format.
    """
    raw = record_to_dict(qrel)

    query_id = first_available(
        raw,
        [
            "query_id",
            "qid",
        ],
    )

    doc_id = first_available(
        raw,
        [
            "doc_id",
            "docid",
        ],
    )

    relevance = first_available(
        raw,
        [
            "relevance",
            "rel",
            "score",
        ],
        default=0,
    )

    if query_id is None or doc_id is None:
        raise ValueError(
            f"Invalid qrel fields: {list(raw.keys())}"
        )

    return {
        "query_id": clean_text(query_id),
        "doc_id": clean_text(doc_id),
        "relevance": int(relevance),
    }


def ensure_output_files_can_be_written(
    output_paths: list[Path],
    overwrite: bool,
):
    """
    Prevent accidental overwriting of generated files.
    """
    existing_files = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing_files and not overwrite:
        existing_text = "\n".join(
            f"- {path}"
            for path in existing_files
        )

        raise FileExistsError(
            "The following output files already exist:\n"
            f"{existing_text}\n"
            "Use --overwrite to replace them."
        )


def export_documents(
    dataset,
    output_path: Path,
    limit: Optional[int],
) -> tuple[int, set[str]]:
    """
    Export the first documents from the dataset.
    """
    exported_count = 0
    exported_document_ids: set[str] = set()

    print(f"Writing documents to: {output_path}")

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for document in dataset.docs_iter():
            if (
                limit is not None
                and exported_count >= limit
            ):
                break

            normalized = normalize_document(document)
            doc_id = normalized["doc_id"]

            if doc_id in exported_document_ids:
                continue

            output_file.write(
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                )
                + "\n"
            )

            exported_document_ids.add(doc_id)
            exported_count += 1

            if exported_count % 10_000 == 0:
                print(
                    f"Exported {exported_count:,} documents...",
                    flush=True,
                )

    return exported_count, exported_document_ids


def export_queries(
    dataset,
    output_path: Path,
    limit: Optional[int],
) -> tuple[int, set[str]]:
    """
    Export the first queries from the dataset.
    """
    exported_count = 0
    exported_query_ids: set[str] = set()

    print(f"Writing queries to: {output_path}")

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        for query in dataset.queries_iter():
            if (
                limit is not None
                and exported_count >= limit
            ):
                break

            normalized = normalize_query(query)
            query_id = normalized["query_id"]

            if query_id in exported_query_ids:
                continue

            output_file.write(
                f"{query_id}\t"
                f"{sanitize_tsv_text(normalized['query'])}\n"
            )

            exported_query_ids.add(query_id)
            exported_count += 1

    return exported_count, exported_query_ids


def export_qrels(
    dataset,
    output_path: Path,
    allowed_query_ids: Optional[set[str]] = None,
    allowed_document_ids: Optional[set[str]] = None,
) -> tuple[int, int]:
    """
    Export qrels matching the exported queries and documents.

    Returns:
        Total exported qrels.
        Number of positive exported qrels.
    """
    exported_count = 0
    positive_qrels_count = 0

    print(f"Writing qrels to: {output_path}")

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        for qrel in dataset.qrels_iter():
            normalized = normalize_qrel(qrel)

            if (
                allowed_query_ids is not None
                and normalized["query_id"]
                not in allowed_query_ids
            ):
                continue

            if (
                allowed_document_ids is not None
                and normalized["doc_id"]
                not in allowed_document_ids
            ):
                continue

            output_file.write(
                f"{normalized['query_id']}\t"
                f"{normalized['doc_id']}\t"
                f"{normalized['relevance']}\n"
            )

            exported_count += 1

            if normalized["relevance"] > 0:
                positive_qrels_count += 1

    return exported_count, positive_qrels_count


def collect_qrels(
    dataset,
) -> tuple[
    list[Dict[str, Any]],
    Dict[str, set[str]],
]:
    """
    Read all qrels and build a mapping from each query ID
    to its positively relevant document IDs.
    """
    normalized_qrels = []
    positive_documents_by_query: Dict[str, set[str]] = {}

    print("Reading qrels for evaluation-sample selection...")

    for qrel in dataset.qrels_iter():
        normalized = normalize_qrel(qrel)
        normalized_qrels.append(normalized)

        if normalized["relevance"] <= 0:
            continue

        query_id = normalized["query_id"]
        doc_id = normalized["doc_id"]

        positive_documents_by_query.setdefault(
            query_id,
            set(),
        ).add(doc_id)

    print(
        "Queries with positive qrels: "
        f"{len(positive_documents_by_query):,}"
    )

    return normalized_qrels, positive_documents_by_query


def select_evaluation_queries(
    dataset,
    positive_documents_by_query: Dict[str, set[str]],
    query_limit: int,
    document_limit: int,
) -> tuple[
    list[Dict[str, str]],
    set[str],
    set[str],
]:
    """
    Select queries that have positive qrels.

    A query is selected only when all of its positive documents
    can fit inside the requested document limit.
    """
    selected_queries: list[Dict[str, str]] = []
    selected_query_ids: set[str] = set()
    required_document_ids: set[str] = set()

    skipped_for_capacity = 0

    print(
        "Selecting evaluation queries with positive relevance judgments..."
    )

    for query in dataset.queries_iter():
        if len(selected_queries) >= query_limit:
            break

        normalized = normalize_query(query)
        query_id = normalized["query_id"]

        if query_id in selected_query_ids:
            continue

        positive_document_ids = (
            positive_documents_by_query.get(
                query_id,
                set(),
            )
        )

        if not positive_document_ids:
            continue

        new_document_ids = (
            positive_document_ids
            - required_document_ids
        )

        if (
            len(required_document_ids)
            + len(new_document_ids)
            > document_limit
        ):
            skipped_for_capacity += 1
            continue

        selected_queries.append(normalized)
        selected_query_ids.add(query_id)
        required_document_ids.update(
            positive_document_ids
        )

    if not selected_queries:
        raise ValueError(
            "No evaluation queries could be selected. "
            "Increase --limit-docs or verify that the dataset "
            "contains positive qrels."
        )

    if len(selected_queries) < query_limit:
        print(
            "Warning: requested "
            f"{query_limit:,} queries, but only "
            f"{len(selected_queries):,} queries could be selected."
        )

    if skipped_for_capacity:
        print(
            "Queries skipped because their relevant documents "
            "would exceed --limit-docs: "
            f"{skipped_for_capacity:,}"
        )

    print(
        "Selected evaluation queries: "
        f"{len(selected_queries):,}"
    )

    print(
        "Required positively relevant documents: "
        f"{len(required_document_ids):,}"
    )

    return (
        selected_queries,
        selected_query_ids,
        required_document_ids,
    )


def export_selected_queries(
    selected_queries: list[Dict[str, str]],
    output_path: Path,
) -> tuple[int, set[str]]:
    """
    Export preselected evaluation queries.
    """
    exported_query_ids: set[str] = set()

    print(f"Writing selected queries to: {output_path}")

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        for query in selected_queries:
            query_id = query["query_id"]

            if query_id in exported_query_ids:
                continue

            output_file.write(
                f"{query_id}\t"
                f"{sanitize_tsv_text(query['query'])}\n"
            )

            exported_query_ids.add(query_id)

    return len(exported_query_ids), exported_query_ids


def export_evaluation_documents(
    dataset,
    output_path: Path,
    required_document_ids: set[str],
    document_limit: int,
) -> tuple[int, set[str], int]:
    """
    Export all required relevant documents and add filler documents
    until document_limit is reached.
    """
    if len(required_document_ids) > document_limit:
        raise ValueError(
            "The number of required relevant documents "
            f"({len(required_document_ids):,}) exceeds "
            f"--limit-docs ({document_limit:,})."
        )

    filler_target = (
        document_limit
        - len(required_document_ids)
    )

    selected_documents: list[Dict[str, str]] = []
    selected_document_ids: set[str] = set()
    missing_required_ids = set(required_document_ids)

    filler_count = 0

    print(
        "Scanning the document collection to build "
        "the evaluation sample..."
    )

    for document in dataset.docs_iter():
        normalized = normalize_document(document)
        doc_id = normalized["doc_id"]

        if doc_id in selected_document_ids:
            continue

        if doc_id in required_document_ids:
            selected_documents.append(normalized)
            selected_document_ids.add(doc_id)
            missing_required_ids.discard(doc_id)

        elif filler_count < filler_target:
            selected_documents.append(normalized)
            selected_document_ids.add(doc_id)
            filler_count += 1

        if (
            not missing_required_ids
            and filler_count >= filler_target
        ):
            break

    if missing_required_ids:
        sample_missing_ids = sorted(
            missing_required_ids
        )[:10]

        raise FileNotFoundError(
            "Some positively relevant documents were not found "
            "in the document collection. Missing count: "
            f"{len(missing_required_ids):,}. "
            f"Examples: {sample_missing_ids}"
        )

    print(f"Writing documents to: {output_path}")

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for document in selected_documents:
            output_file.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

    if len(selected_documents) < document_limit:
        print(
            "Warning: the dataset ended before reaching "
            f"{document_limit:,} unique documents."
        )

    print(
        "Relevant documents included: "
        f"{len(required_document_ids):,}"
    )

    print(
        "Additional filler documents included: "
        f"{filler_count:,}"
    )

    return (
        len(selected_documents),
        selected_document_ids,
        filler_count,
    )


def export_qrel_records(
    qrels: list[Dict[str, Any]],
    output_path: Path,
    allowed_query_ids: set[str],
    allowed_document_ids: set[str],
) -> tuple[int, int]:
    """
    Export qrels from preloaded normalized records.
    """
    exported_count = 0
    positive_qrels_count = 0
    seen_records: set[tuple[str, str, int]] = set()

    print(f"Writing qrels to: {output_path}")

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        for qrel in qrels:
            if qrel["query_id"] not in allowed_query_ids:
                continue

            if qrel["doc_id"] not in allowed_document_ids:
                continue

            record_key = (
                qrel["query_id"],
                qrel["doc_id"],
                int(qrel["relevance"]),
            )

            if record_key in seen_records:
                continue

            output_file.write(
                f"{qrel['query_id']}\t"
                f"{qrel['doc_id']}\t"
                f"{qrel['relevance']}\n"
            )

            seen_records.add(record_key)
            exported_count += 1

            if qrel["relevance"] > 0:
                positive_qrels_count += 1

    return exported_count, positive_qrels_count


def write_metadata(
    output_path: Path,
    dataset_alias: str,
    dataset_id: str,
    documents_count: int,
    unique_documents_count: int,
    queries_count: int,
    qrels_count: int,
    positive_qrels_count: int,
    limited_documents: Optional[int],
    limited_queries: Optional[int],
    evaluation_sample: bool,
    required_relevant_documents: int = 0,
    filler_documents: int = 0,
):
    """
    Write dataset preparation metadata.
    """
    metadata = {
        "dataset_alias": dataset_alias,
        "ir_datasets_id": dataset_id,
        "documents_exported": documents_count,
        "unique_document_ids": unique_documents_count,
        "queries_exported": queries_count,
        "qrels_exported": qrels_count,
        "positive_qrels_exported": positive_qrels_count,
        "document_limit": limited_documents,
        "query_limit": limited_queries,
        "evaluation_sample": evaluation_sample,
        "required_relevant_documents": (
            required_relevant_documents
        ),
        "filler_documents": filler_documents,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            metadata,
            output_file,
            ensure_ascii=False,
            indent=2,
        )


def prepare_dataset(
    dataset_alias: str,
    output_dir: Path,
    limit_docs: Optional[int],
    limit_queries: Optional[int],
    overwrite: bool,
    evaluation_sample: bool,
):
    """
    Download, normalize, and export a supported dataset.
    """
    dataset_config = SUPPORTED_DATASETS[dataset_alias]
    dataset_id = dataset_config["dataset_id"]

    documents_path = output_dir / "documents.jsonl"
    queries_path = output_dir / "queries.tsv"
    qrels_path = output_dir / "qrels.tsv"
    metadata_path = output_dir / "metadata.json"

    output_paths = [
        documents_path,
        queries_path,
        qrels_path,
        metadata_path,
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ensure_output_files_can_be_written(
        output_paths=output_paths,
        overwrite=overwrite,
    )

    print("=" * 70)
    print(f"Dataset alias: {dataset_alias}")
    print(f"ir_datasets ID: {dataset_id}")
    print(f"Output directory: {output_dir}")
    print(
        "Preparation mode: "
        + (
            "relevance-aware evaluation sample"
            if evaluation_sample
            else "standard prefix sample"
        )
    )
    print("=" * 70)

    print(
        "Loading dataset. ir_datasets may download "
        "required files during the first run."
    )

    dataset = ir_datasets.load(dataset_id)

    required_relevant_documents = 0
    filler_documents = 0

    if evaluation_sample:
        if limit_docs is None:
            raise ValueError(
                "--limit-docs is required when "
                "--evaluation-sample is enabled."
            )

        if limit_queries is None:
            raise ValueError(
                "--limit-queries is required when "
                "--evaluation-sample is enabled."
            )

        (
            all_qrels,
            positive_documents_by_query,
        ) = collect_qrels(dataset)

        (
            selected_queries,
            selected_query_ids,
            required_document_ids,
        ) = select_evaluation_queries(
            dataset=dataset,
            positive_documents_by_query=(
                positive_documents_by_query
            ),
            query_limit=limit_queries,
            document_limit=limit_docs,
        )

        required_relevant_documents = len(
            required_document_ids
        )

        (
            documents_count,
            allowed_document_ids,
            filler_documents,
        ) = export_evaluation_documents(
            dataset=dataset,
            output_path=documents_path,
            required_document_ids=required_document_ids,
            document_limit=limit_docs,
        )

        (
            queries_count,
            allowed_query_ids,
        ) = export_selected_queries(
            selected_queries=selected_queries,
            output_path=queries_path,
        )

        (
            qrels_count,
            positive_qrels_count,
        ) = export_qrel_records(
            qrels=all_qrels,
            output_path=qrels_path,
            allowed_query_ids=allowed_query_ids,
            allowed_document_ids=allowed_document_ids,
        )

    else:
        (
            documents_count,
            allowed_document_ids,
        ) = export_documents(
            dataset=dataset,
            output_path=documents_path,
            limit=limit_docs,
        )

        (
            queries_count,
            allowed_query_ids,
        ) = export_queries(
            dataset=dataset,
            output_path=queries_path,
            limit=limit_queries,
        )

        (
            qrels_count,
            positive_qrels_count,
        ) = export_qrels(
            dataset=dataset,
            output_path=qrels_path,
            allowed_query_ids=allowed_query_ids,
            allowed_document_ids=allowed_document_ids,
        )

    print(
        "Document IDs available for qrels filtering: "
        f"{len(allowed_document_ids):,}"
    )

    print(
        "Query IDs available for qrels filtering: "
        f"{len(allowed_query_ids):,}"
    )

    write_metadata(
        output_path=metadata_path,
        dataset_alias=dataset_alias,
        dataset_id=dataset_id,
        documents_count=documents_count,
        unique_documents_count=len(
            allowed_document_ids
        ),
        queries_count=queries_count,
        qrels_count=qrels_count,
        positive_qrels_count=positive_qrels_count,
        limited_documents=limit_docs,
        limited_queries=limit_queries,
        evaluation_sample=evaluation_sample,
        required_relevant_documents=(
            required_relevant_documents
        ),
        filler_documents=filler_documents,
    )

    print()
    print("=" * 70)
    print("Dataset preparation completed successfully.")
    print(f"Documents exported: {documents_count:,}")
    print(
        "Unique document IDs: "
        f"{len(allowed_document_ids):,}"
    )
    print(f"Queries exported: {queries_count:,}")
    print(f"Qrels exported: {qrels_count:,}")
    print(
        "Positive qrels exported: "
        f"{positive_qrels_count:,}"
    )
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    if positive_qrels_count == 0:
        print()
        print(
            "Warning: the exported sample contains no "
            "positive relevance judgments."
        )
        print(
            "Do not use this sample for evaluation."
        )

    elif evaluation_sample:
        print()
        print(
            "The sample contains positively relevant "
            "documents and can be used for development evaluation."
        )
        print(
            "It is still a reduced sample and should not be "
            "reported as the final full-dataset evaluation."
        )


def validate_positive_limit(
    value: Optional[int],
    argument_name: str,
):
    """
    Validate an optional positive integer argument.
    """
    if value is not None and value <= 0:
        raise ValueError(
            f"{argument_name} must be greater than zero."
        )


def main():
    args = parse_args()

    try:
        validate_positive_limit(
            args.limit_docs,
            "--limit-docs",
        )

        validate_positive_limit(
            args.limit_queries,
            "--limit-queries",
        )

        if args.evaluation_sample:
            if args.limit_docs is None:
                raise ValueError(
                    "--limit-docs is required with "
                    "--evaluation-sample."
                )

            if args.limit_queries is None:
                raise ValueError(
                    "--limit-queries is required with "
                    "--evaluation-sample."
                )

        output_dir = get_output_dir(
            dataset_alias=args.dataset,
            custom_output_dir=args.output_dir,
        )

        prepare_dataset(
            dataset_alias=args.dataset,
            output_dir=output_dir,
            limit_docs=args.limit_docs,
            limit_queries=args.limit_queries,
            overwrite=args.overwrite,
            evaluation_sample=args.evaluation_sample,
        )

    except Exception as error:
        print()
        print(f"Dataset preparation failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()