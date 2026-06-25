import os
from django.conf import settings


def get_project_root() -> str:
    return settings.BASE_DIR.parent


def get_data_root() -> str:
    return str(
        getattr(
            settings,
            "DATA_DIR",
            settings.BASE_DIR.parent / "data",
        )
    )


DATASET_REGISTRY = {
    "sample_dataset": {
        "name": "Sample Dataset",
        "documents_path": os.path.join(
            get_data_root(),
            "sample_dataset",
            "documents.json"
        ),
        "queries_path": os.path.join(
            get_data_root(),
            "sample_dataset",
            "queries.tsv"
        ),
        "qrels_path": os.path.join(
            get_data_root(),
            "sample_dataset",
            "qrels.tsv"
        ),
        "format": "json",
    },

    "clinical_trials": {
        "name": "Clinical Trials TREC PM 2018",
        "documents_path": os.path.join(
            get_data_root(),
            "clinical_trials",
            "documents.jsonl"
        ),
        "queries_path": os.path.join(
            get_data_root(),
            "clinical_trials",
            "queries.tsv"
        ),
        "qrels_path": os.path.join(
            get_data_root(),
            "clinical_trials",
            "qrels.tsv"
        ),
        "format": "jsonl",
    },
    
    "quora": {
        "name": "BEIR Quora Test",
        "documents_path": os.path.join(
            get_data_root(),
            "quora",
            "documents.jsonl"
        ),
        "queries_path": os.path.join(
            get_data_root(),
            "quora",
            "queries.tsv"
        ),
        "qrels_path": os.path.join(
            get_data_root(),
            "quora",
            "qrels.tsv"
        ),
        "format": "jsonl",
    },
}


def get_dataset_config(dataset_key: str) -> dict:
    if dataset_key not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. Available datasets: {available}"
        )

    return DATASET_REGISTRY[dataset_key]


def list_datasets() -> list:
    return [
        {
            "key": key,
            "name": value["name"],
            "format": value["format"],
        }
        for key, value in DATASET_REGISTRY.items()
    ]
