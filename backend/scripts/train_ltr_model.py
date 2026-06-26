import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = None

for candidate in [
    SCRIPT_PATH.parent,
    *SCRIPT_PATH.parents,
]:
    if candidate.name == "backend":
        BACKEND_DIR = candidate
        break

if BACKEND_DIR is None:
    BACKEND_DIR = SCRIPT_PATH.parents[1]

PROJECT_ROOT = BACKEND_DIR.parent

for path in (
    PROJECT_ROOT,
    BACKEND_DIR,
):
    path_string = str(path)

    if path_string not in sys.path:
        sys.path.insert(
            0,
            path_string,
        )


os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

import django

django.setup()

from django.conf import settings

from datasets.dataset_loader import DatasetLoader
from evaluation.metrics import (
    average_precision,
    ndcg_at_k,
)
from retrieval.ltr_feature_extractor import (
    LTRFeatureExtractor,
    build_ltr_labels,
    normalize_ltr_candidate_models,
)
from retrieval.ltr_service import (
    DEFAULT_LTR_CANDIDATE_COUNT,
    LTRCandidateGenerator,
    get_ltr_metadata_path,
    validate_ltr_candidate_count,
)


SUPPORTED_DATASETS = [
    "quora",
    "clinical_trials",
]

DEFAULT_RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a Learning-to-Rank reranker from "
            "existing retrieval candidates and qrels."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=DEFAULT_LTR_CANDIDATE_COUNT,
    )

    parser.add_argument(
        "--candidate-models",
        "--ltr-candidate-models",
        dest="candidate_models",
        nargs="+",
        default=None,
        help=(
            "Candidate sources. Default: bm25 tfidf embedding."
        ),
    )

    parser.add_argument(
        "--include-biomedical",
        action="store_true",
        help=(
            "Include biomedical PubMedBERT candidates. "
            "Only valid for clinical_trials."
        ),
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help=(
            "Optional maximum number of qrel-backed queries "
            "used for training and validation."
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output model path. Default: "
            "artifacts/models/ltr/<dataset>_ltr.joblib"
        ),
    )

    return parser.parse_args()


def default_output_path(
    dataset_key: str,
) -> Path:
    return (
        Path(settings.ARTIFACTS_DIR)
        / "models"
        / "ltr"
        / f"{dataset_key}_ltr.joblib"
    )


def validate_training_args(args):
    validate_ltr_candidate_count(
        args.candidate_count
    )

    if (
        args.max_queries is not None
        and args.max_queries <= 0
    ):
        raise ValueError(
            "max-queries must be greater than zero."
        )

    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError(
            "validation-fraction must be between 0 and 1."
        )

    normalize_ltr_candidate_models(
        args.candidate_models,
        include_biomedical=args.include_biomedical,
        dataset_key=args.dataset,
    )


def build_query_map(
    queries: List[Dict[str, Any]],
) -> Dict[str, str]:
    result = {}

    for query in queries:
        query_id = str(
            query.get("query_id", "")
            or ""
        ).strip()
        query_text = str(
            query.get("query", "")
            or ""
        ).strip()

        if query_id and query_text:
            result[query_id] = query_text

    return result


def split_query_ids(
    query_ids: List[str],
    validation_fraction: float,
    random_seed: int,
) -> tuple[List[str], List[str]]:
    shuffled = list(query_ids)
    random.Random(
        random_seed
    ).shuffle(shuffled)

    if len(shuffled) == 1:
        return (
            shuffled,
            [],
        )

    validation_count = max(
        1,
        int(
            round(
                len(shuffled)
                * validation_fraction
            )
        ),
    )
    validation_count = min(
        validation_count,
        len(shuffled) - 1,
    )

    return (
        shuffled[validation_count:],
        shuffled[:validation_count],
    )


def build_ltr_rows(
    query_ids: List[str],
    query_by_id: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    candidate_generator: LTRCandidateGenerator,
    feature_extractor: LTRFeatureExtractor,
    candidate_count: int,
) -> Dict[str, Any]:
    feature_rows = []
    labels = []
    groups: Dict[str, List[int]] = {}
    candidate_rows = 0

    for query_number, query_id in enumerate(
        query_ids,
        start=1,
    ):
        query_text = query_by_id[query_id]
        candidates = candidate_generator.generate(
            query=query_text,
            candidate_count=candidate_count,
        )
        documents_by_id = (
            candidate_generator
            .fetch_documents_for_candidates(
                candidates
            )
        )
        feature_matrix, _feature_dicts = (
            feature_extractor
            .extract_feature_matrix(
                query=query_text,
                candidates=candidates,
                documents_by_id=documents_by_id,
            )
        )
        query_labels = build_ltr_labels(
            candidates,
            qrels.get(query_id, {}),
        )

        start_index = len(labels)

        feature_rows.extend(
            feature_matrix.tolist()
        )
        labels.extend(query_labels)
        groups[query_id] = list(
            range(
                start_index,
                start_index
                + len(query_labels),
            )
        )
        candidate_rows += len(candidates)

        if (
            query_number % 10 == 0
            or query_number == len(query_ids)
        ):
            print(
                "Built LTR rows for queries: "
                f"{query_number:,}/"
                f"{len(query_ids):,}",
                flush=True,
            )

    matrix = np.asarray(
        feature_rows,
        dtype=np.float32,
    )

    label_array = np.asarray(
        labels,
        dtype=np.float32,
    )

    return {
        "features": matrix,
        "labels": label_array,
        "groups": groups,
        "candidate_rows": candidate_rows,
    }


def evaluate_validation(
    model,
    validation_rows: Dict[str, Any],
    qrels: Dict[str, Dict[str, int]],
    candidates_by_query: Dict[
        str,
        List[str],
    ],
) -> Dict[str, float]:
    features = validation_rows["features"]

    if features.size == 0:
        return {
            "MAP": 0.0,
            "nDCG@10": 0.0,
        }

    predictions = np.asarray(
        model.predict(features),
        dtype=np.float64,
    )
    ap_scores = []
    ndcg_scores = []

    for query_id, row_indices in (
        validation_rows["groups"].items()
    ):
        doc_ids = candidates_by_query.get(
            query_id,
            [],
        )
        scored_docs = [
            (
                doc_ids[position],
                float(predictions[row_index]),
            )
            for position, row_index in enumerate(
                row_indices
            )
            if position < len(doc_ids)
        ]
        ranked_doc_ids = [
            doc_id
            for doc_id, score in sorted(
                scored_docs,
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ]
        relevant_doc_ids = {
            doc_id
            for doc_id, relevance
            in qrels.get(query_id, {}).items()
            if relevance > 0
        }

        ap_scores.append(
            average_precision(
                retrieved_doc_ids=ranked_doc_ids,
                relevant_doc_ids=(
                    relevant_doc_ids
                ),
            )
        )
        ndcg_scores.append(
            ndcg_at_k(
                retrieved_doc_ids=ranked_doc_ids,
                relevance_scores=qrels.get(
                    query_id,
                    {},
                ),
                k=10,
            )
        )

    return {
        "MAP": round(
            float(
                np.mean(ap_scores)
                if ap_scores
                else 0.0
            ),
            6,
        ),
        "nDCG@10": round(
            float(
                np.mean(ndcg_scores)
                if ndcg_scores
                else 0.0
            ),
            6,
        ),
    }


def collect_candidate_doc_ids(
    query_ids: List[str],
    query_by_id: Dict[str, str],
    candidate_generator: LTRCandidateGenerator,
    candidate_count: int,
) -> Dict[str, List[str]]:
    result = {}

    for query_id in query_ids:
        candidates = candidate_generator.generate(
            query=query_by_id[query_id],
            candidate_count=candidate_count,
        )
        result[query_id] = [
            str(candidate["doc_id"])
            for candidate in candidates
        ]

    return result


def train_ltr_model(
    dataset_key: str,
    candidate_count: int,
    output_path: Path,
    candidate_models: List[str] | None = None,
    include_biomedical: bool = False,
    max_queries: int | None = None,
    validation_fraction: float = 0.2,
    random_seed: int = DEFAULT_RANDOM_SEED,
    candidate_generator: LTRCandidateGenerator | None = None,
) -> Dict[str, Any]:
    candidate_count = validate_ltr_candidate_count(
        candidate_count
    )
    normalized_candidate_models = (
        normalize_ltr_candidate_models(
            candidate_models,
            include_biomedical=include_biomedical,
            dataset_key=dataset_key,
        )
    )

    queries = DatasetLoader.load_queries(
        dataset_key
    )
    qrels = DatasetLoader.load_qrels(
        dataset_key
    )
    query_by_id = build_query_map(
        queries
    )

    query_ids = [
        query_id
        for query_id in query_by_id
        if query_id in qrels
    ]

    if max_queries is not None:
        query_ids = query_ids[:max_queries]

    if not query_ids:
        raise ValueError(
            "No qrel-backed queries are available "
            f"for dataset '{dataset_key}'."
        )

    train_query_ids, validation_query_ids = (
        split_query_ids(
            query_ids=query_ids,
            validation_fraction=validation_fraction,
            random_seed=random_seed,
        )
    )

    if candidate_generator is None:
        candidate_generator = LTRCandidateGenerator(
            dataset_key=dataset_key,
            candidate_models=(
                normalized_candidate_models
            ),
            include_biomedical=False,
        )

    feature_extractor = LTRFeatureExtractor(
        dataset_key=dataset_key
    )

    print(
        "Number of queries: "
        f"{len(query_ids):,}"
    )
    print(
        "Number of qrels queries: "
        f"{len(qrels):,}"
    )
    print(
        "Train queries: "
        f"{len(train_query_ids):,}"
    )
    print(
        "Validation queries: "
        f"{len(validation_query_ids):,}"
    )

    build_start = perf_counter()

    train_rows = build_ltr_rows(
        query_ids=train_query_ids,
        query_by_id=query_by_id,
        qrels=qrels,
        candidate_generator=(
            candidate_generator
        ),
        feature_extractor=(
            feature_extractor
        ),
        candidate_count=candidate_count,
    )

    validation_rows = build_ltr_rows(
        query_ids=validation_query_ids,
        query_by_id=query_by_id,
        qrels=qrels,
        candidate_generator=(
            candidate_generator
        ),
        feature_extractor=(
            feature_extractor
        ),
        candidate_count=candidate_count,
    )

    if train_rows["features"].size == 0:
        raise ValueError(
            "No LTR training rows were generated."
        )

    model = GradientBoostingRegressor(
        random_state=random_seed
    )

    model.fit(
        train_rows["features"],
        train_rows["labels"],
    )

    validation_candidate_doc_ids = (
        collect_candidate_doc_ids(
            query_ids=validation_query_ids,
            query_by_id=query_by_id,
            candidate_generator=(
                candidate_generator
            ),
            candidate_count=candidate_count,
        )
        if validation_query_ids
        else {}
    )

    validation_metrics = evaluate_validation(
        model=model,
        validation_rows=validation_rows,
        qrels=qrels,
        candidates_by_query=(
            validation_candidate_doc_ids
        ),
    )

    label_counter = Counter(
        float(label)
        for label in train_rows[
            "labels"
        ].tolist()
    )

    positive_labels = int(
        sum(
            count
            for label, count
            in label_counter.items()
            if label > 0
        )
    )
    negative_labels = int(
        label_counter.get(0.0, 0)
    )

    metadata = {
        "dataset": dataset_key,
        "model_type": type(model).__name__,
        "feature_names": (
            feature_extractor.feature_names
        ),
        "candidate_models": (
            normalized_candidate_models
        ),
        "include_biomedical": (
            "biomedical"
            in normalized_candidate_models
        ),
        "candidate_count": candidate_count,
        "train_query_count": len(
            train_query_ids
        ),
        "validation_query_count": len(
            validation_query_ids
        ),
        "training_rows": int(
            train_rows["features"].shape[0]
        ),
        "validation_rows": int(
            validation_rows["features"].shape[0]
        ),
        "label_distribution": {
            str(label): int(count)
            for label, count
            in sorted(label_counter.items())
        },
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "validation_metrics": (
            validation_metrics
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_seed": random_seed,
        "training_wall_time_seconds": round(
            perf_counter() - build_start,
            3,
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(
        model,
        output_path,
        compress=3,
    )

    metadata_path = get_ltr_metadata_path(
        output_path
    )
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Candidate rows: "
        f"{metadata['training_rows'] + metadata['validation_rows']:,}"
    )
    print(
        "Positive labels: "
        f"{positive_labels:,}"
    )
    print(
        "Negative labels: "
        f"{negative_labels:,}"
    )
    print(
        "Validation MAP: "
        f"{validation_metrics['MAP']}"
    )
    print(
        "Validation nDCG@10: "
        f"{validation_metrics['nDCG@10']}"
    )
    print(
        "Saved model path: "
        f"{output_path}"
    )
    print(
        "Saved metadata path: "
        f"{metadata_path}"
    )

    return {
        "model_path": str(output_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def main():
    args = parse_args()

    try:
        validate_training_args(args)
        output_path = (
            Path(args.output).expanduser()
            if args.output
            else default_output_path(
                args.dataset
            )
        )

        if not output_path.is_absolute():
            output_path = (
                settings.BASE_DIR.parent
                / output_path
            ).resolve()

        train_ltr_model(
            dataset_key=args.dataset,
            candidate_count=(
                args.candidate_count
            ),
            output_path=output_path,
            candidate_models=(
                args.candidate_models
            ),
            include_biomedical=(
                args.include_biomedical
            ),
            max_queries=args.max_queries,
            validation_fraction=(
                args.validation_fraction
            ),
            random_seed=args.random_seed,
        )

    except KeyboardInterrupt:
        print()
        print("LTR training interrupted.")
        sys.exit(130)

    except Exception as error:
        print()
        print(f"LTR training failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
