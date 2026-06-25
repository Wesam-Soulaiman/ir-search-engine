import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from agents.retrieval_strategy_agent import (
    RetrievalStrategyAgent,
)
from datasets.dataset_registry import (
    get_dataset_config,
    list_datasets,
)
from document_store.repository import (
    DocumentStoreError,
)
from query_refinement.pseudo_relevance_feedback import (
    PseudoRelevanceFeedbackService,
)
from query_refinement.spelling_correction_service import (
    SpellingCorrectionService,
)
from retrieval.bm25_service import (
    BM25RetrievalService,
)
from retrieval.biomedical_embedding_service import (
    BiomedicalEmbeddingService,
)
from retrieval.embedding_service import (
    EmbeddingRetrievalService,
)
from retrieval.hybrid_parallel_service import (
    HybridParallelRetrievalService,
)
from retrieval.hybrid_serial_service import (
    HybridSerialRetrievalService,
)
from retrieval.personalization_service import (
    PersonalizedQueryService,
)
from retrieval.request_validation import (
    parse_boolean,
    parse_float,
    parse_integer,
)
from retrieval.result_enrichment import (
    document_store_is_required,
    enrich_search_results,
    get_raw_document,
)
from retrieval.search_history_store import (
    SearchHistoryStore,
)
from retrieval.tfidf_service import (
    TfidfRetrievalService,
)


logger = logging.getLogger(__name__)


SUPPORTED_MODELS = {
    "tfidf",
    "bm25",
    "embedding",
    "biomedical_embedding",
    "hybrid_serial",
    "hybrid_parallel",
    "agent",
}


DEFAULT_DATASET = "sample_dataset"
DEFAULT_MODEL = "tfidf"

DEFAULT_TOP_K = 10
MAX_TOP_K = 1000

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

DEFAULT_CANDIDATE_COUNT = 1000
MAX_CANDIDATE_COUNT = 10_000

DEFAULT_RRF_K = 60
MAX_RRF_K = 100_000

DEFAULT_TFIDF_WEIGHT = 1.0
DEFAULT_BM25_WEIGHT = 1.0
DEFAULT_EMBEDDING_WEIGHT = 1.0
DEFAULT_BIOMEDICAL_WEIGHT = 0.0
MAX_HYBRID_WEIGHT = 100.0

DEFAULT_FEEDBACK_DOCS = 3
MAX_FEEDBACK_DOCS = 100

DEFAULT_EXPANSION_TERMS = 5
MAX_EXPANSION_TERMS = 100

DEFAULT_SNIPPET_LENGTH = 500
MAX_SNIPPET_LENGTH = 5000

DEFAULT_INCLUDE_RAW_TEXT = False
MAX_RAW_TEXT_RESULTS = 50

DEFAULT_MAX_PERSONALIZATION_TERMS = 3
MAX_PERSONALIZATION_TERMS = 20
DEFAULT_PERSONALIZATION_HISTORY_LIMIT = 20
DEFAULT_USE_SPELLING_CORRECTION = False


@lru_cache(maxsize=4)
def get_tfidf_service(
    dataset_key: str = DEFAULT_DATASET,
):
    return TfidfRetrievalService(
        dataset_key=dataset_key
    )


@lru_cache(maxsize=16)
def get_bm25_service(
    dataset_key: str = DEFAULT_DATASET,
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
):
    return BM25RetrievalService(
        dataset_key=dataset_key,
        k1=float(k1),
        b=float(b),
    )


@lru_cache(maxsize=4)
def get_embedding_service(
    dataset_key: str = DEFAULT_DATASET,
):
    return EmbeddingRetrievalService(
        dataset_key=dataset_key
    )


@lru_cache(maxsize=1)
def get_biomedical_embedding_service(
    dataset_key: str = "clinical_trials",
):
    return BiomedicalEmbeddingService(
        dataset_key=dataset_key
    )


@lru_cache(maxsize=16)
def get_hybrid_serial_service(
    dataset_key: str = DEFAULT_DATASET,
    bm25_k1: float = DEFAULT_BM25_K1,
    bm25_b: float = DEFAULT_BM25_B,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
):
    return HybridSerialRetrievalService(
        dataset_key=dataset_key,
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
        candidate_count=int(candidate_count),
    )


@lru_cache(maxsize=64)
def get_hybrid_parallel_service(
    dataset_key: str = DEFAULT_DATASET,
    bm25_k1: float = DEFAULT_BM25_K1,
    bm25_b: float = DEFAULT_BM25_B,
    rrf_k: int = DEFAULT_RRF_K,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    tfidf_weight: float = DEFAULT_TFIDF_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    biomedical_weight: float = DEFAULT_BIOMEDICAL_WEIGHT,
):
    return HybridParallelRetrievalService(
        dataset_key=dataset_key,
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
        rrf_k=int(rrf_k),
        candidate_count=int(candidate_count),
        tfidf_weight=float(tfidf_weight),
        bm25_weight=float(bm25_weight),
        embedding_weight=float(embedding_weight),
        biomedical_weight=float(biomedical_weight),
    )


@lru_cache(maxsize=16)
def get_query_refinement_service(
    dataset_key: str = DEFAULT_DATASET,
    bm25_k1: float = DEFAULT_BM25_K1,
    bm25_b: float = DEFAULT_BM25_B,
    feedback_docs: int = DEFAULT_FEEDBACK_DOCS,
    expansion_terms: int = DEFAULT_EXPANSION_TERMS,
):
    return PseudoRelevanceFeedbackService(
        dataset_key=dataset_key,
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
        feedback_docs=int(feedback_docs),
        expansion_terms=int(expansion_terms),
    )


@lru_cache(maxsize=1)
def get_retrieval_strategy_agent():
    return RetrievalStrategyAgent()


@lru_cache(maxsize=1)
def get_search_history_store():
    artifacts_dir = Path(
        getattr(
            settings,
            "ARTIFACTS_DIR",
            settings.BASE_DIR.parent / "artifacts",
        )
    ).expanduser().resolve()

    store = SearchHistoryStore(
        artifacts_dir
        / "database"
        / "search_history.sqlite3"
    )

    store.initialize()

    return store


def get_personalized_query_service(
    dataset_key: str,
    max_personalization_terms: int,
):
    return PersonalizedQueryService(
        dataset_key=dataset_key,
        max_personalization_terms=max_personalization_terms,
    )


@lru_cache(maxsize=4)
def get_spelling_correction_service(
    dataset_key: str,
):
    return SpellingCorrectionService(
        dataset_key=dataset_key,
    )


def record_search_history_safely(
    user_id: str | None,
    dataset_key: str,
    query: str,
):
    if not user_id:
        return

    try:
        store = get_search_history_store()
        store.record_query(
            user_id=user_id,
            dataset=dataset_key,
            query=query,
        )

    except Exception:
        logger.exception(
            "Failed to record anonymous search history."
        )


def normalize_optional_string_field(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"'{field_name}' must be a string."
        )

    normalized_value = value.strip()

    return normalized_value or None


def normalize_string_field(
    value: Any,
    field_name: str,
    default: str | None = None,
) -> str:
    if value is None:
        if default is None:
            raise ValueError(
                f"'{field_name}' is required."
            )

        value = default

    if not isinstance(value, str):
        raise ValueError(
            f"'{field_name}' must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"'{field_name}' cannot be empty."
        )

    return normalized_value


def validate_dataset(
    dataset_key: str,
):
    get_dataset_config(dataset_key)


def validate_model(
    model_name: str,
):
    if model_name not in SUPPORTED_MODELS:
        available_models = ", ".join(
            sorted(SUPPORTED_MODELS)
        )

        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available_models}"
        )


def parse_search_request(
    request_data: Dict[str, Any],
) -> Dict[str, Any]:
    query = normalize_string_field(
        request_data.get("query"),
        field_name="query",
    )

    dataset_key = normalize_string_field(
        request_data.get(
            "dataset",
            DEFAULT_DATASET,
        ),
        field_name="dataset",
    ).lower()

    model = normalize_string_field(
        request_data.get(
            "model",
            DEFAULT_MODEL,
        ),
        field_name="model",
    ).lower()

    validate_dataset(dataset_key)
    validate_model(model)

    user_id = normalize_optional_string_field(
        request_data.get("user_id"),
        field_name="user_id",
    )

    session_id = normalize_optional_string_field(
        request_data.get("session_id"),
        field_name="session_id",
    )

    personalization_user_id = (
        user_id
        or session_id
    )

    top_k = parse_integer(
        value=request_data.get("top_k"),
        field_name="top_k",
        default=DEFAULT_TOP_K,
        minimum=1,
        maximum=MAX_TOP_K,
    )

    bm25_k1 = parse_float(
        value=request_data.get("bm25_k1"),
        field_name="bm25_k1",
        default=DEFAULT_BM25_K1,
        minimum=0.000001,
        maximum=100.0,
    )

    bm25_b = parse_float(
        value=request_data.get("bm25_b"),
        field_name="bm25_b",
        default=DEFAULT_BM25_B,
        minimum=0.0,
        maximum=1.0,
    )

    candidate_count = parse_integer(
        value=request_data.get(
            "candidate_count"
        ),
        field_name="candidate_count",
        default=DEFAULT_CANDIDATE_COUNT,
        minimum=1,
        maximum=MAX_CANDIDATE_COUNT,
    )

    rrf_k = parse_integer(
        value=request_data.get("rrf_k"),
        field_name="rrf_k",
        default=DEFAULT_RRF_K,
        minimum=1,
        maximum=MAX_RRF_K,
    )

    tfidf_weight = parse_float(
        value=request_data.get("tfidf_weight"),
        field_name="tfidf_weight",
        default=DEFAULT_TFIDF_WEIGHT,
        minimum=0.0,
        maximum=MAX_HYBRID_WEIGHT,
    )

    bm25_weight = parse_float(
        value=request_data.get("bm25_weight"),
        field_name="bm25_weight",
        default=DEFAULT_BM25_WEIGHT,
        minimum=0.0,
        maximum=MAX_HYBRID_WEIGHT,
    )

    embedding_weight = parse_float(
        value=request_data.get("embedding_weight"),
        field_name="embedding_weight",
        default=DEFAULT_EMBEDDING_WEIGHT,
        minimum=0.0,
        maximum=MAX_HYBRID_WEIGHT,
    )

    biomedical_weight = parse_float(
        value=request_data.get("biomedical_weight"),
        field_name="biomedical_weight",
        default=DEFAULT_BIOMEDICAL_WEIGHT,
        minimum=0.0,
        maximum=MAX_HYBRID_WEIGHT,
    )

    if (
        model == "biomedical_embedding"
        and dataset_key != "clinical_trials"
    ):
        raise ValueError(
            "biomedical_embedding is only supported for the "
            "clinical_trials dataset."
        )

    if (
        model == "hybrid_parallel"
        and biomedical_weight > 0
        and dataset_key != "clinical_trials"
    ):
        raise ValueError(
            "biomedical_weight can only be used with the "
            "clinical_trials dataset."
        )

    use_query_refinement = parse_boolean(
        value=request_data.get(
            "use_query_refinement",
            False,
        ),
        field_name="use_query_refinement",
    )

    use_spelling_correction = parse_boolean(
        value=request_data.get(
            "use_spelling_correction",
            DEFAULT_USE_SPELLING_CORRECTION,
        ),
        field_name="use_spelling_correction",
    )

    feedback_docs = parse_integer(
        value=request_data.get(
            "feedback_docs"
        ),
        field_name="feedback_docs",
        default=DEFAULT_FEEDBACK_DOCS,
        minimum=1,
        maximum=MAX_FEEDBACK_DOCS,
    )

    expansion_terms = parse_integer(
        value=request_data.get(
            "expansion_terms"
        ),
        field_name="expansion_terms",
        default=DEFAULT_EXPANSION_TERMS,
        minimum=1,
        maximum=MAX_EXPANSION_TERMS,
    )

    snippet_length = parse_integer(
        value=request_data.get(
            "snippet_length"
        ),
        field_name="snippet_length",
        default=DEFAULT_SNIPPET_LENGTH,
        minimum=1,
        maximum=MAX_SNIPPET_LENGTH,
    )

    include_raw_text = parse_boolean(
        value=request_data.get(
            "include_raw_text",
            DEFAULT_INCLUDE_RAW_TEXT,
        ),
        field_name="include_raw_text",
    )

    use_personalization = parse_boolean(
        value=request_data.get(
            "use_personalization",
            False,
        ),
        field_name="use_personalization",
    )

    max_personalization_terms = parse_integer(
        value=request_data.get(
            "max_personalization_terms"
        ),
        field_name="max_personalization_terms",
        default=DEFAULT_MAX_PERSONALIZATION_TERMS,
        minimum=1,
        maximum=MAX_PERSONALIZATION_TERMS,
    )

    if (
        model
        in {
            "hybrid_serial",
            "hybrid_parallel",
        }
        and candidate_count < top_k
    ):
        raise ValueError(
            "'candidate_count' must be greater than "
            "or equal to 'top_k' for hybrid models."
        )

    if (
        model == "hybrid_parallel"
        and (
            tfidf_weight
            + bm25_weight
            + embedding_weight
            + biomedical_weight
        )
        <= 0
    ):
        raise ValueError(
            "At least one hybrid parallel weight must be greater than 0."
        )

    if (
        include_raw_text
        and top_k > MAX_RAW_TEXT_RESULTS
    ):
        raise ValueError(
            "'include_raw_text' can only be used "
            f"when 'top_k' does not exceed "
            f"{MAX_RAW_TEXT_RESULTS}."
        )

    return {
        "query": query,
        "dataset_key": dataset_key,
        "model": model,
        "top_k": top_k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "candidate_count": candidate_count,
        "rrf_k": rrf_k,
        "tfidf_weight": tfidf_weight,
        "bm25_weight": bm25_weight,
        "embedding_weight": embedding_weight,
        "biomedical_weight": biomedical_weight,
        "use_query_refinement": (
            use_query_refinement
        ),
        "use_spelling_correction": (
            use_spelling_correction
        ),
        "feedback_docs": feedback_docs,
        "expansion_terms": expansion_terms,
        "snippet_length": snippet_length,
        "include_raw_text": include_raw_text,
        "user_id": personalization_user_id,
        "use_personalization": use_personalization,
        "max_personalization_terms": (
            max_personalization_terms
        ),
    }


def resolve_agent_model(
    requested_model: str,
    dataset_key: str,
    query: str,
    use_query_refinement: bool,
) -> Dict[str, Any]:
    if requested_model != "agent":
        return {
            "requested_model": requested_model,
            "executed_model": requested_model,
            "agent_selected_model": None,
            "agent_reason": None,
            "agent_features": None,
            "agent_fallback": None,
        }

    agent = get_retrieval_strategy_agent()

    decision = agent.decide(
        dataset_key=dataset_key,
        query=query,
        requested_features={
            "use_query_refinement": use_query_refinement,
        },
    )

    agent_selected_model = decision.selected_model
    executed_model = agent_selected_model
    agent_fallback = None

    if executed_model == "multilingual":
        executed_model = "embedding"
        agent_fallback = (
            "The agent selected multilingual retrieval, but the "
            "multilingual service is not implemented yet. Falling back "
            "to embedding retrieval."
        )

    if (
        executed_model not in SUPPORTED_MODELS
        or executed_model == "agent"
    ):
        executed_model = "bm25"
        agent_fallback = (
            "The agent selected an unavailable model. Falling back to BM25."
        )

    return {
        "requested_model": requested_model,
        "executed_model": executed_model,
        "agent_selected_model": agent_selected_model,
        "agent_reason": decision.reason,
        "agent_features": decision.features,
        "agent_fallback": agent_fallback,
    }


def run_search(
    model: str,
    dataset_key: str,
    query: str,
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
    candidate_count: int,
    rrf_k: int,
    tfidf_weight: float,
    bm25_weight: float,
    embedding_weight: float,
    biomedical_weight: float = DEFAULT_BIOMEDICAL_WEIGHT,
):
    if model == "tfidf":
        service = get_tfidf_service(
            dataset_key=dataset_key
        )

        return service.search(
            query=query,
            top_k=top_k,
        )

    if model == "bm25":
        service = get_bm25_service(
            dataset_key=dataset_key,
            k1=bm25_k1,
            b=bm25_b,
        )

        return service.search(
            query=query,
            top_k=top_k,
        )

    if model == "embedding":
        service = get_embedding_service(
            dataset_key=dataset_key
        )

        return service.search(
            query=query,
            top_k=top_k,
        )

    if model == "biomedical_embedding":
        service = get_biomedical_embedding_service(
            dataset_key=dataset_key
        )

        return service.search(
            query=query,
            top_k=top_k,
        )

    if model == "hybrid_serial":
        service = get_hybrid_serial_service(
            dataset_key=dataset_key,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            candidate_count=candidate_count,
        )

        return service.search(
            query=query,
            top_k=top_k,
        )

    if model == "hybrid_parallel":
        service = get_hybrid_parallel_service(
            dataset_key=dataset_key,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            rrf_k=rrf_k,
            candidate_count=candidate_count,
            tfidf_weight=tfidf_weight,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
            biomedical_weight=biomedical_weight,
        )

        return service.search(
            query=query,
            top_k=top_k,
        )

    raise ValueError(
        f"Unsupported model '{model}'."
    )


@api_view(["GET"])
def datasets_view(request):
    return Response({
        "datasets": list_datasets(),
        "models": sorted(SUPPORTED_MODELS),
    })


@api_view(["GET"])
def document_view(
    request,
    dataset_key: str,
    doc_id: str,
):
    try:
        dataset_key = normalize_string_field(
            dataset_key,
            field_name="dataset",
        ).lower()

        doc_id = normalize_string_field(
            doc_id,
            field_name="doc_id",
        )

        validate_dataset(dataset_key)

        document = get_raw_document(
            dataset_key=dataset_key,
            doc_id=doc_id,
        )

        if document is None:
            return Response(
                {
                    "error": (
                        "Document not found in the "
                        "offline document store."
                    ),
                },
                status=404,
            )

        return Response({
            "dataset": dataset_key,
            "doc_id": document["doc_id"],
            "title": document.get(
                "title",
                "",
            ),
            "raw_text": document.get(
                "raw_text",
                "",
            ),
            "metadata": document.get(
                "metadata",
                {},
            ),
            "document_source": (
                "sqlite_document_store"
            ),
        })

    except ValueError as error:
        return Response(
            {
                "error": str(error),
            },
            status=400,
        )

    except DocumentStoreError as error:
        logger.exception(
            "Document-store error while retrieving "
            "a raw document."
        )

        return Response(
            {
                "error": str(error),
            },
            status=500,
        )

    except FileNotFoundError as error:
        return Response(
            {
                "error": str(error),
            },
            status=404,
        )

    except Exception:
        logger.exception(
            "Unexpected document API error."
        )

        return Response(
            {
                "error": (
                    "An unexpected server error occurred."
                ),
            },
            status=500,
        )


@api_view(["POST"])
def search_view(request):
    try:
        parameters = parse_search_request(
            request.data
        )

        original_query = parameters["query"]
        corrected_query = original_query
        personalized_query = original_query
        refined_query = original_query
        retrieval_query = original_query
        spelling_corrections = []
        spelling_correction_used = False
        personalization_terms = []
        personalization_used = False

        if parameters["use_spelling_correction"]:
            spelling_service = (
                get_spelling_correction_service(
                    parameters["dataset_key"]
                )
            )

            spelling_result = spelling_service.correct(
                original_query
            )

            corrected_query = str(
                spelling_result["corrected_query"]
            )

            spelling_corrections = list(
                spelling_result[
                    "spelling_corrections"
                ]
            )

            spelling_correction_used = bool(
                spelling_result[
                    "spelling_correction_used"
                ]
            )

        retrieval_query = corrected_query
        personalized_query = retrieval_query

        if (
            parameters["use_personalization"]
            and parameters["user_id"]
        ):
            store = get_search_history_store()

            previous_queries = store.get_recent_queries(
                user_id=parameters["user_id"],
                dataset=parameters["dataset_key"],
                limit=DEFAULT_PERSONALIZATION_HISTORY_LIMIT,
            )

            personalization_service = (
                get_personalized_query_service(
                    dataset_key=parameters[
                        "dataset_key"
                    ],
                    max_personalization_terms=(
                        parameters[
                            "max_personalization_terms"
                        ]
                    ),
                )
            )

            personalization_result = (
                personalization_service.personalize(
                    query=retrieval_query,
                    previous_queries=previous_queries,
                )
            )

            personalized_query = str(
                personalization_result[
                    "personalized_query"
                ]
            )

            personalization_terms = list(
                personalization_result[
                    "personalization_terms"
                ]
            )

            personalization_used = bool(
                personalization_terms
            )

        retrieval_query = personalized_query

        if parameters["use_query_refinement"]:
            refinement_service = (
                get_query_refinement_service(
                    dataset_key=(
                        parameters["dataset_key"]
                    ),
                    bm25_k1=(
                        parameters["bm25_k1"]
                    ),
                    bm25_b=(
                        parameters["bm25_b"]
                    ),
                    feedback_docs=(
                        parameters["feedback_docs"]
                    ),
                    expansion_terms=(
                        parameters["expansion_terms"]
                    ),
                )
            )

            refined_query = (
                refinement_service.refine(
                    retrieval_query
                )
            )

            retrieval_query = refined_query

        else:
            refined_query = original_query

        if not parameters["use_personalization"]:
            personalized_query = corrected_query

        agent_resolution = resolve_agent_model(
            requested_model=parameters["model"],
            dataset_key=parameters["dataset_key"],
            query=original_query,
            use_query_refinement=parameters[
                "use_query_refinement"
            ],
        )

        executed_model = agent_resolution["executed_model"]

        if (
            executed_model
            in {
                "hybrid_serial",
                "hybrid_parallel",
            }
            and parameters["candidate_count"]
            < parameters["top_k"]
        ):
            raise ValueError(
                "'candidate_count' must be greater than "
                "or equal to 'top_k' for hybrid models."
            )

        if (
            executed_model == "hybrid_parallel"
            and (
                parameters["tfidf_weight"]
                + parameters["bm25_weight"]
                + parameters["embedding_weight"]
                + parameters["biomedical_weight"]
            )
            <= 0
        ):
            raise ValueError(
                "At least one hybrid parallel weight must be greater than 0."
            )

        if (
            executed_model == "hybrid_parallel"
            and parameters["biomedical_weight"] > 0
            and parameters["dataset_key"] != "clinical_trials"
        ):
            raise ValueError(
                "biomedical_weight can only be used with the "
                "clinical_trials dataset."
            )

        if (
            executed_model == "biomedical_embedding"
            and parameters["dataset_key"] != "clinical_trials"
        ):
            raise ValueError(
                "biomedical_embedding is only supported for the "
                "clinical_trials dataset."
            )

        results = run_search(
            model=executed_model,
            dataset_key=(
                parameters["dataset_key"]
            ),
            query=retrieval_query,
            top_k=parameters["top_k"],
            bm25_k1=parameters["bm25_k1"],
            bm25_b=parameters["bm25_b"],
            candidate_count=(
                parameters["candidate_count"]
            ),
            rrf_k=parameters["rrf_k"],
            tfidf_weight=parameters["tfidf_weight"],
            bm25_weight=parameters["bm25_weight"],
            embedding_weight=parameters["embedding_weight"],
            biomedical_weight=parameters["biomedical_weight"],
        )

        results = enrich_search_results(
            dataset_key=parameters[
                "dataset_key"
            ],
            results=results,
            snippet_length=parameters[
                "snippet_length"
            ],
            include_raw_text=parameters[
                "include_raw_text"
            ],
            strict=True,
        )

        record_search_history_safely(
            user_id=parameters["user_id"],
            dataset_key=parameters["dataset_key"],
            query=original_query,
        )

        model_uses_bm25 = (
            executed_model
            in {
                "bm25",
                "hybrid_serial",
                "hybrid_parallel",
            }
        )

        model_is_hybrid = (
            executed_model
            in {
                "hybrid_serial",
                "hybrid_parallel",
            }
        )

        model_is_weighted_parallel = (
            executed_model == "hybrid_parallel"
        )

        document_source = (
            "sqlite_document_store"
            if document_store_is_required(
                parameters["dataset_key"]
            )
            else "retrieval_index"
        )

        return Response({
            "query": retrieval_query,
            "original_query": original_query,
            "corrected_query": corrected_query,
            "spelling_correction_used": (
                spelling_correction_used
            ),
            "spelling_corrections": (
                spelling_corrections
            ),
            "use_spelling_correction": parameters[
                "use_spelling_correction"
            ],
            "refined_query": refined_query,
            "personalized_query": personalized_query,
            "personalization_terms": (
                personalization_terms
            ),
            "use_personalization": parameters[
                "use_personalization"
            ],
            "personalization_used": (
                personalization_used
            ),
            "max_personalization_terms": parameters[
                "max_personalization_terms"
            ],
            "dataset": parameters[
                "dataset_key"
            ],
            "requested_model": agent_resolution[
                "requested_model"
            ],
            "model": executed_model,
            "executed_model": executed_model,
            "agent_selected_model": agent_resolution[
                "agent_selected_model"
            ],
            "agent_reason": agent_resolution[
                "agent_reason"
            ],
            "agent_features": agent_resolution[
                "agent_features"
            ],
            "agent_fallback": agent_resolution[
                "agent_fallback"
            ],
            "top_k": parameters["top_k"],
            "bm25_k1": (
                parameters["bm25_k1"]
                if model_uses_bm25
                else None
            ),
            "bm25_b": (
                parameters["bm25_b"]
                if model_uses_bm25
                else None
            ),
            "candidate_count": (
                parameters["candidate_count"]
                if model_is_hybrid
                else None
            ),
            "rrf_k": (
                parameters["rrf_k"]
                if model_is_weighted_parallel
                else None
            ),
            "tfidf_weight": (
                parameters["tfidf_weight"]
                if model_is_weighted_parallel
                else None
            ),
            "bm25_weight": (
                parameters["bm25_weight"]
                if model_is_weighted_parallel
                else None
            ),
            "embedding_weight": (
                parameters["embedding_weight"]
                if model_is_weighted_parallel
                else None
            ),
            "biomedical_weight": (
                parameters["biomedical_weight"]
                if model_is_weighted_parallel
                else None
            ),
            "fusion_method": (
                "Weighted RRF"
                if model_is_weighted_parallel
                else None
            ),
            "use_query_refinement": (
                parameters[
                    "use_query_refinement"
                ]
            ),
            "feedback_docs": (
                parameters["feedback_docs"]
                if parameters[
                    "use_query_refinement"
                ]
                else None
            ),
            "expansion_terms": (
                parameters["expansion_terms"]
                if parameters[
                    "use_query_refinement"
                ]
                else None
            ),
            "snippet_length": parameters[
                "snippet_length"
            ],
            "include_raw_text": parameters[
                "include_raw_text"
            ],
            "document_source": document_source,
            "result_count": len(results),
            "results": results,
        })

    except ValueError as error:
        return Response(
            {
                "error": str(error),
                "results": [],
            },
            status=400,
        )

    except DocumentStoreError as error:
        logger.exception(
            "Document-store error while enriching "
            "search results."
        )

        return Response(
            {
                "error": str(error),
                "results": [],
            },
            status=500,
        )

    except FileNotFoundError as error:
        return Response(
            {
                "error": str(error),
                "results": [],
            },
            status=404,
        )

    except RuntimeError as error:
        logger.exception(
            "Retrieval runtime error."
        )

        return Response(
            {
                "error": str(error),
                "results": [],
            },
            status=500,
        )

    except Exception:
        logger.exception(
            "Unexpected search API error."
        )

        return Response(
            {
                "error": (
                    "An unexpected server error occurred."
                ),
                "results": [],
            },
            status=500,
        )
