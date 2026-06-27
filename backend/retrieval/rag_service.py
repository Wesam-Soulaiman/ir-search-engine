from typing import Any, Callable, Dict, List

from retrieval.rag_answer_generator import (
    GENERATION_MODE,
    INSUFFICIENT_CONTEXT_ANSWER,
    OfflineExtractiveRAGAnswerGenerator,
)
from retrieval.rag_llm_client import (
    DEFAULT_RAG_LLM_BASE_URL,
    DEFAULT_RAG_LLM_MAX_TOKENS,
    DEFAULT_RAG_LLM_MODEL,
    DEFAULT_RAG_LLM_PROVIDER,
    DEFAULT_RAG_LLM_TEMPERATURE,
    MAX_RAG_LLM_MAX_TOKENS,
    MAX_RAG_LLM_TEMPERATURE,
    OllamaChatClient,
)


DEFAULT_RAG_RETRIEVER_MODEL = "hybrid_serial"
DEFAULT_RAG_CONTEXT_DOCS = 5
DEFAULT_RAG_ANSWER_SENTENCES = 4
DEFAULT_RAG_GENERATION_MODE = "extractive_offline"
MAX_RAG_CONTEXT_DOCS = 50
MAX_RAG_ANSWER_SENTENCES = 10

RAG_GENERATION_MODES = {
    "extractive_offline",
    "local_llm",
}

RAG_LLM_PROVIDERS = {
    "ollama",
}

ALLOWED_RAG_RETRIEVER_MODELS = {
    "bm25",
    "tfidf",
    "embedding",
    "hybrid_serial",
    "hybrid_parallel",
    "ltr",
    "biomedical_embedding",
    "distributed_bm25",
}


class RAGRetrievalService:
    """
    Offline grounded RAG coordinator over existing retrieval models.
    """

    def __init__(
        self,
        dataset_key: str,
        retriever: Callable[..., Any],
        answer_generator: (
            OfflineExtractiveRAGAnswerGenerator
            | None
        ) = None,
        llm_client_factory: Callable[..., Any] | None = None,
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()
        self.retriever = retriever
        self.answer_generator = (
            answer_generator
            or OfflineExtractiveRAGAnswerGenerator()
        )
        self.llm_client_factory = (
            llm_client_factory
            or OllamaChatClient
        )
        self.last_search_metadata: Dict[
            str,
            Any,
        ] = {}

    def search(
        self,
        query: str,
        top_k: int,
        retriever_model: str = DEFAULT_RAG_RETRIEVER_MODEL,
        context_docs: int = DEFAULT_RAG_CONTEXT_DOCS,
        answer_sentences: int = DEFAULT_RAG_ANSWER_SENTENCES,
        include_sources: bool = True,
        generation_mode: str = DEFAULT_RAG_GENERATION_MODE,
        llm_provider: str = DEFAULT_RAG_LLM_PROVIDER,
        llm_model: str = DEFAULT_RAG_LLM_MODEL,
        llm_base_url: str = DEFAULT_RAG_LLM_BASE_URL,
        llm_temperature: float = DEFAULT_RAG_LLM_TEMPERATURE,
        llm_max_tokens: int = DEFAULT_RAG_LLM_MAX_TOKENS,
        retriever_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        normalized_retriever_model = (
            normalize_rag_retriever_model(
                retriever_model,
                dataset_key=self.dataset_key,
            )
        )
        parsed_context_docs = validate_rag_context_docs(
            context_docs
        )
        parsed_answer_sentences = (
            validate_rag_answer_sentences(
                answer_sentences
            )
        )
        normalized_generation_mode = (
            normalize_rag_generation_mode(
                generation_mode
            )
        )
        normalized_llm_provider = (
            normalize_rag_llm_provider(
                llm_provider
            )
        )
        parsed_llm_temperature = (
            validate_rag_llm_temperature(
                llm_temperature
            )
        )
        parsed_llm_max_tokens = (
            validate_rag_llm_max_tokens(
                llm_max_tokens
            )
        )
        normalized_llm_model = str(
            llm_model or DEFAULT_RAG_LLM_MODEL
        ).strip()
        normalized_llm_base_url = str(
            llm_base_url or DEFAULT_RAG_LLM_BASE_URL
        ).strip()

        retrieval_response = self.retriever(
            model=normalized_retriever_model,
            top_k=top_k,
            **(retriever_kwargs or {}),
        )

        results, retriever_metadata = (
            normalize_retrieval_response(
                retrieval_response
            )
        )

        if normalized_generation_mode == "local_llm":
            answer_payload = (
                self._generate_local_llm_answer(
                    query=query,
                    results=results,
                    context_docs=parsed_context_docs,
                    answer_sentences=(
                        parsed_answer_sentences
                    ),
                    include_sources=include_sources,
                    llm_provider=(
                        normalized_llm_provider
                    ),
                    llm_model=normalized_llm_model,
                    llm_base_url=(
                        normalized_llm_base_url
                    ),
                    llm_temperature=(
                        parsed_llm_temperature
                    ),
                    llm_max_tokens=(
                        parsed_llm_max_tokens
                    ),
                )
            )

        else:
            answer_payload = (
                self.answer_generator.generate(
                    query=query,
                    retrieved_results=results,
                    max_context_docs=(
                        parsed_context_docs
                    ),
                    max_answer_sentences=(
                        parsed_answer_sentences
                    ),
                    include_sources=include_sources,
                )
            )

        metadata = {
            "rag": True,
            "retriever_model": (
                normalized_retriever_model
            ),
            "rag_retriever_model": (
                normalized_retriever_model
            ),
            "rag_context_docs": (
                parsed_context_docs
            ),
            "rag_answer_sentences": (
                parsed_answer_sentences
            ),
            "include_sources": bool(
                include_sources
            ),
            "rag_generation_mode": (
                normalized_generation_mode
            ),
            "rag_llm_provider": (
                normalized_llm_provider
            ),
            "rag_llm_model": (
                normalized_llm_model
            ),
            "rag_llm_base_url": (
                normalized_llm_base_url
            ),
            "rag_llm_temperature": (
                parsed_llm_temperature
            ),
            "rag_llm_max_tokens": (
                parsed_llm_max_tokens
            ),
            "answer": answer_payload[
                "answer"
            ],
            "answer_confidence": (
                answer_payload[
                    "confidence"
                ]
            ),
            "sources": answer_payload[
                "sources"
            ],
            "context_docs_used": (
                answer_payload[
                    "context_docs_used"
                ]
            ),
            "generation_mode": GENERATION_MODE,
            "external_llm_used": False,
            "local_llm_used": False,
            "llm_provider": None,
            "llm_model": None,
            "llm_base_url": None,
            "retriever_metadata": (
                retriever_metadata
            ),
        }
        metadata.update(
            answer_payload.get(
                "metadata_overrides",
                {},
            )
        )

        self.last_search_metadata = metadata

        return {
            "results": results,
            "metadata": metadata,
        }

    def get_last_search_metadata(
        self,
    ) -> Dict[str, Any]:
        return dict(
            self.last_search_metadata
        )

    def _generate_local_llm_answer(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context_docs: int,
        answer_sentences: int,
        include_sources: bool,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        llm_temperature: float,
        llm_max_tokens: int,
    ) -> Dict[str, Any]:
        context = list(
            results[:context_docs]
        )
        sufficiency_probe = (
            self.answer_generator.generate(
                query=query,
                retrieved_results=results,
                max_context_docs=context_docs,
                max_answer_sentences=(
                    answer_sentences
                ),
                include_sources=True,
            )
        )

        metadata_overrides = {
            "generation_mode": "local_llm",
            "external_llm_used": False,
            "local_llm_used": True,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_base_url": llm_base_url,
        }

        if (
            sufficiency_probe.get("confidence")
            == "insufficient"
        ):
            return {
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "confidence": "insufficient",
                "sources": (
                    build_rag_sources(context)
                    if include_sources
                    else []
                ),
                "context_docs_used": len(context),
                "metadata_overrides": (
                    metadata_overrides
                ),
            }

        if llm_provider != "ollama":
            raise ValueError(
                "Only the ollama RAG LLM provider is supported."
            )

        client = self.llm_client_factory(
            base_url=llm_base_url,
            model=llm_model,
        )
        answer = client.generate(
            query=query,
            context_docs=context,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )

        if not answer:
            answer = INSUFFICIENT_CONTEXT_ANSWER

        return {
            "answer": answer,
            "confidence": (
                sufficiency_probe.get(
                    "confidence",
                    "medium",
                )
            ),
            "sources": (
                build_rag_sources(context)
                if include_sources
                else []
            ),
            "context_docs_used": len(context),
            "metadata_overrides": (
                metadata_overrides
            ),
        }


def normalize_rag_retriever_model(
    retriever_model: str | None,
    dataset_key: str,
) -> str:
    model = str(
        retriever_model
        or DEFAULT_RAG_RETRIEVER_MODEL
    ).strip().lower()

    if model not in ALLOWED_RAG_RETRIEVER_MODELS:
        available = ", ".join(
            sorted(ALLOWED_RAG_RETRIEVER_MODELS)
        )
        raise ValueError(
            "Unsupported RAG retriever model "
            f"'{model}'. Available retrievers: "
            f"{available}."
        )

    if (
        model == "biomedical_embedding"
        and dataset_key != "clinical_trials"
    ):
        raise ValueError(
            "biomedical_embedding RAG retrieval is only "
            "supported for the clinical_trials dataset."
        )

    return model


def normalize_rag_generation_mode(
    generation_mode: str | None,
) -> str:
    mode = str(
        generation_mode
        or DEFAULT_RAG_GENERATION_MODE
    ).strip().lower()

    if mode not in RAG_GENERATION_MODES:
        raise ValueError(
            "rag_generation_mode must be one of: "
            + ", ".join(
                sorted(RAG_GENERATION_MODES)
            )
            + "."
        )

    return mode


def normalize_rag_llm_provider(
    provider: str | None,
) -> str:
    normalized = str(
        provider or DEFAULT_RAG_LLM_PROVIDER
    ).strip().lower()

    if normalized not in RAG_LLM_PROVIDERS:
        raise ValueError(
            "rag_llm_provider must be one of: "
            + ", ".join(
                sorted(RAG_LLM_PROVIDERS)
            )
            + "."
        )

    return normalized


def validate_rag_context_docs(
    value: int,
) -> int:
    return _validate_positive_int(
        value=value,
        field_name="rag_context_docs",
        maximum=MAX_RAG_CONTEXT_DOCS,
    )


def validate_rag_answer_sentences(
    value: int,
) -> int:
    return _validate_positive_int(
        value=value,
        field_name="rag_answer_sentences",
        maximum=MAX_RAG_ANSWER_SENTENCES,
    )


def validate_rag_llm_temperature(
    value: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "rag_llm_temperature must be numeric."
        ) from error

    if parsed < 0.0:
        raise ValueError(
            "rag_llm_temperature must be at least 0."
        )

    if parsed > MAX_RAG_LLM_TEMPERATURE:
        raise ValueError(
            "rag_llm_temperature must not exceed "
            f"{MAX_RAG_LLM_TEMPERATURE}."
        )

    return parsed


def validate_rag_llm_max_tokens(
    value: int,
) -> int:
    return _validate_positive_int(
        value=value,
        field_name="rag_llm_max_tokens",
        maximum=MAX_RAG_LLM_MAX_TOKENS,
    )


def build_rag_sources(
    context_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sources = []

    for source_id, result in enumerate(
        context_docs,
        start=1,
    ):
        sources.append({
            "source_id": source_id,
            "doc_id": str(
                result.get("doc_id", "")
            ),
            "title": str(
                result.get("title", "")
                or ""
            ),
            "snippet": str(
                result.get(
                    "snippet",
                    result.get("raw_text", ""),
                )
                or ""
            )[:500],
            "rank": result.get(
                "rank",
                source_id,
            ),
            "score": result.get("score"),
        })

    return sources


def normalize_retrieval_response(
    retrieval_response: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if (
        isinstance(retrieval_response, dict)
        and "results" in retrieval_response
    ):
        return (
            list(
                retrieval_response.get(
                    "results",
                    [],
                )
            ),
            dict(
                retrieval_response.get(
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    return (
        list(retrieval_response or []),
        {},
    )


def _validate_positive_int(
    value: int,
    field_name: str,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be an integer."
        ) from error

    if parsed <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )

    if parsed > maximum:
        raise ValueError(
            f"{field_name} must not exceed {maximum}."
        )

    return parsed
