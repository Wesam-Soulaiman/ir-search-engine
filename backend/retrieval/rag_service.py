from typing import Any, Callable, Dict, List

from retrieval.rag_answer_generator import (
    GENERATION_MODE,
    OfflineExtractiveRAGAnswerGenerator,
)


DEFAULT_RAG_RETRIEVER_MODEL = "hybrid_serial"
DEFAULT_RAG_CONTEXT_DOCS = 5
DEFAULT_RAG_ANSWER_SENTENCES = 4
MAX_RAG_CONTEXT_DOCS = 50
MAX_RAG_ANSWER_SENTENCES = 10

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
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()
        self.retriever = retriever
        self.answer_generator = (
            answer_generator
            or OfflineExtractiveRAGAnswerGenerator()
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

        answer_payload = (
            self.answer_generator.generate(
                query=query,
                retrieved_results=results,
                max_context_docs=parsed_context_docs,
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
            "retriever_metadata": (
                retriever_metadata
            ),
        }

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
