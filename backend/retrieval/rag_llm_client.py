import json
import urllib.error
import urllib.request
from typing import Any, Dict, List

from retrieval.rag_answer_generator import (
    INSUFFICIENT_CONTEXT_ANSWER,
)


DEFAULT_RAG_LLM_PROVIDER = "ollama"
DEFAULT_RAG_LLM_MODEL = "llama3.2:3b"
DEFAULT_RAG_LLM_BASE_URL = "http://localhost:11434"
DEFAULT_RAG_LLM_TEMPERATURE = 0.0
DEFAULT_RAG_LLM_MAX_TOKENS = 350
DEFAULT_RAG_LLM_TIMEOUT_SECONDS = 60
MAX_RAG_LLM_MAX_TOKENS = 4096
MAX_RAG_LLM_TEMPERATURE = 2.0

OLLAMA_SYSTEM_PROMPT = (
    "You are an Information Retrieval assistant. "
    "In this project, RAG always means Retrieval-Augmented Generation. "
    "Answer only from the provided retrieved context. "
    "Do not invent facts. "
    "Use citations like [1], [2], [3]. "
    "If the context does not answer the question, say the retrieved "
    "documents do not contain enough information to answer confidently. "
    "Keep the answer concise. "
    "Prefer English unless the user query is clearly in another language."
)


class LocalLLMGenerationError(RuntimeError):
    """
    Raised when local LLM generation cannot complete.
    """


class OllamaChatClient:
    """
    Minimal standard-library client for a local Ollama /api/chat server.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_RAG_LLM_BASE_URL,
        model: str = DEFAULT_RAG_LLM_MODEL,
        timeout_seconds: int = DEFAULT_RAG_LLM_TIMEOUT_SECONDS,
    ):
        self.base_url = normalize_ollama_base_url(
            base_url
        )
        self.model = str(
            model or DEFAULT_RAG_LLM_MODEL
        ).strip()
        self.timeout_seconds = int(
            timeout_seconds
        )

    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        temperature: float = DEFAULT_RAG_LLM_TEMPERATURE,
        max_tokens: int = DEFAULT_RAG_LLM_MAX_TOKENS,
    ) -> str:
        payload = build_ollama_chat_payload(
            query=query,
            context_docs=context_docs,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(
                "utf-8"
            ),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_payload = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as error:
            raise LocalLLMGenerationError(
                local_llm_unavailable_message(
                    base_url=self.base_url,
                    model=self.model,
                )
            ) from error

        content = extract_ollama_message(
            response_payload
        )

        if not content:
            raise LocalLLMGenerationError(
                "Local LLM generation returned an empty response."
            )

        return content


def normalize_ollama_base_url(
    base_url: str,
) -> str:
    normalized = str(
        base_url or DEFAULT_RAG_LLM_BASE_URL
    ).strip()

    return normalized.rstrip("/")


def build_ollama_chat_payload(
    query: str,
    context_docs: List[Dict[str, Any]],
    model: str = DEFAULT_RAG_LLM_MODEL,
    temperature: float = DEFAULT_RAG_LLM_TEMPERATURE,
    max_tokens: int = DEFAULT_RAG_LLM_MAX_TOKENS,
) -> Dict[str, Any]:
    return {
        "model": str(
            model or DEFAULT_RAG_LLM_MODEL
        ).strip(),
        "messages": [
            {
                "role": "system",
                "content": OLLAMA_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Question: "
                    f"{str(query or '').strip()}\n\n"
                    "Context:\n"
                    f"{build_ollama_context(context_docs)}"
                ),
            },
        ],
        "stream": False,
        "options": {
            "temperature": float(
                temperature
            ),
            "num_predict": int(
                max_tokens
            ),
        },
    }


def build_ollama_context(
    context_docs: List[Dict[str, Any]],
) -> str:
    if not context_docs:
        return (
            "[No retrieved context]\n"
            f"{INSUFFICIENT_CONTEXT_ANSWER}"
        )

    blocks = []

    for source_id, result in enumerate(
        context_docs,
        start=1,
    ):
        title = str(
            result.get("title", "")
            or f"Document {result.get('doc_id', '')}"
        ).strip()
        snippet = str(
            result.get(
                "snippet",
                result.get(
                    "raw_text",
                    result.get("text", ""),
                ),
            )
            or ""
        ).strip()
        snippet = snippet[:1800]

        blocks.append(
            f"[{source_id}] Title: {title}\n"
            f"Snippet: {snippet}"
        )

    return "\n\n".join(blocks)


def extract_ollama_message(
    response_payload: Dict[str, Any],
) -> str:
    message = response_payload.get(
        "message",
        {},
    )

    if isinstance(message, dict):
        content = message.get(
            "content"
        )

        if content:
            return str(content).strip()

    response = response_payload.get(
        "response"
    )

    return str(response or "").strip()


def local_llm_unavailable_message(
    base_url: str,
    model: str,
) -> str:
    return (
        "Local LLM generation requires Ollama running at "
        f"{normalize_ollama_base_url(base_url)} and model "
        f"{model} pulled."
    )
