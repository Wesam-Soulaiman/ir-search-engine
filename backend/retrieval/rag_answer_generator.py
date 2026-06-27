import re
from typing import Any, Dict, List


INSUFFICIENT_CONTEXT_ANSWER = (
    "The retrieved documents do not contain enough information "
    "to answer confidently."
)

GENERATION_MODE = "offline_extractive_grounded"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class OfflineExtractiveRAGAnswerGenerator:
    """
    Deterministic grounded answer synthesis over retrieved snippets.
    """

    def generate(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        max_context_docs: int = 5,
        max_answer_sentences: int = 4,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        context_docs = self._context_documents(
            retrieved_results=retrieved_results,
            max_context_docs=max_context_docs,
        )
        query_terms = self._content_tokens(query)

        if not query_terms or not context_docs:
            return self._insufficient_response(
                context_docs=context_docs,
                include_sources=include_sources,
            )

        scored_sentences = []

        for source_id, result in enumerate(
            context_docs,
            start=1,
        ):
            text = self._result_text(result)
            sentences = self._split_sentences(text)

            for sentence_index, sentence in enumerate(
                sentences
            ):
                sentence_terms = self._content_tokens(
                    sentence
                )
                overlap = query_terms & sentence_terms

                if not overlap:
                    continue

                rank = int(
                    result.get(
                        "rank",
                        source_id,
                    )
                    or source_id
                )
                overlap_ratio = (
                    len(overlap)
                    / max(len(query_terms), 1)
                )
                score = (
                    len(overlap) * 2.0
                    + overlap_ratio
                    + (1.0 / max(rank, 1))
                )

                scored_sentences.append({
                    "sentence": sentence,
                    "source_id": source_id,
                    "rank": rank,
                    "position": sentence_index,
                    "score": score,
                    "overlap_count": len(overlap),
                })

        if not scored_sentences:
            return self._insufficient_response(
                context_docs=context_docs,
                include_sources=include_sources,
            )

        scored_sentences.sort(
            key=lambda item: (
                -float(item["score"]),
                int(item["rank"]),
                int(item["position"]),
                item["sentence"].lower(),
            )
        )

        selected = []
        seen_sentences = set()

        for item in scored_sentences:
            normalized_sentence = self._normalize_sentence(
                item["sentence"]
            )

            if normalized_sentence in seen_sentences:
                continue

            seen_sentences.add(
                normalized_sentence
            )
            selected.append(item)

            if len(selected) >= max_answer_sentences:
                break

        if not selected:
            return self._insufficient_response(
                context_docs=context_docs,
                include_sources=include_sources,
            )

        selected.sort(
            key=lambda item: (
                int(item["source_id"]),
                int(item["position"]),
            )
        )

        answer_parts = []

        for item in selected:
            citation = (
                f" [{item['source_id']}]"
                if include_sources
                else ""
            )
            answer_parts.append(
                f"{item['sentence']}{citation}"
            )

        used_source_ids = {
            int(item["source_id"])
            for item in selected
        }
        total_overlap = sum(
            int(item["overlap_count"])
            for item in selected
        )

        confidence = self._confidence(
            selected_count=len(selected),
            source_count=len(used_source_ids),
            total_overlap=total_overlap,
        )

        return {
            "answer": " ".join(answer_parts),
            "confidence": confidence,
            "sources": (
                self._sources(
                    context_docs=context_docs,
                    used_source_ids=used_source_ids,
                )
                if include_sources
                else []
            ),
            "context_docs_used": len(context_docs),
            "generation_mode": GENERATION_MODE,
            "external_llm_used": False,
        }

    def _insufficient_response(
        self,
        context_docs: List[Dict[str, Any]],
        include_sources: bool,
    ) -> Dict[str, Any]:
        return {
            "answer": INSUFFICIENT_CONTEXT_ANSWER,
            "confidence": "insufficient",
            "sources": (
                self._sources(
                    context_docs=context_docs,
                    used_source_ids=set(),
                )
                if include_sources
                else []
            ),
            "context_docs_used": len(context_docs),
            "generation_mode": GENERATION_MODE,
            "external_llm_used": False,
        }

    @staticmethod
    def _context_documents(
        retrieved_results: List[Dict[str, Any]],
        max_context_docs: int,
    ) -> List[Dict[str, Any]]:
        limit = max(
            int(max_context_docs),
            0,
        )

        return list(
            retrieved_results[:limit]
        )

    @staticmethod
    def _result_text(
        result: Dict[str, Any],
    ) -> str:
        parts = [
            result.get("title", ""),
            result.get("snippet", ""),
            result.get("raw_text", ""),
            result.get("text", ""),
        ]
        text = " ".join(
            str(part).strip()
            for part in parts
            if str(part or "").strip()
        )

        return text[:4000]

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> List[str]:
        cleaned_text = re.sub(
            r"\s+",
            " ",
            str(text or "").strip(),
        )

        if not cleaned_text:
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                cleaned_text,
            )
            if sentence.strip()
        ]

        return sentences or [cleaned_text]

    @staticmethod
    def _content_tokens(
        text: str,
    ) -> set[str]:
        tokens = {
            token
            for token in re.findall(
                r"[a-zA-Z0-9]+",
                str(text or "").lower(),
            )
            if len(token) > 1
        }

        return {
            token
            for token in tokens
            if token not in STOPWORDS
        }

    @staticmethod
    def _normalize_sentence(
        sentence: str,
    ) -> str:
        return re.sub(
            r"\W+",
            " ",
            str(sentence or "").lower(),
        ).strip()

    @staticmethod
    def _sources(
        context_docs: List[Dict[str, Any]],
        used_source_ids: set[int],
    ) -> List[Dict[str, Any]]:
        sources = []

        for source_id, result in enumerate(
            context_docs,
            start=1,
        ):
            if (
                used_source_ids
                and source_id not in used_source_ids
            ):
                continue

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
                "score": result.get(
                    "score"
                ),
            })

        return sources

    @staticmethod
    def _confidence(
        selected_count: int,
        source_count: int,
        total_overlap: int,
    ) -> str:
        if selected_count <= 0:
            return "insufficient"

        if (
            source_count >= 2
            and selected_count >= 2
            and total_overlap >= 4
        ):
            return "high"

        if total_overlap >= 2:
            return "medium"

        return "low"
