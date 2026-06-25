import math
from typing import Dict, List, Set


def precision_at_k(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: Set[str],
    k: int = 10
) -> float:
    """
    Precision@K = relevant retrieved documents in top K / K
    """
    if k <= 0:
        return 0.0

    top_k_docs = retrieved_doc_ids[:k]

    if not top_k_docs:
        return 0.0

    relevant_retrieved = sum(
        1 for doc_id in top_k_docs if doc_id in relevant_doc_ids
    )

    return relevant_retrieved / k


def recall_at_k(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: Set[str],
    k: int = 10
) -> float:
    """
    Recall@K = relevant retrieved documents in top K / all relevant documents
    """
    if not relevant_doc_ids:
        return 0.0

    top_k_docs = retrieved_doc_ids[:k]

    relevant_retrieved = sum(
        1 for doc_id in top_k_docs if doc_id in relevant_doc_ids
    )

    return relevant_retrieved / len(relevant_doc_ids)


def average_precision(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: Set[str]
) -> float:
    """
    Average Precision:
    Average of Precision@rank for each relevant retrieved document.
    """
    if not relevant_doc_ids:
        return 0.0

    relevant_hits = 0
    precision_sum = 0.0

    for index, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            relevant_hits += 1
            precision_sum += relevant_hits / index

    if relevant_hits == 0:
        return 0.0

    return precision_sum / len(relevant_doc_ids)


def mean_average_precision(
    retrieved_results: Dict[str, List[str]],
    qrels: Dict[str, Set[str]]
) -> float:
    """
    MAP = mean of Average Precision over all evaluated queries.
    retrieved_results format:
        {
            "query_id": ["doc1", "doc2", ...]
        }

    qrels format:
        {
            "query_id": {"doc1", "doc5", ...}
        }
    """
    if not qrels:
        return 0.0

    ap_scores = []

    for query_id, relevant_doc_ids in qrels.items():
        retrieved_doc_ids = retrieved_results.get(query_id, [])
        ap_scores.append(
            average_precision(retrieved_doc_ids, relevant_doc_ids)
        )

    if not ap_scores:
        return 0.0

    return sum(ap_scores) / len(ap_scores)


def dcg_at_k(
    retrieved_doc_ids: List[str],
    relevance_scores: Dict[str, int],
    k: int = 10
) -> float:
    """
    DCG@K = sum((2^rel - 1) / log2(rank + 1))
    """
    if k <= 0:
        return 0.0

    dcg = 0.0

    for index, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        relevance = relevance_scores.get(doc_id, 0)

        if relevance <= 0:
            continue

        dcg += (math.pow(2, relevance) - 1) / math.log2(index + 1)

    return dcg


def ndcg_at_k(
    retrieved_doc_ids: List[str],
    relevance_scores: Dict[str, int],
    k: int = 10
) -> float:
    """
    nDCG@K = DCG@K / Ideal DCG@K
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    actual_dcg = dcg_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevance_scores=relevance_scores,
        k=k
    )

    ideal_relevances = sorted(
        relevance_scores.values(),
        reverse=True
    )

    ideal_dcg = 0.0

    for index, relevance in enumerate(ideal_relevances[:k], start=1):
        if relevance <= 0:
            continue

        ideal_dcg += (math.pow(2, relevance) - 1) / math.log2(index + 1)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg