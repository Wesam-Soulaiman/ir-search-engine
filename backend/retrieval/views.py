import os

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from retrieval.bm25_service import BM25RetrievalService
from retrieval.embedding_service import EmbeddingRetrievalService
from retrieval.tfidf_service import TfidfRetrievalService


_tfidf_service = None
_embedding_service = None
_bm25_services = {}


def get_dataset_path():
    project_root = settings.BASE_DIR.parent
    return os.path.join(
        project_root,
        "data",
        "sample_dataset",
        "documents.json"
    )


def get_tfidf_service():
    global _tfidf_service

    if _tfidf_service is None:
        _tfidf_service = TfidfRetrievalService(get_dataset_path())

    return _tfidf_service


def get_bm25_service(k1: float = 1.5, b: float = 0.75):
    global _bm25_services

    key = f"k1={k1}_b={b}"

    if key not in _bm25_services:
        _bm25_services[key] = BM25RetrievalService(
            dataset_path=get_dataset_path(),
            k1=k1,
            b=b
        )

    return _bm25_services[key]


def get_embedding_service():
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingRetrievalService(get_dataset_path())

    return _embedding_service


@api_view(["POST"])
def search_view(request):
    query = request.data.get("query", "")
    dataset = request.data.get("dataset", "sample_dataset")
    model = request.data.get("model", "tfidf")
    top_k = int(request.data.get("top_k", 10))

    bm25_k1 = float(request.data.get("bm25_k1", 1.5))
    bm25_b = float(request.data.get("bm25_b", 0.75))

    if not query.strip():
        return Response({
            "error": "Query cannot be empty.",
            "results": []
        }, status=400)

    if model == "tfidf":
        service = get_tfidf_service()
        results = service.search(query=query, top_k=top_k)

    elif model == "bm25":
        service = get_bm25_service(k1=bm25_k1, b=bm25_b)
        results = service.search(query=query, top_k=top_k)

    elif model == "embedding":
        service = get_embedding_service()
        results = service.search(query=query, top_k=top_k)

    else:
        results = [
            {
                "rank": 1,
                "doc_id": "TEMP-001",
                "title": "Temporary result",
                "snippet": f"Model {model} is not implemented yet.",
                "score": 0.0,
            }
        ]

    return Response({
        "query": query,
        "dataset": dataset,
        "model": model,
        "top_k": top_k,
        "bm25_k1": bm25_k1 if model == "bm25" else None,
        "bm25_b": bm25_b if model == "bm25" else None,
        "results": results,
    })