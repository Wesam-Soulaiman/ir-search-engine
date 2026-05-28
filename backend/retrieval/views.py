import os

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from retrieval.tfidf_service import TfidfRetrievalService


_tfidf_service = None


def get_tfidf_service():
    global _tfidf_service

    if _tfidf_service is None:
        project_root = settings.BASE_DIR.parent
        dataset_path = os.path.join(
            project_root,
            "data",
            "sample_dataset",
            "documents.json"
        )
        _tfidf_service = TfidfRetrievalService(dataset_path)

    return _tfidf_service


@api_view(["POST"])
def search_view(request):
    query = request.data.get("query", "")
    dataset = request.data.get("dataset", "sample_dataset")
    model = request.data.get("model", "tfidf")
    top_k = int(request.data.get("top_k", 10))

    if not query.strip():
        return Response({
            "error": "Query cannot be empty.",
            "results": []
        }, status=400)

    if model == "tfidf":
        service = get_tfidf_service()
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
        "results": results,
    })