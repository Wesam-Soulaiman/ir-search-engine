from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(["POST"])
def search_view(request):
    query = request.data.get("query", "")
    dataset = request.data.get("dataset", "dataset1")
    model = request.data.get("model", "bm25")
    top_k = int(request.data.get("top_k", 10))

    results = [
        {
            "rank": 1,
            "doc_id": "DOC-001",
            "title": "Sample result 1",
            "snippet": f"This is a temporary result for query: {query}",
            "score": 1.0,
        },
        {
            "rank": 2,
            "doc_id": "DOC-002",
            "title": "Sample result 2",
            "snippet": f"Dataset selected: {dataset}, model selected: {model}",
            "score": 0.85,
        },
    ]

    return Response({
        "query": query,
        "dataset": dataset,
        "model": model,
        "top_k": top_k,
        "results": results[:top_k],
    })