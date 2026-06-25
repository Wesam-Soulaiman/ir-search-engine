from django.urls import path

from .topic_views import (
    cluster_summary_view,
    cluster_topics_view,
)
from .views import (
    datasets_view,
    document_view,
    search_view,
)

urlpatterns = [
    path("search/", search_view, name="search"),
    path("datasets/", datasets_view, name="datasets"),
    path(
        "documents/<str:dataset_key>/<str:doc_id>/",
        document_view,
        name="document",
    ),
    path(
        "clusters/<str:dataset_key>/",
        cluster_summary_view,
        name="cluster-summary",
    ),
    path(
        "topics/<str:dataset_key>/",
        cluster_topics_view,
        name="cluster-topics",
    ),
]
