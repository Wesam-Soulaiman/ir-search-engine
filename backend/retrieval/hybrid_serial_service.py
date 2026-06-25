from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from django.conf import settings

from retrieval.bm25_service import BM25RetrievalService
from retrieval.embedding_service import EmbeddingRetrievalService


class HybridSerialRetrievalService:
    """
    Hybrid Serial Retrieval:
    1. Retrieve candidate documents using BM25.
    2. Re-rank those candidates using the already-built document
       embeddings from the scalable embedding index.

    This version does not download any Hugging Face model at runtime.
    It uses the local model and saved FAISS/memmap artifacts produced by
    build_scalable_embedding_index.py.
    """

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        candidate_count: int = 1000,
        device: str = "auto",
    ):
        self.dataset_key = dataset_key
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.candidate_count = candidate_count
        self.device = device

        self.bm25_service = BM25RetrievalService(
            dataset_key=dataset_key,
            k1=bm25_k1,
            b=bm25_b,
        )

        self.embedding_service = EmbeddingRetrievalService(
            dataset_key=dataset_key,
            use_saved_index=True,
            device=device,
        )

        self.embedding_matrix: Optional[np.memmap] = None
        self.doc_id_to_index: Dict[str, int] = {}
        self._load_document_embedding_matrix()

    def _load_document_embedding_matrix(self):
        index_dir = Path(self.embedding_service.index_dir)
        memmap_path = index_dir / "_build" / "embeddings.float32.memmap"

        if not memmap_path.is_file():
            raise FileNotFoundError(
                "Hybrid serial re-ranking requires the document embedding "
                "memmap produced during index construction. Missing file: "
                f"{memmap_path}"
            )

        document_count = int(self.embedding_service.document_count)
        embedding_dimension = int(self.embedding_service.embedding_dimension)

        self.embedding_matrix = np.memmap(
            memmap_path,
            dtype="float32",
            mode="r",
            shape=(document_count, embedding_dimension),
        )

        self.doc_id_to_index = {
            str(doc_id): index
            for index, doc_id in enumerate(self.embedding_service.doc_ids)
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        query = query.strip()

        if not query:
            return []

        candidate_results = self.bm25_service.search(
            query=query,
            top_k=self.candidate_count,
        )

        if not candidate_results:
            return []

        if self.embedding_matrix is None:
            raise RuntimeError("Document embedding matrix is not loaded.")

        candidate_vectors = []
        aligned_candidates = []

        for result in candidate_results:
            doc_id = str(result.get("doc_id", ""))
            vector_index = self.doc_id_to_index.get(doc_id)

            if vector_index is None:
                continue

            candidate_vectors.append(self.embedding_matrix[vector_index])
            aligned_candidates.append(result)

        if not candidate_vectors:
            return []

        candidate_matrix = np.ascontiguousarray(
            np.vstack(candidate_vectors),
            dtype="float32",
        )

        query_embedding = self.embedding_service._encode_queries(
            [query],
            batch_size=1,
        )[0]

        scores = candidate_matrix @ query_embedding
        order = np.argsort(-scores)

        rerank_top_k = min(int(top_k), len(order))
        results: List[Dict] = []

        for new_rank, candidate_position in enumerate(
            order[:rerank_top_k],
            start=1,
        ):
            original_result = aligned_candidates[int(candidate_position)]
            score = float(scores[int(candidate_position)])

            results.append(
                {
                    "rank": new_rank,
                    "doc_id": original_result.get("doc_id"),
                    "title": original_result.get("title"),
                    "snippet": original_result.get("snippet"),
                    "score": round(score, 6),
                    "bm25_rank": original_result.get("rank"),
                    "bm25_score": original_result.get("score"),
                    "hybrid_method": "bm25_candidates_embedding_rerank",
                }
            )

        return results
