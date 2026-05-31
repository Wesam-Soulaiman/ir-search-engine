import json
import os
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingRetrievalService:
    def __init__(
        self,
        dataset_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.dataset_path = dataset_path
        self.model_name = model_name

        self.documents = []
        self.texts = []
        self.model = None
        self.index = None
        self.embeddings = None

        self.load_documents()
        self.load_model()
        self.build_index()

    def load_documents(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

        self.texts = [
            f"{doc.get('title', '')}. {doc.get('text', '')}"
            for doc in self.documents
        ]

    def load_model(self):
        self.model = SentenceTransformer(self.model_name)

    def build_index(self):
        self.embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, index in enumerate(indices[0], start=1):
            if index == -1:
                continue

            doc = self.documents[int(index)]
            score = float(scores[0][rank - 1])

            results.append({
                "rank": rank,
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "snippet": doc.get("text", "")[:250],
                "score": round(score, 4),
            })

        return results