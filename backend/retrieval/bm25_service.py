import json
import os
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from preprocessing.preprocessing_service import TextPreprocessor


class BM25RetrievalService:
    def __init__(self, dataset_path: str, k1: float = 1.5, b: float = 0.75):
        self.dataset_path = dataset_path
        self.k1 = k1
        self.b = b
        self.preprocessor = TextPreprocessor()

        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

        self.load_documents()
        self.build_index()

    def load_documents(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

        self.tokenized_corpus = [
            self.preprocessor.preprocess_tokens(
                f"{doc.get('title', '')} {doc.get('text', '')}"
            )
            for doc in self.documents
        ]

    def build_index(self):
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        tokenized_query = self.preprocessor.preprocess_tokens(query)

        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, index in enumerate(ranked_indices, start=1):
            doc = self.documents[index]
            results.append({
                "rank": rank,
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "snippet": doc.get("text", "")[:250],
                "score": round(float(scores[index]), 4),
            })

        return results