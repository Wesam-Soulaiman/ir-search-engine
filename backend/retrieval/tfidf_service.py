import json
import os
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.preprocessing_service import TextPreprocessor


class TfidfRetrievalService:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.preprocessor = TextPreprocessor()

        self.documents = []
        self.processed_texts = []
        self.vectorizer = None
        self.tfidf_matrix = None

        self.load_documents()
        self.build_index()

    def load_documents(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

        self.processed_texts = [
            self.preprocessor.preprocess(
                f"{doc.get('title', '')} {doc.get('text', '')}"
            )
            for doc in self.documents
        ]

    def build_index(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_texts)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        processed_query = self.preprocessor.preprocess(query)
        query_vector = self.vectorizer.transform([processed_query])

        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for rank, index in enumerate(ranked_indices, start=1):
            doc = self.documents[index]
            results.append({
                "rank": rank,
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "snippet": doc.get("text")[:250],
                "score": round(float(scores[index]), 4),
            })

        return results