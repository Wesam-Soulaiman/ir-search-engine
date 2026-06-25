from evaluation.metrics import (
    average_precision,
    dcg_at_k,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def run_manual_tests():
    retrieved = ["D1", "D2", "D3", "D4", "D5"]
    relevant = {"D1", "D3", "D6"}

    print("Precision@3:", precision_at_k(retrieved, relevant, k=3))
    print("Recall@3:", recall_at_k(retrieved, relevant, k=3))
    print("Average Precision:", average_precision(retrieved, relevant))

    retrieved_results = {
        "Q1": ["D1", "D2", "D3"],
        "Q2": ["D4", "D5", "D6"],
    }

    qrels = {
        "Q1": {"D1", "D3"},
        "Q2": {"D6"},
    }

    print("MAP:", mean_average_precision(retrieved_results, qrels))

    relevance_scores = {
        "D1": 3,
        "D3": 2,
        "D6": 1,
    }

    print("DCG@5:", dcg_at_k(retrieved, relevance_scores, k=5))
    print("nDCG@5:", ndcg_at_k(retrieved, relevance_scores, k=5))


if __name__ == "__main__":
    run_manual_tests()