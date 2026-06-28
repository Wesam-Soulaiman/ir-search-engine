from unittest.mock import patch

from django.test import SimpleTestCase

from evaluation.evaluator import (
    EvaluationRunner,
)
from evaluation.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class EvaluationMetricsTests(SimpleTestCase):
    def test_precision_at_k(self):
        score = precision_at_k(
            retrieved_doc_ids=[
                "d1",
                "d3",
            ],
            relevant_doc_ids={
                "d1",
                "d2",
            },
            k=2,
        )

        self.assertAlmostEqual(
            score,
            0.5,
        )

    def test_recall_at_k(self):
        score = recall_at_k(
            retrieved_doc_ids=[
                "d1",
                "d3",
                "d2",
            ],
            relevant_doc_ids={
                "d1",
                "d2",
            },
            k=3,
        )

        self.assertAlmostEqual(
            score,
            1.0,
        )

    def test_average_precision(self):
        score = average_precision(
            retrieved_doc_ids=[
                "d1",
                "d3",
                "d2",
            ],
            relevant_doc_ids={
                "d1",
                "d2",
            },
        )

        expected = (
            1.0
            + (2.0 / 3.0)
        ) / 2.0

        self.assertAlmostEqual(
            score,
            expected,
        )

    def test_ndcg_is_one_for_ideal_ranking(self):
        score = ndcg_at_k(
            retrieved_doc_ids=[
                "d1",
                "d2",
                "d3",
            ],
            relevance_scores={
                "d1": 2,
                "d2": 1,
                "d3": 0,
            },
            k=3,
        )

        self.assertAlmostEqual(
            score,
            1.0,
        )

    def test_metrics_handle_no_relevant_documents(self):
        self.assertEqual(
            average_precision(
                retrieved_doc_ids=["d1"],
                relevant_doc_ids=set(),
            ),
            0.0,
        )

        self.assertEqual(
            recall_at_k(
                retrieved_doc_ids=["d1"],
                relevant_doc_ids=set(),
                k=1,
            ),
            0.0,
        )


class FakeRetrievalService:
    def __init__(self):
        self.results_by_query = {
            "alpha query": [
                {
                    "doc_id": "d1",
                    "score": 1.0,
                },
                {
                    "doc_id": "d2",
                    "score": 0.5,
                },
            ],
            "beta query": [
                {
                    "doc_id": "d3",
                    "score": 1.0,
                },
                {
                    "doc_id": "d4",
                    "score": 0.5,
                },
            ],
        }

    def search(
        self,
        query: str,
        top_k: int,
    ):
        return self.results_by_query[
            query
        ][:top_k]


class EvaluationRunnerTests(SimpleTestCase):
    def test_evaluates_every_query_in_qrels(self):
        queries = [
            {
                "query_id": "q1",
                "query": "alpha query",
            },
            {
                "query_id": "q2",
                "query": "beta query",
            },
            {
                "query_id": "q3",
                "query": "unused query",
            },
        ]

        qrels = {
            "q1": {
                "d1": 2,
                "d2": 0,
            },
            "q2": {
                "d3": 1,
            },
        }

        fake_service = (
            FakeRetrievalService()
        )

        with (
            patch(
                "evaluation.evaluator."
                "DatasetLoader.load_queries",
                return_value=queries,
            ),
            patch(
                "evaluation.evaluator."
                "DatasetLoader.load_qrels",
                return_value=qrels,
            ),
            patch.object(
                EvaluationRunner,
                "_build_retrieval_service",
                return_value=fake_service,
            ),
            patch.object(
                EvaluationRunner,
                "_build_query_refinement_service",
                return_value=None,
            ),
        ):
            runner = EvaluationRunner(
                dataset_key="sample_dataset",
                model_name="tfidf",
                retrieval_depth=2,
                precision_k=1,
                recall_k=2,
                ndcg_k=1,
                candidate_count=2,
            )

            result = runner.evaluate()

        self.assertEqual(
            result["loaded_queries"],
            3,
        )

        self.assertEqual(
            result["qrels_queries"],
            2,
        )

        self.assertEqual(
            result["evaluated_queries"],
            2,
        )

        self.assertEqual(
            result[
                "queries_with_positive_qrels"
            ],
            2,
        )

        self.assertAlmostEqual(
            result["MAP@2"],
            1.0,
        )

        self.assertAlmostEqual(
            result["Precision@1"],
            1.0,
        )

        self.assertAlmostEqual(
            result["Recall@2"],
            1.0,
        )

        self.assertAlmostEqual(
            result["nDCG@1"],
            1.0,
        )

    def test_can_evaluate_requested_held_out_query_ids_only(self):
        queries = [
            {
                "query_id": "q1",
                "query": "alpha query",
            },
            {
                "query_id": "q2",
                "query": "beta query",
            },
            {
                "query_id": "q3",
                "query": "unused query",
            },
        ]
        qrels = {
            "q1": {
                "d1": 1,
            },
            "q2": {
                "d3": 1,
            },
            "q3": {
                "d9": 1,
            },
        }
        fake_service = FakeRetrievalService()

        with (
            patch(
                "evaluation.evaluator."
                "DatasetLoader.load_queries",
                return_value=queries,
            ),
            patch(
                "evaluation.evaluator."
                "DatasetLoader.load_qrels",
                return_value=qrels,
            ),
            patch.object(
                EvaluationRunner,
                "_build_retrieval_service",
                return_value=fake_service,
            ),
            patch.object(
                EvaluationRunner,
                "_build_query_refinement_service",
                return_value=None,
            ),
        ):
            runner = EvaluationRunner(
                dataset_key="sample_dataset",
                model_name="tfidf",
                retrieval_depth=2,
                precision_k=1,
                recall_k=2,
                ndcg_k=1,
                candidate_count=2,
                evaluation_query_ids=["q2"],
            )
            result = runner.evaluate()

        self.assertEqual(
            runner.evaluation_query_ids,
            ["q2"],
        )
        self.assertEqual(
            result["qrels_queries"],
            1,
        )
        self.assertEqual(
            result["evaluated_queries"],
            1,
        )
        self.assertAlmostEqual(
            result["MAP@2"],
            1.0,
        )

    def test_missing_query_text_causes_failure(self):
        queries = [
            {
                "query_id": "q1",
                "query": "alpha query",
            },
        ]

        qrels = {
            "q1": {
                "d1": 1,
            },
            "q2": {
                "d2": 1,
            },
        }

        with (
            patch(
                "evaluation.evaluator."
                "DatasetLoader.load_queries",
                return_value=queries,
            ),
            patch(
                "evaluation.evaluator."
                "DatasetLoader.load_qrels",
                return_value=qrels,
            ),
        ):
            with self.assertRaises(
                ValueError
            ) as context:
                EvaluationRunner(
                    dataset_key=(
                        "sample_dataset"
                    ),
                    model_name="tfidf",
                    retrieval_depth=10,
                    precision_k=10,
                    recall_k=10,
                    ndcg_k=10,
                    candidate_count=10,
                )

        self.assertIn(
            "no matching query text",
            str(context.exception),
        )

    def test_duplicate_query_id_causes_failure(self):
        queries = [
            {
                "query_id": "q1",
                "query": "first",
            },
            {
                "query_id": "q1",
                "query": "second",
            },
        ]

        qrels = {
            "q1": {
                "d1": 1,
            },
        }

        with (
            patch(
                "evaluation.evaluator."
                "DatasetLoader.load_queries",
                return_value=queries,
            ),
            patch(
                "evaluation.evaluator."
                "DatasetLoader.load_qrels",
                return_value=qrels,
            ),
        ):
            with self.assertRaises(
                ValueError
            ) as context:
                EvaluationRunner(
                    dataset_key=(
                        "sample_dataset"
                    ),
                    model_name="tfidf",
                    retrieval_depth=10,
                    precision_k=10,
                    recall_k=10,
                    ndcg_k=10,
                    candidate_count=10,
                )

        self.assertIn(
            "Duplicate query ID",
            str(context.exception),
        )

    def test_metric_cutoff_cannot_exceed_depth(self):
        with self.assertRaises(ValueError):
            EvaluationRunner(
                dataset_key="sample_dataset",
                model_name="tfidf",
                retrieval_depth=10,
                precision_k=20,
                recall_k=10,
                ndcg_k=10,
                candidate_count=20,
            )

    def test_hybrid_candidates_cannot_be_less_than_depth(
        self,
    ):
        with self.assertRaises(ValueError):
            EvaluationRunner(
                dataset_key="sample_dataset",
                model_name=(
                    "hybrid_parallel"
                ),
                retrieval_depth=100,
                precision_k=10,
                recall_k=100,
                ndcg_k=10,
                candidate_count=50,
            )
