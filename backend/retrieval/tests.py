from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
from rest_framework.test import APITestCase

from query_refinement.spelling_correction_service import (
    SpellingCorrectionService,
)
from retrieval.bm25_service import (
    BM25RetrievalService,
)
from retrieval.biomedical_embedding_service import (
    BiomedicalEmbeddingService,
)
from retrieval.embedding_service import (
    EmbeddingIndexError,
)
from retrieval.personalization_service import (
    PersonalizedQueryService,
)
from retrieval.request_validation import (
    parse_boolean,
    parse_float,
    parse_integer,
)
from retrieval.search_history_store import (
    SearchHistoryStore,
)
from retrieval.hybrid_parallel_service import (
    HybridParallelRetrievalService,
)
from retrieval.tfidf_service import (
    TfidfRetrievalService,
)

import tempfile
from pathlib import Path
from unittest.mock import patch

from document_store.repository import (
    DocumentStoreError,
    DocumentStoreRepository,
)
from retrieval.result_enrichment import (
    enrich_search_results,
)


class RequestValidationTests(SimpleTestCase):
    def test_parse_boolean_false_string(self):
        self.assertFalse(
            parse_boolean(
                "false",
                "field",
            )
        )

    def test_parse_boolean_true_string(self):
        self.assertTrue(
            parse_boolean(
                "true",
                "field",
            )
        )

    def test_parse_boolean_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            parse_boolean(
                "maybe",
                "field",
            )

    def test_parse_integer_applies_default(self):
        value = parse_integer(
            value=None,
            field_name="top_k",
            default=10,
            minimum=1,
        )

        self.assertEqual(value, 10)

    def test_parse_integer_rejects_non_integer(self):
        with self.assertRaises(ValueError):
            parse_integer(
                value="abc",
                field_name="top_k",
            )

    def test_parse_integer_checks_minimum(self):
        with self.assertRaises(ValueError):
            parse_integer(
                value=0,
                field_name="top_k",
                minimum=1,
            )

    def test_parse_float_checks_maximum(self):
        with self.assertRaises(ValueError):
            parse_float(
                value=1.5,
                field_name="bm25_b",
                maximum=1.0,
            )


class SearchHistoryStoreTests(SimpleTestCase):
    def test_history_persists_to_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "search_history.sqlite3"
            )

            first_store = SearchHistoryStore(
                database_path
            )

            first_store.record_query(
                user_id="user-1",
                dataset="quora",
                query="neural ranking models",
            )

            second_store = SearchHistoryStore(
                database_path
            )

            recent_queries = (
                second_store.get_recent_queries(
                    user_id="user-1",
                    dataset="quora",
                    limit=5,
                )
            )

            self.assertEqual(
                recent_queries,
                [
                    "neural ranking models",
                ],
            )


class BiomedicalModelDownloadTests(SimpleTestCase):
    def test_download_script_excludes_bin_files(self):
        from scripts import (
            download_biomedical_embedding_model
            as downloader,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)

            with patch.object(
                downloader,
                "snapshot_download",
                return_value=str(output_dir),
            ) as mocked_snapshot_download:
                downloader.download_model_snapshot(
                    model_name=(
                        "NeuML/"
                        "pubmedbert-base-embeddings"
                    ),
                    output_dir=output_dir,
                )

        mocked_snapshot_download.assert_called_once()
        self.assertEqual(
            mocked_snapshot_download.call_args.kwargs[
                "ignore_patterns"
            ],
            [
                "*.bin",
            ],
        )

    def test_model_validation_requires_safetensors(self):
        from scripts import (
            download_biomedical_embedding_model
            as downloader,
        )

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (
                model_dir
                / "config.json"
            ).write_text(
                "{}",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "model.safetensors",
            ):
                downloader.validate_local_model_directory(
                    model_dir
                )


class PersonalizedQueryServiceTests(SimpleTestCase):
    def test_personalization_adds_history_terms_only(self):
        service = PersonalizedQueryService(
            dataset_key="sample_dataset",
            max_personalization_terms=2,
        )

        result = service.personalize(
            query="machine learning",
            previous_queries=[
                "neural network learning",
                "neural search systems",
            ],
        )

        self.assertEqual(
            result["personalized_query"],
            "machine learning neural network",
        )

        self.assertEqual(
            result["personalization_terms"],
            [
                "neural",
                "network",
            ],
        )


class SpellingCorrectionServiceTests(SimpleTestCase):
    def test_corrects_misspelled_query_terms(self):
        service = SpellingCorrectionService(
            dataset_key="sample_dataset",
            vocabulary=[
                "diabetes",
                "insulin",
                "treatment",
            ],
        )

        result = service.correct(
            "diabtes inslin tretment"
        )

        self.assertEqual(
            result["corrected_query"],
            "diabetes insulin treatment",
        )

        self.assertTrue(
            result["spelling_correction_used"]
        )

        self.assertEqual(
            result["spelling_corrections"],
            [
                {
                    "original": "diabtes",
                    "corrected": "diabetes",
                },
                {
                    "original": "inslin",
                    "corrected": "insulin",
                },
                {
                    "original": "tretment",
                    "corrected": "treatment",
                },
            ],
        )

    def test_preserves_biomedical_tokens(self):
        service = SpellingCorrectionService(
            dataset_key="clinical_trials",
            vocabulary=[
                "diabetes",
                "insulin",
                "treatment",
            ],
        )

        result = service.correct(
            "BRAF HER2 V600E diabtes"
        )

        self.assertEqual(
            result["corrected_query"],
            "BRAF HER2 V600E diabetes",
        )

        self.assertEqual(
            result["spelling_corrections"],
            [
                {
                    "original": "diabtes",
                    "corrected": "diabetes",
                },
            ],
        )

    def test_uses_natural_surface_words_not_stems(self):
        service = SpellingCorrectionService(
            dataset_key="sample_dataset",
            vocabulary=[
                "english",
                "language",
                "learn",
            ],
        )

        result = service.correct(
            "how to lern english langauge"
        )

        self.assertEqual(
            result["corrected_query"],
            "how to learn english language",
        )

        self.assertNotIn(
            "langaug",
            result["corrected_query"],
        )

    def test_corrects_common_quora_like_typos(self):
        service = SpellingCorrectionService(
            dataset_key="quora",
            vocabulary=[
                "what",
                "best",
                "way",
                "learn",
                "programming",
            ],
        )

        result = service.correct(
            "whatt is the best waay to lern programing"
        )

        self.assertEqual(
            result["corrected_query"],
            "what is the best way to learn programming",
        )


class LexicalRetrievalIntegrationTests(
    SimpleTestCase
):
    """
    Small integration tests using sample_dataset.

    use_saved_index=False guarantees that these tests do not depend on
    local index files that are excluded from Git.
    """

    def test_bm25_searches_sample_dataset(self):
        service = BM25RetrievalService(
            dataset_key="sample_dataset",
            use_saved_index=False,
        )

        results = service.search(
            query="machine learning",
            top_k=3,
        )

        self.assertEqual(
            len(results),
            3,
        )

        self.assertEqual(
            results[0]["doc_id"],
            "DOC001",
        )

        self.assertIsNotNone(
            service.bm25
        )

    def test_tfidf_searches_sample_dataset(self):
        service = TfidfRetrievalService(
            dataset_key="sample_dataset",
            use_saved_index=False,
        )

        results = service.search(
            query="machine learning",
            top_k=3,
        )

        self.assertEqual(
            len(results),
            3,
        )

        self.assertEqual(
            results[0]["doc_id"],
            "DOC001",
        )

    def test_bm25_rejects_invalid_b(self):
        with self.assertRaises(ValueError):
            BM25RetrievalService(
                dataset_key="sample_dataset",
                b=1.5,
                use_saved_index=False,
            )

    def test_bm25_rejects_invalid_k1(self):
        with self.assertRaises(ValueError):
            BM25RetrievalService(
                dataset_key="sample_dataset",
                k1=0,
                use_saved_index=False,
            )

    def test_bm25_rejects_invalid_top_k(self):
        service = BM25RetrievalService(
            dataset_key="sample_dataset",
            use_saved_index=False,
        )

        with self.assertRaises(ValueError):
            service.search(
                query="machine learning",
                top_k=0,
            )

    def test_tfidf_returns_empty_for_blank_query(self):
        service = TfidfRetrievalService(
            dataset_key="sample_dataset",
            use_saved_index=False,
        )

        self.assertEqual(
            service.search(
                query="   ",
                top_k=3,
            ),
            [],
        )


class HybridParallelBiomedicalTests(SimpleTestCase):
    @patch(
        "retrieval.hybrid_parallel_service."
        "BiomedicalEmbeddingService"
    )
    @patch(
        "retrieval.hybrid_parallel_service."
        "EmbeddingRetrievalService"
    )
    @patch(
        "retrieval.hybrid_parallel_service."
        "BM25RetrievalService"
    )
    @patch(
        "retrieval.hybrid_parallel_service."
        "TfidfRetrievalService"
    )
    def test_default_biomedical_weight_is_zero(
        self,
        mocked_tfidf_service,
        mocked_bm25_service,
        mocked_embedding_service,
        mocked_biomedical_service,
    ):
        service = HybridParallelRetrievalService(
            dataset_key="clinical_trials",
        )

        self.assertEqual(
            service.model_weights["biomedical"],
            0.0,
        )

        self.assertIsNone(
            service.biomedical_service
        )

        mocked_tfidf_service.assert_called_once()
        mocked_bm25_service.assert_called_once()
        mocked_embedding_service.assert_called_once()
        mocked_biomedical_service.assert_not_called()

    @patch(
        "retrieval.hybrid_parallel_service."
        "BiomedicalEmbeddingService"
    )
    @patch(
        "retrieval.hybrid_parallel_service."
        "EmbeddingRetrievalService"
    )
    @patch(
        "retrieval.hybrid_parallel_service."
        "BM25RetrievalService"
    )
    @patch(
        "retrieval.hybrid_parallel_service."
        "TfidfRetrievalService"
    )
    def test_biomedical_service_participates_when_weight_is_positive(
        self,
        mocked_tfidf_service,
        mocked_bm25_service,
        mocked_embedding_service,
        mocked_biomedical_service,
    ):
        service = HybridParallelRetrievalService(
            dataset_key="clinical_trials",
            biomedical_weight=1.5,
        )

        self.assertEqual(
            service.biomedical_service,
            mocked_biomedical_service.return_value,
        )

        mocked_tfidf_service.assert_called_once()
        mocked_bm25_service.assert_called_once()
        mocked_embedding_service.assert_called_once()
        mocked_biomedical_service.assert_called_once()

        active_model_names = [
            model_name
            for model_name, service_instance
            in service._active_services()
        ]

        self.assertIn(
            "biomedical",
            active_model_names,
        )

    def test_biomedical_fusion_weight_is_hidden_until_enabled(
        self,
    ):
        service = HybridParallelRetrievalService.__new__(
            HybridParallelRetrievalService
        )
        service.rrf_k = 60
        service.model_weights = {
            "tfidf": 1.0,
            "bm25": 1.0,
            "embedding": 1.0,
            "biomedical": 0.0,
        }

        results = service._weighted_reciprocal_rank_fusion(
            {
                "tfidf": [
                    {
                        "rank": 1,
                        "doc_id": "DOC001",
                        "title": "Title",
                        "snippet": "Snippet",
                        "score": 0.9,
                    }
                ],
                "biomedical": [
                    {
                        "rank": 1,
                        "doc_id": "DOC002",
                        "title": "Bio title",
                        "snippet": "Bio snippet",
                        "score": 0.8,
                    }
                ],
            }
        )

        self.assertNotIn(
            "biomedical",
            results[0]["fusion_weights"],
        )

        service.model_weights["biomedical"] = 2.0

        results = service._weighted_reciprocal_rank_fusion(
            {
                "biomedical": [
                    {
                        "rank": 1,
                        "doc_id": "DOC002",
                        "title": "Bio title",
                        "snippet": "Bio snippet",
                        "score": 0.8,
                    }
                ]
            }
        )

        self.assertEqual(
            results[0]["fusion_weights"]["biomedical"],
            2.0,
        )

        self.assertIn(
            "biomedical",
            results[0]["model_details"],
        )


class BiomedicalEmbeddingServiceTests(SimpleTestCase):
    def test_manifest_model_name_must_match_configuration(self):
        service = BiomedicalEmbeddingService.__new__(
            BiomedicalEmbeddingService
        )
        service.dataset_key = "clinical_trials"
        service.manifest = {
            "index_type": "sentence_transformer_faiss",
            "faiss_index_type": "flat",
            "dataset": "clinical_trials",
            "document_count": 1,
            "embedding_dimension": 768,
            "model_path": "artifacts/models/biomedical-embedding",
            "model_name": "wrong/model",
            "vectors_are_normalized": True,
            "similarity": "cosine_via_inner_product",
            "faiss_ntotal": 1,
        }

        with self.assertRaises(EmbeddingIndexError):
            service._validate_manifest()


class SearchAPITests(APITestCase):
    search_url = "/api/search/"
    datasets_url = "/api/datasets/"

    @patch(
        "retrieval.views.run_search"
    )
    def test_valid_search_request(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = [
            {
                "rank": 1,
                "doc_id": "DOC001",
                "title": "Test title",
                "snippet": "Test text",
                "score": 1.0,
            }
        ]

        response = self.client.post(
            self.search_url,
            {
                "query": "machine learning",
                "dataset": "sample_dataset",
                "model": "bm25",
                "top_k": 5,
                "bm25_k1": 1.5,
                "bm25_b": 0.75,
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.data["result_count"],
            1,
        )

        self.assertFalse(
            response.data[
                "use_query_refinement"
            ]
        )

        mocked_run_search.assert_called_once()

    @patch(
        "retrieval.views.run_search"
    )
    def test_false_string_remains_false(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        response = self.client.post(
            self.search_url,
            {
                "query": "What causes a nightmare?",
                "dataset": "quora",
                "model": "bm25",
                "use_query_refinement": "false",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertFalse(
            response.data[
                "use_query_refinement"
            ]
        )

        self.assertEqual(
            response.data["original_query"],
            response.data["refined_query"],
        )

    @patch(
        "retrieval.views.run_search"
    )
    @patch(
        "retrieval.views."
        "get_query_refinement_service"
    )
    def test_query_refinement_can_be_enabled(
        self,
        mocked_get_refinement_service,
        mocked_run_search,
    ):
        refinement_service = MagicMock()
        refinement_service.refine.return_value = (
            "machine learning neural networks"
        )

        mocked_get_refinement_service.return_value = (
            refinement_service
        )

        mocked_run_search.return_value = []

        response = self.client.post(
            self.search_url,
            {
                "query": "machine learning",
                "dataset": "sample_dataset",
                "model": "bm25",
                "use_query_refinement": True,
                "feedback_docs": 3,
                "expansion_terms": 5,
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertTrue(
            response.data[
                "use_query_refinement"
            ]
        )

        self.assertEqual(
            response.data["refined_query"],
            "machine learning neural networks",
        )

        refinement_service.refine.assert_called_once_with(
            "machine learning"
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_spelling_correction_disabled_by_default(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        response = self.client.post(
            self.search_url,
            {
                "query": "machne learnng",
                "dataset": "sample_dataset",
                "model": "bm25",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertFalse(
            response.data["use_spelling_correction"]
        )

        self.assertFalse(
            response.data["spelling_correction_used"]
        )

        self.assertEqual(
            response.data["corrected_query"],
            "machne learnng",
        )

        self.assertEqual(
            response.data["spelling_corrections"],
            [],
        )

        self.assertEqual(
            mocked_run_search.call_args.kwargs["query"],
            "machne learnng",
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_spelling_correction_can_be_enabled(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        response = self.client.post(
            self.search_url,
            {
                "query": "machne learnng",
                "dataset": "sample_dataset",
                "model": "bm25",
                "use_spelling_correction": True,
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertTrue(
            response.data["use_spelling_correction"]
        )

        self.assertTrue(
            response.data["spelling_correction_used"]
        )

        self.assertEqual(
            response.data["corrected_query"],
            "machine learning",
        )

        self.assertEqual(
            response.data["spelling_corrections"],
            [
                {
                    "original": "machne",
                    "corrected": "machine",
                },
                {
                    "original": "learnng",
                    "corrected": "learning",
                },
            ],
        )

        self.assertEqual(
            response.data["query"],
            "machine learning",
        )

        self.assertEqual(
            mocked_run_search.call_args.kwargs["query"],
            "machine learning",
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_search_api_works_without_spelling_flag(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        response = self.client.post(
            self.search_url,
            {
                "query": "machine learning",
                "dataset": "sample_dataset",
                "model": "bm25",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertFalse(
            response.data["use_spelling_correction"]
        )

        self.assertEqual(
            response.data["query"],
            "machine learning",
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_personalization_disabled_by_default(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        with tempfile.TemporaryDirectory() as directory:
            history_store = SearchHistoryStore(
                Path(directory)
                / "search_history.sqlite3"
            )

            history_store.record_query(
                user_id="session-1",
                dataset="sample_dataset",
                query="neural networks",
            )

            with patch(
                "retrieval.views."
                "get_search_history_store",
                return_value=history_store,
            ):
                response = self.client.post(
                    self.search_url,
                    {
                        "query": "machine learning",
                        "dataset": "sample_dataset",
                        "model": "bm25",
                        "user_id": "session-1",
                    },
                    format="json",
                )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertFalse(
            response.data["use_personalization"]
        )

        self.assertFalse(
            response.data["personalization_used"]
        )

        self.assertEqual(
            response.data["query"],
            "machine learning",
        )

        self.assertEqual(
            response.data["personalization_terms"],
            [],
        )

        self.assertEqual(
            mocked_run_search.call_args.kwargs["query"],
            "machine learning",
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_personalization_expands_query_when_enabled(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        with tempfile.TemporaryDirectory() as directory:
            history_store = SearchHistoryStore(
                Path(directory)
                / "search_history.sqlite3"
            )

            history_store.record_query(
                user_id="session-2",
                dataset="sample_dataset",
                query="neural network learning",
            )

            with patch(
                "retrieval.views."
                "get_search_history_store",
                return_value=history_store,
            ):
                response = self.client.post(
                    self.search_url,
                    {
                        "query": "machine learning",
                        "dataset": "sample_dataset",
                        "model": "bm25",
                        "user_id": "session-2",
                        "use_personalization": True,
                        "max_personalization_terms": 2,
                    },
                    format="json",
                )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertTrue(
            response.data["use_personalization"]
        )

        self.assertTrue(
            response.data["personalization_used"]
        )

        self.assertEqual(
            response.data["personalization_terms"],
            [
                "neural",
                "network",
            ],
        )

        self.assertEqual(
            response.data["personalized_query"],
            "machine learning neural network",
        )

        self.assertEqual(
            mocked_run_search.call_args.kwargs["query"],
            "machine learning neural network",
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_personalization_runs_after_spelling_correction(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        with tempfile.TemporaryDirectory() as directory:
            history_store = SearchHistoryStore(
                Path(directory)
                / "search_history.sqlite3"
            )

            history_store.record_query(
                user_id="session-4",
                dataset="sample_dataset",
                query="neural network learning",
            )

            with patch(
                "retrieval.views."
                "get_search_history_store",
                return_value=history_store,
            ):
                response = self.client.post(
                    self.search_url,
                    {
                        "query": "machne learning",
                        "dataset": "sample_dataset",
                        "model": "bm25",
                        "user_id": "session-4",
                        "use_spelling_correction": True,
                        "use_personalization": True,
                        "max_personalization_terms": 2,
                    },
                    format="json",
                )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.data["corrected_query"],
            "machine learning",
        )

        self.assertEqual(
            response.data["personalized_query"],
            "machine learning neural network",
        )

        self.assertEqual(
            mocked_run_search.call_args.kwargs["query"],
            "machine learning neural network",
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_search_api_works_without_user_id(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        response = self.client.post(
            self.search_url,
            {
                "query": "machine learning",
                "dataset": "sample_dataset",
                "model": "bm25",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertFalse(
            response.data["use_personalization"]
        )

        self.assertEqual(
            response.data["results"],
            [],
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_search_api_records_history_for_user_id(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        with tempfile.TemporaryDirectory() as directory:
            history_store = SearchHistoryStore(
                Path(directory)
                / "search_history.sqlite3"
            )

            with patch(
                "retrieval.views."
                "get_search_history_store",
                return_value=history_store,
            ):
                response = self.client.post(
                    self.search_url,
                    {
                        "query": "database indexing",
                        "dataset": "sample_dataset",
                        "model": "bm25",
                        "user_id": "session-3",
                    },
                    format="json",
                )

                recent_queries = (
                    history_store.get_recent_queries(
                        user_id="session-3",
                        dataset="sample_dataset",
                        limit=5,
                    )
                )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            recent_queries,
            [
                "database indexing",
            ],
        )

    def test_unknown_model_returns_400(self):
        response = self.client.post(
            self.search_url,
            {
                "query": "test",
                "dataset": "quora",
                "model": "unknown_model",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertIn(
            "Unknown model",
            response.data["error"],
        )

        self.assertEqual(
            response.data["results"],
            [],
        )

    def test_unknown_dataset_returns_400(self):
        response = self.client.post(
            self.search_url,
            {
                "query": "test",
                "dataset": "missing_dataset",
                "model": "bm25",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertIn(
            "Unknown dataset",
            response.data["error"],
        )

    def test_biomedical_embedding_rejects_unsupported_dataset(
        self,
    ):
        response = self.client.post(
            self.search_url,
            {
                "query": "diabetes insulin",
                "dataset": "quora",
                "model": "biomedical_embedding",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertIn(
            "clinical_trials",
            response.data["error"],
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_biomedical_embedding_search_can_be_requested_for_clinical_trials(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        response = self.client.post(
            self.search_url,
            {
                "query": "diabetes insulin treatment",
                "dataset": "clinical_trials",
                "model": "biomedical_embedding",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.data["model"],
            "biomedical_embedding",
        )

        self.assertEqual(
            mocked_run_search.call_args.kwargs["model"],
            "biomedical_embedding",
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_hybrid_parallel_accepts_optional_biomedical_weight(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = []

        response = self.client.post(
            self.search_url,
            {
                "query": "diabetes insulin treatment",
                "dataset": "clinical_trials",
                "model": "hybrid_parallel",
                "top_k": 5,
                "candidate_count": 10,
                "biomedical_weight": 2.5,
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.data["biomedical_weight"],
            2.5,
        )

        self.assertEqual(
            mocked_run_search.call_args.kwargs["biomedical_weight"],
            2.5,
        )

    def test_invalid_bm25_b_returns_400(self):
        response = self.client.post(
            self.search_url,
            {
                "query": "test",
                "dataset": "quora",
                "model": "bm25",
                "bm25_b": 1.5,
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertIn(
            "bm25_b",
            response.data["error"],
        )

    def test_hybrid_candidate_count_must_cover_top_k(
        self,
    ):
        response = self.client.post(
            self.search_url,
            {
                "query": "test",
                "dataset": "clinical_trials",
                "model": "hybrid_parallel",
                "top_k": 20,
                "candidate_count": 10,
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertIn(
            "candidate_count",
            response.data["error"],
        )

    def test_blank_query_returns_400(self):
        response = self.client.post(
            self.search_url,
            {
                "query": "   ",
                "dataset": "quora",
                "model": "tfidf",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertIn(
            "cannot be empty",
            response.data["error"],
        )

    def test_datasets_endpoint_lists_models(self):
        response = self.client.get(
            self.datasets_url
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        model_names = set(
            response.data["models"]
        )

        self.assertEqual(
            model_names,
            {
                "agent",
                "tfidf",
                "bm25",
                "embedding",
                "biomedical_embedding",
                "hybrid_serial",
                "hybrid_parallel",
            },
        )

        dataset_keys = {
            dataset["key"]
            for dataset
            in response.data["datasets"]
        }

        self.assertIn(
            "quora",
            dataset_keys,
        )

        self.assertIn(
            "clinical_trials",
            dataset_keys,
        )
        
        
class SearchResultEnrichmentTests(
    SimpleTestCase
):
    def setUp(self):
        self.temporary_directory = (
            tempfile.TemporaryDirectory()
        )

        database_path = (
            Path(
                self.temporary_directory.name
            )
            / "corpus.sqlite3"
        )

        self.repository = (
            DocumentStoreRepository(
                database_path
            )
        )

        self.repository.initialize()

        self.repository.upsert_dataset(
            dataset_key="quora",
            display_name="Quora",
            ir_dataset_id="beir/quora/test",
        )

        self.repository.bulk_upsert_documents(
            "quora",
            [
                {
                    "doc_id": "D1",
                    "title": "Raw title one",
                    "text": (
                        "Original raw text for "
                        "document one."
                    ),
                },
                {
                    "doc_id": "D2",
                    "title": "Raw title two",
                    "text": (
                        "Original raw text for "
                        "document two."
                    ),
                },
            ],
        )

        self.repository_patch = patch(
            "retrieval.result_enrichment."
            "get_document_store_repository",
            return_value=self.repository,
        )

        self.repository_patch.start()

    def tearDown(self):
        self.repository_patch.stop()
        self.temporary_directory.cleanup()

    def test_enrichment_preserves_ranking_and_scores(
        self,
    ):
        results = enrich_search_results(
            dataset_key="quora",
            results=[
                {
                    "rank": 1,
                    "doc_id": "D2",
                    "score": 9.5,
                    "title": "Index title",
                    "snippet": "Index snippet",
                },
                {
                    "rank": 2,
                    "doc_id": "D1",
                    "score": 8.5,
                    "title": "Index title",
                    "snippet": "Index snippet",
                },
            ],
            snippet_length=100,
        )

        self.assertEqual(
            [
                result["doc_id"]
                for result in results
            ],
            [
                "D2",
                "D1",
            ],
        )

        self.assertEqual(
            results[0]["rank"],
            1,
        )

        self.assertEqual(
            results[0]["score"],
            9.5,
        )

        self.assertEqual(
            results[0]["title"],
            "Raw title two",
        )

        self.assertEqual(
            results[0]["document_source"],
            "sqlite_document_store",
        )

    def test_snippet_length_is_applied(self):
        results = enrich_search_results(
            dataset_key="quora",
            results=[
                {
                    "rank": 1,
                    "doc_id": "D1",
                    "score": 1.0,
                },
            ],
            snippet_length=8,
        )

        self.assertEqual(
            results[0]["snippet"],
            "Original",
        )

    def test_raw_text_is_optional(self):
        without_raw_text = (
            enrich_search_results(
                dataset_key="quora",
                results=[
                    {
                        "rank": 1,
                        "doc_id": "D1",
                        "score": 1.0,
                    },
                ],
                include_raw_text=False,
            )
        )

        self.assertNotIn(
            "raw_text",
            without_raw_text[0],
        )

        with_raw_text = (
            enrich_search_results(
                dataset_key="quora",
                results=[
                    {
                        "rank": 1,
                        "doc_id": "D1",
                        "score": 1.0,
                    },
                ],
                include_raw_text=True,
            )
        )

        self.assertEqual(
            with_raw_text[0]["raw_text"],
            (
                "Original raw text for "
                "document one."
            ),
        )

    def test_missing_document_raises_in_strict_mode(
        self,
    ):
        with self.assertRaises(
            DocumentStoreError
        ):
            enrich_search_results(
                dataset_key="quora",
                results=[
                    {
                        "rank": 1,
                        "doc_id": "MISSING",
                        "score": 1.0,
                    },
                ],
                strict=True,
            )

    def test_sample_dataset_uses_index_metadata(
        self,
    ):
        results = enrich_search_results(
            dataset_key="sample_dataset",
            results=[
                {
                    "rank": 1,
                    "doc_id": "DOC001",
                    "title": "Index title",
                    "snippet": "Index text",
                    "score": 1.0,
                },
            ],
        )

        self.assertEqual(
            results[0]["title"],
            "Index title",
        )

        self.assertEqual(
            results[0]["document_source"],
            "retrieval_index",
        )
