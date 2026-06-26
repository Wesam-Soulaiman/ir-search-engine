import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
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
from indexing.distributed_bm25_index import (
    MERGE_METHOD,
    SHARDING_STRATEGY,
    assign_shard,
    validate_num_shards,
)
from retrieval.distributed_bm25_service import (
    DistributedBM25RetrievalService,
)
from retrieval.ltr_feature_extractor import (
    LTRFeatureExtractor,
    build_ltr_labels,
    merge_ltr_candidate_results,
)
from retrieval.ltr_service import (
    LTRModelNotTrainedError,
    LTRRetrievalService,
)
from retrieval.result_enrichment import (
    enrich_search_results,
)


class ScoreFromRrfModel:
    def predict(self, feature_matrix):
        return np.asarray(
            feature_matrix[:, -1],
            dtype=float,
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


class DistributedBM25Tests(SimpleTestCase):
    def _write_distributed_manifest(
        self,
        root: Path,
        dataset_key: str = "sample_dataset",
        num_shards: int = 2,
    ) -> Path:
        index_dir = (
            root
            / dataset_key
            / "distributed_bm25"
        )
        shards_dir = index_dir / "shards"
        shard_counts = {
            f"shard_{shard_id}": 1
            for shard_id in range(num_shards)
        }

        index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            index_dir
            / "manifest.json"
        ).write_text(
            json.dumps({
                "dataset": dataset_key,
                "model": "distributed_bm25",
                "version": 1,
                "num_shards": num_shards,
                "total_documents": num_shards,
                "sharding_strategy": (
                    SHARDING_STRATEGY
                ),
                "merge_method": MERGE_METHOD,
                "rrf_k": 60,
                "shard_document_counts": (
                    shard_counts
                ),
            }),
            encoding="utf-8",
        )

        for shard_id in range(num_shards):
            shard_dir = (
                shards_dir
                / f"shard_{shard_id}"
            )
            shard_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                shard_dir
                / "manifest.json"
            ).write_text(
                json.dumps({
                    "dataset": dataset_key,
                    "index_type": (
                        "sqlite_inverted_bm25"
                    ),
                    "version": 1,
                    "document_count": 1,
                    "vocabulary_size": 1,
                    "posting_count": 1,
                    "average_document_length": 1.0,
                    "average_idf": 1.0,
                    "epsilon": 0.25,
                    "preprocessing": {},
                }),
                encoding="utf-8",
            )

            (
                shard_dir
                / "shard_manifest.json"
            ).write_text(
                json.dumps({
                    "dataset": dataset_key,
                    "model": "distributed_bm25",
                    "shard_id": shard_id,
                    "num_shards": num_shards,
                    "document_count": 1,
                    "sharding_strategy": (
                        SHARDING_STRATEGY
                    ),
                }),
                encoding="utf-8",
            )

        return index_dir

    def _mocked_service(
        self,
        shard_results,
    ):
        service = DistributedBM25RetrievalService.__new__(
            DistributedBM25RetrievalService
        )
        service.dataset_key = "sample_dataset"
        service.num_shards = len(
            shard_results
        )
        service.shard_document_counts = {
            f"shard_{shard_id}": 1
            for shard_id in range(
                service.num_shards
            )
        }
        service.shard_services = {}
        service.repository = None
        service.manifest = {}
        service.default_shard_top_k = None
        service.default_rrf_k = 60
        service.last_search_metadata = {}

        for shard_id, results in shard_results.items():
            shard_service = MagicMock()
            shard_service.search.return_value = (
                results
            )
            service.shard_services[
                shard_id
            ] = shard_service

        return service

    def test_shard_assignment_is_deterministic(self):
        first_assignment = assign_shard(
            "DOC001",
            4,
        )

        second_assignment = assign_shard(
            "DOC001",
            4,
        )

        self.assertEqual(
            first_assignment,
            second_assignment,
        )

    def test_every_document_gets_exactly_one_shard(self):
        doc_ids = [
            "D1",
            "D2",
            "D3",
            "D4",
        ]

        assignments = [
            assign_shard(
                doc_id,
                4,
            )
            for doc_id in doc_ids
        ]

        self.assertEqual(
            len(assignments),
            len(doc_ids),
        )

        self.assertTrue(
            all(
                0 <= shard_id < 4
                for shard_id in assignments
            )
        )

    def test_invalid_shard_count_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_num_shards(0)

    def test_missing_distributed_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                DistributedBM25RetrievalService(
                    dataset_key="sample_dataset",
                    indexes_root=Path(directory),
                )

    @patch(
        "retrieval.distributed_bm25_service."
        "BM25RetrievalService"
    )
    def test_manifest_and_shards_are_accepted(
        self,
        mocked_bm25_service,
    ):
        with tempfile.TemporaryDirectory() as directory:
            self._write_distributed_manifest(
                Path(directory),
                num_shards=2,
            )

            service = DistributedBM25RetrievalService(
                dataset_key="sample_dataset",
                indexes_root=Path(directory),
            )

        self.assertEqual(
            service.num_shards,
            2,
        )

        self.assertEqual(
            mocked_bm25_service.call_count,
            2,
        )

    def test_num_shards_mismatch_is_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_distributed_manifest(
                Path(directory),
                num_shards=2,
            )

            with self.assertRaisesRegex(
                ValueError,
                "Requested num_shards",
            ):
                DistributedBM25RetrievalService(
                    dataset_key="sample_dataset",
                    indexes_root=Path(directory),
                    num_shards=3,
                )

    def test_coordinator_queries_all_shards(self):
        service = self._mocked_service({
            0: [
                {
                    "rank": 1,
                    "doc_id": "D1",
                    "score": 4.0,
                }
            ],
            1: [
                {
                    "rank": 1,
                    "doc_id": "D2",
                    "score": 3.0,
                }
            ],
        })

        results = service.search(
            query="diabetes insulin",
            top_k=2,
            shard_top_k=5,
            hydrate=False,
        )

        self.assertEqual(
            len(results),
            2,
        )

        for shard_service in (
            service.shard_services.values()
        ):
            shard_service.search.assert_called_once()

        self.assertEqual(
            service.last_search_metadata[
                "shards_queried"
            ],
            2,
        )

    def test_rrf_merge_is_deterministic(self):
        service = self._mocked_service({})

        results = service._reciprocal_rank_fusion(
            shard_results={
                "shard_1": [
                    {
                        "rank": 1,
                        "doc_id": "D2",
                        "score": 9.0,
                    }
                ],
                "shard_0": [
                    {
                        "rank": 1,
                        "doc_id": "D1",
                        "score": 8.0,
                    }
                ],
            },
            top_k=2,
            rrf_k=60,
        )

        self.assertEqual(
            [
                result["doc_id"]
                for result in results
            ],
            [
                "D1",
                "D2",
            ],
        )

        self.assertEqual(
            results[0]["merge_method"],
            "RRF",
        )

        self.assertTrue(
            results[0]["distributed"]
        )


class LTRFeatureExtractorTests(SimpleTestCase):
    def test_feature_names_are_stable_and_numeric(self):
        extractor = LTRFeatureExtractor(
            dataset_key="sample_dataset"
        )
        candidate = {
            "doc_id": "D1",
            "title": "Diabetes insulin",
            "snippet": (
                "Diabetes insulin treatment"
            ),
            "source_details": {
                "bm25": {
                    "rank": 1,
                    "score": 3.5,
                }
            },
        }

        features = extractor.extract_features(
            query="diabetes insulin",
            candidate=candidate,
        )

        self.assertEqual(
            list(features),
            extractor.feature_names,
        )

        self.assertTrue(
            all(
                isinstance(value, float)
                for value in features.values()
            )
        )

        self.assertEqual(
            features["bm25_score"],
            3.5,
        )

        self.assertEqual(
            features["in_tfidf"],
            0.0,
        )

    def test_missing_model_scores_are_safe(self):
        extractor = LTRFeatureExtractor(
            dataset_key="sample_dataset"
        )

        features = extractor.extract_features(
            query="machine learning",
            candidate={
                "doc_id": "D1",
                "source_details": {},
            },
        )

        self.assertEqual(
            features["bm25_score"],
            0.0,
        )

        self.assertEqual(
            features["bm25_rank"],
            0.0,
        )

        self.assertEqual(
            features["rrf_sum"],
            0.0,
        )

    def test_candidates_are_deduplicated_by_doc_id(self):
        candidates = merge_ltr_candidate_results({
            "bm25": [
                {
                    "rank": 1,
                    "doc_id": "D1",
                    "score": 2.0,
                }
            ],
            "tfidf": [
                {
                    "rank": 2,
                    "doc_id": "D1",
                    "score": 0.5,
                },
                {
                    "rank": 1,
                    "doc_id": "D2",
                    "score": 0.8,
                },
            ],
        })

        self.assertEqual(
            [
                candidate["doc_id"]
                for candidate in candidates
            ],
            [
                "D1",
                "D2",
            ],
        )

        self.assertEqual(
            set(
                candidates[0][
                    "source_details"
                ]
            ),
            {
                "bm25",
                "tfidf",
            },
        )

    def test_labels_are_created_from_qrels(self):
        labels = build_ltr_labels(
            candidates=[
                {
                    "doc_id": "D1",
                },
                {
                    "doc_id": "D2",
                },
            ],
            qrels_for_query={
                "D2": 2,
            },
        )

        self.assertEqual(
            labels,
            [
                0.0,
                2.0,
            ],
        )


class LTRRetrievalServiceTests(SimpleTestCase):
    def test_missing_trained_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = (
                Path(directory)
                / "missing_ltr.joblib"
            )

            with self.assertRaisesRegex(
                LTRModelNotTrainedError,
                "LTR model is not trained",
            ):
                LTRRetrievalService(
                    dataset_key="sample_dataset",
                    candidate_count=2,
                    candidate_models=["bm25"],
                    model_path=missing_path,
                )

    def test_reranks_candidates_with_mocked_model(self):
        candidate_generator = MagicMock()
        candidate_generator.generate.return_value = [
            {
                "doc_id": "D1",
                "title": "First",
                "snippet": "alpha",
                "candidate_sources": ["bm25"],
                "source_details": {
                    "bm25": {
                        "rank": 1,
                        "score": 10.0,
                    }
                },
            },
            {
                "doc_id": "D2",
                "title": "Second",
                "snippet": "beta",
                "candidate_sources": ["bm25"],
                "source_details": {
                    "bm25": {
                        "rank": 2,
                        "score": 1.0,
                    }
                },
            },
        ]
        candidate_generator.fetch_documents_for_candidates.return_value = {
            "D1": {
                "doc_id": "D1",
                "title": "First",
                "text": "alpha",
            },
            "D2": {
                "doc_id": "D2",
                "title": "Second",
                "text": "beta",
            },
        }

        service = LTRRetrievalService.__new__(
            LTRRetrievalService
        )
        service.dataset_key = "sample_dataset"
        service.candidate_count = 2
        service.candidate_models = ["bm25"]
        service.include_biomedical = False
        service.model_path = Path(
            "artifacts/models/ltr/sample_ltr.joblib"
        )
        service.training_metadata = {
            "dataset": "sample_dataset",
            "model_type": "MockModel",
        }
        service.feature_names = (
            LTRFeatureExtractor.feature_names
        )
        service.extractor = LTRFeatureExtractor(
            dataset_key="sample_dataset"
        )
        service.candidate_generator = (
            candidate_generator
        )
        service.model = MagicMock()
        service.model.predict.return_value = np.asarray(
            [
                0.1,
                0.9,
            ]
        )
        service.last_search_metadata = {}

        results = service.search(
            query="beta",
            top_k=2,
        )

        self.assertEqual(
            results[0]["doc_id"],
            "D2",
        )

        self.assertEqual(
            results[0]["model"],
            "ltr",
        )

        self.assertTrue(
            service.last_search_metadata["ltr"]
        )

    def test_training_script_saves_model_and_metadata(self):
        from scripts import train_ltr_model as trainer

        class FakeCandidateGenerator:
            def generate(self, query, candidate_count):
                return [
                    {
                        "doc_id": "D1",
                        "title": "Alpha",
                        "snippet": "alpha beta",
                        "candidate_sources": [
                            "bm25",
                        ],
                        "source_details": {
                            "bm25": {
                                "rank": 1,
                                "score": 1.0,
                            }
                        },
                    },
                    {
                        "doc_id": "D2",
                        "title": "Beta",
                        "snippet": "beta gamma",
                        "candidate_sources": [
                            "bm25",
                        ],
                        "source_details": {
                            "bm25": {
                                "rank": 2,
                                "score": 0.5,
                            }
                        },
                    },
                ]

            def fetch_documents_for_candidates(self, candidates):
                return {
                    candidate["doc_id"]: {
                        "doc_id": candidate["doc_id"],
                        "title": candidate["title"],
                        "text": candidate["snippet"],
                    }
                    for candidate in candidates
                }

        with tempfile.TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "sample_ltr.joblib"
            )

            with patch.object(
                trainer.DatasetLoader,
                "load_queries",
                return_value=[
                    {
                        "query_id": "Q1",
                        "query": "alpha",
                    },
                    {
                        "query_id": "Q2",
                        "query": "beta",
                    },
                ],
            ), patch.object(
                trainer.DatasetLoader,
                "load_qrels",
                return_value={
                    "Q1": {
                        "D1": 1,
                    },
                    "Q2": {
                        "D2": 1,
                    },
                },
            ):
                summary = trainer.train_ltr_model(
                    dataset_key="sample_dataset",
                    candidate_count=2,
                    output_path=output_path,
                    candidate_models=["bm25"],
                    validation_fraction=0.5,
                    random_seed=7,
                    candidate_generator=(
                        FakeCandidateGenerator()
                    ),
                )

            metadata_path = Path(
                summary["metadata_path"]
            )

            self.assertTrue(
                output_path.is_file()
            )

            self.assertTrue(
                metadata_path.is_file()
            )

            model = joblib.load(
                output_path
            )

            self.assertTrue(
                hasattr(model, "predict")
            )

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                metadata["dataset"],
                "sample_dataset",
            )

            self.assertEqual(
                metadata["feature_names"],
                LTRFeatureExtractor.feature_names,
            )


class LTREvaluationSupportTests(SimpleTestCase):
    def _evaluation_script_args(self, **overrides):
        defaults = {
            "models": ["ltr"],
            "retrieval_depth": 1000,
            "precision_k": 10,
            "recall_k": 1000,
            "ndcg_k": 10,
            "candidate_count": 1000,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "rrf_k": 60,
            "num_shards": 4,
            "shard_top_k": None,
            "ltr_candidate_models": None,
            "include_biomedical": True,
            "ltr_model_path": None,
            "biomedical_weight": 0.0,
            "use_query_refinement": False,
            "feedback_docs": 3,
            "expansion_terms": 5,
            "query_batch_size": None,
        }
        defaults.update(overrides)

        return SimpleNamespace(**defaults)

    def test_ltr_print_configuration_uses_default_model_path(self):
        from scripts import run_all_evaluations

        args = self._evaluation_script_args(
            ltr_model_path=None,
        )
        expected_path = (
            run_all_evaluations.resolve_ltr_model_path(
                dataset_key="clinical_trials",
                ltr_model_path=None,
            )
        )

        with patch("builtins.print") as mocked_print:
            run_all_evaluations.print_configuration(
                args=args,
                dataset_key="clinical_trials",
                output_path="reports/evaluation/out.csv",
            )

        printed_output = "\n".join(
            str(call.args[0])
            for call in mocked_print.call_args_list
            if call.args
        )

        self.assertIn(
            "LTR model path: ",
            printed_output,
        )
        self.assertIn(
            expected_path,
            printed_output,
        )

    def test_ltr_evaluation_script_passes_default_model_path(self):
        from scripts import run_all_evaluations

        args = self._evaluation_script_args(
            ltr_model_path=None,
        )
        expected_path = (
            run_all_evaluations.resolve_ltr_model_path(
                dataset_key="clinical_trials",
                ltr_model_path=None,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(
                Path(tmpdir) / "ltr.csv"
            )

            with (
                patch.object(
                    run_all_evaluations,
                    "print_configuration",
                ),
                patch.object(
                    run_all_evaluations,
                    "run_model_evaluation",
                    return_value={
                        "dataset": "clinical_trials",
                        "model": "ltr",
                    },
                ) as mocked_run_model,
                patch.object(
                    run_all_evaluations,
                    "save_results_to_csv",
                ),
            ):
                failure_count = (
                    run_all_evaluations.evaluate_dataset(
                        args=args,
                        dataset_key="clinical_trials",
                        output_path=output_path,
                    )
                )

        self.assertEqual(
            failure_count,
            0,
        )
        self.assertEqual(
            mocked_run_model.call_args.kwargs[
                "ltr_model_path"
            ],
            expected_path,
        )
        self.assertEqual(
            mocked_run_model.call_args.kwargs[
                "query_batch_size"
            ],
            1,
        )

    @patch(
        "evaluation.evaluator.LTRRetrievalService"
    )
    @patch(
        "evaluation.evaluator.DatasetLoader.load_qrels"
    )
    @patch(
        "evaluation.evaluator.DatasetLoader.load_queries"
    )
    def test_evaluation_supports_ltr_model(
        self,
        mocked_load_queries,
        mocked_load_qrels,
        mocked_ltr_service,
    ):
        from evaluation.evaluator import (
            EvaluationRunner,
        )

        mocked_load_queries.return_value = [
            {
                "query_id": "Q1",
                "query": "alpha",
            }
        ]
        mocked_load_qrels.return_value = {
            "Q1": {
                "D1": 1,
            }
        }

        service = MagicMock()
        service.model_path = Path(
            "artifacts/models/ltr/"
            "sample_dataset_ltr.joblib"
        )
        service.search.return_value = [
            {
                "rank": 1,
                "doc_id": "D1",
                "score": 0.9,
            }
        ]
        mocked_ltr_service.return_value = service

        runner = EvaluationRunner(
            dataset_key="sample_dataset",
            model_name="ltr",
            retrieval_depth=1,
            precision_k=1,
            recall_k=1,
            ndcg_k=1,
            candidate_count=2,
            ltr_candidate_models=["bm25"],
            ltr_model_path=(
                "artifacts/models/ltr/"
                "sample_dataset_ltr.joblib"
            ),
        )

        result = runner.evaluate()

        self.assertEqual(
            result["model"],
            "ltr",
        )

        self.assertEqual(
            result["candidate_count"],
            2,
        )

        self.assertEqual(
            result["ltr_candidate_models"],
            "bm25",
        )
        service.search_batch.assert_not_called()

    @patch(
        "evaluation.evaluator.gc.collect"
    )
    @patch(
        "evaluation.evaluator.LTRRetrievalService"
    )
    @patch(
        "evaluation.evaluator.DatasetLoader.load_qrels"
    )
    @patch(
        "evaluation.evaluator.DatasetLoader.load_queries"
    )
    def test_ltr_evaluation_streams_with_query_batch_size_one(
        self,
        mocked_load_queries,
        mocked_load_qrels,
        mocked_ltr_service,
        mocked_gc_collect,
    ):
        from evaluation.evaluator import (
            EvaluationRunner,
        )

        mocked_load_queries.return_value = [
            {
                "query_id": "Q1",
                "query": "alpha",
            },
            {
                "query_id": "Q2",
                "query": "beta",
            },
        ]
        mocked_load_qrels.return_value = {
            "Q1": {
                "D1": 1,
            },
            "Q2": {
                "D2": 1,
            },
        }

        service = MagicMock()
        service.model_path = Path(
            "artifacts/models/ltr/"
            "sample_dataset_ltr.joblib"
        )
        service.search.side_effect = [
            [
                {
                    "rank": 1,
                    "doc_id": "D1",
                    "score": 0.9,
                }
            ],
            [
                {
                    "rank": 1,
                    "doc_id": "D2",
                    "score": 0.8,
                }
            ],
        ]
        mocked_ltr_service.return_value = service

        runner = EvaluationRunner(
            dataset_key="sample_dataset",
            model_name="ltr",
            retrieval_depth=1,
            precision_k=1,
            recall_k=1,
            ndcg_k=1,
            candidate_count=2,
            ltr_candidate_models=["bm25"],
            ltr_model_path=(
                "artifacts/models/ltr/"
                "sample_dataset_ltr.joblib"
            ),
            query_batch_size=1,
        )

        result = runner.evaluate()

        self.assertEqual(
            result["retrieval_mode"],
            "ltr_streaming",
        )
        self.assertEqual(
            result["query_batch_size"],
            1,
        )
        self.assertEqual(
            result["evaluated_queries"],
            2,
        )
        self.assertEqual(
            service.search.call_count,
            2,
        )
        service.search_batch.assert_not_called()
        mocked_gc_collect.assert_called()

        for call in service.search.call_args_list:
            self.assertFalse(
                call.kwargs["hydrate"]
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
    def test_distributed_bm25_response_includes_metadata(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = {
            "results": [
                {
                    "rank": 1,
                    "doc_id": "DOC001",
                    "title": "Distributed title",
                    "snippet": "Distributed text",
                    "score": 0.01639344,
                    "model": "distributed_bm25",
                    "distributed": True,
                    "shard_id": 0,
                    "local_rank": 1,
                    "local_score": 12.5,
                    "merge_method": "RRF",
                }
            ],
            "metadata": {
                "distributed": True,
                "num_shards": 4,
                "shards_queried": 4,
                "shard_top_k": 100,
                "merge_method": "RRF",
                "rrf_k": 60,
                "shard_result_counts": {
                    "shard_0": 1,
                    "shard_1": 1,
                    "shard_2": 1,
                    "shard_3": 1,
                },
                "shard_document_counts": {
                    "shard_0": 10,
                    "shard_1": 10,
                    "shard_2": 10,
                    "shard_3": 10,
                },
            },
        }

        response = self.client.post(
            self.search_url,
            {
                "query": "machine learning",
                "dataset": "sample_dataset",
                "model": "distributed_bm25",
                "top_k": 1,
                "num_shards": 4,
                "shard_top_k": 100,
                "rrf_k": 60,
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertTrue(
            response.data["distributed"]
        )

        self.assertEqual(
            response.data["num_shards"],
            4,
        )

        self.assertEqual(
            response.data["shards_queried"],
            4,
        )

        self.assertEqual(
            response.data["merge_method"],
            "RRF",
        )

        self.assertEqual(
            response.data["shard_top_k"],
            100,
        )

        self.assertEqual(
            mocked_run_search.call_args.kwargs[
                "num_shards"
            ],
            4,
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_distributed_bm25_num_shards_mismatch_returns_400(
        self,
        mocked_run_search,
    ):
        mocked_run_search.side_effect = ValueError(
            "Requested num_shards does not match the built "
            "distributed BM25 index."
        )

        response = self.client.post(
            self.search_url,
            {
                "query": "machine learning",
                "dataset": "sample_dataset",
                "model": "distributed_bm25",
                "top_k": 1,
                "num_shards": 8,
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertIn(
            "num_shards",
            response.data["error"],
        )

    @patch(
        "retrieval.views.run_search"
    )
    def test_ltr_search_can_be_requested(
        self,
        mocked_run_search,
    ):
        mocked_run_search.return_value = {
            "results": [
                {
                    "rank": 1,
                    "doc_id": "DOC001",
                    "title": "LTR title",
                    "snippet": "LTR text",
                    "score": 0.9,
                    "model": "ltr",
                    "ltr_score": 0.9,
                }
            ],
            "metadata": {
                "ltr": True,
                "ltr_model_path": (
                    "artifacts/models/ltr/"
                    "sample_dataset_ltr.joblib"
                ),
                "candidate_count": 5,
                "candidate_models": [
                    "bm25",
                    "tfidf",
                ],
                "include_biomedical": False,
                "feature_count": 27,
                "training_metadata": {
                    "model_type": (
                        "GradientBoostingRegressor"
                    ),
                },
            },
        }

        response = self.client.post(
            self.search_url,
            {
                "query": "machine learning",
                "dataset": "sample_dataset",
                "model": "ltr",
                "top_k": 1,
                "candidate_count": 5,
                "ltr_candidate_models": [
                    "bm25",
                    "tfidf",
                ],
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertTrue(
            response.data["ltr"]
        )

        self.assertEqual(
            response.data["candidate_models"],
            [
                "bm25",
                "tfidf",
            ],
        )

        self.assertEqual(
            response.data["feature_count"],
            27,
        )

        self.assertEqual(
            mocked_run_search.call_args.kwargs[
                "model"
            ],
            "ltr",
        )

    def test_ltr_rejects_biomedical_for_quora(self):
        response = self.client.post(
            self.search_url,
            {
                "query": "machine learning",
                "dataset": "quora",
                "model": "ltr",
                "include_biomedical": True,
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertIn(
            "include_biomedical",
            response.data["error"],
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
                "distributed_bm25",
                "embedding",
                "biomedical_embedding",
                "hybrid_serial",
                "hybrid_parallel",
                "ltr",
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
