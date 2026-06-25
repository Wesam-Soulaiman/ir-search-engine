import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from document_store.repository import (
    DocumentStoreRepository,
)


class DocumentStoreRepositoryTests(
    SimpleTestCase
):
    def setUp(self):
        self.temporary_directory = (
            tempfile.TemporaryDirectory()
        )

        self.database_path = (
            Path(
                self.temporary_directory.name
            )
            / "test_corpus.sqlite3"
        )

        self.repository = (
            DocumentStoreRepository(
                self.database_path
            )
        )

        self.repository.initialize()

        self.repository.upsert_dataset(
            dataset_key="test_dataset",
            display_name="Test Dataset",
            ir_dataset_id="test/dataset",
            metadata={
                "purpose": "automated test",
            },
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_initialize_creates_database(self):
        self.assertTrue(
            self.database_path.is_file()
        )

    def test_dataset_can_be_created_and_read(self):
        dataset = self.repository.get_dataset(
            "test_dataset"
        )

        self.assertIsNotNone(dataset)

        self.assertEqual(
            dataset["dataset_key"],
            "test_dataset",
        )

        self.assertEqual(
            dataset["display_name"],
            "Test Dataset",
        )

        self.assertEqual(
            dataset["ir_dataset_id"],
            "test/dataset",
        )

        self.assertEqual(
            dataset["metadata"],
            {
                "purpose": "automated test",
            },
        )

    def test_documents_can_be_inserted_and_read(self):
        inserted_count = (
            self.repository.bulk_upsert_documents(
                "test_dataset",
                [
                    {
                        "doc_id": "D1",
                        "title": "First document",
                        "text": "First raw text.",
                    },
                    {
                        "doc_id": "D2",
                        "title": "Second document",
                        "text": "Second raw text.",
                    },
                ],
            )
        )

        self.assertEqual(
            inserted_count,
            2,
        )

        document = (
            self.repository.get_document(
                "test_dataset",
                "D1",
            )
        )

        self.assertIsNotNone(document)

        self.assertEqual(
            document["title"],
            "First document",
        )

        self.assertEqual(
            document["raw_text"],
            "First raw text.",
        )

    def test_document_upsert_updates_existing_record(
        self,
    ):
        self.repository.bulk_upsert_documents(
            "test_dataset",
            [
                {
                    "doc_id": "D1",
                    "title": "Old title",
                    "text": "Old text",
                },
            ],
        )

        self.repository.bulk_upsert_documents(
            "test_dataset",
            [
                {
                    "doc_id": "D1",
                    "title": "New title",
                    "text": "New text",
                },
            ],
        )

        document = (
            self.repository.get_document(
                "test_dataset",
                "D1",
            )
        )

        self.assertEqual(
            document["title"],
            "New title",
        )

        self.assertEqual(
            document["raw_text"],
            "New text",
        )

        counts = (
            self.repository.get_dataset_counts(
                "test_dataset"
            )
        )

        self.assertEqual(
            counts["documents"],
            1,
        )

    def test_get_documents_preserves_requested_order(
        self,
    ):
        self.repository.bulk_upsert_documents(
            "test_dataset",
            [
                {
                    "doc_id": "D1",
                    "title": "First",
                    "text": "First text",
                },
                {
                    "doc_id": "D2",
                    "title": "Second",
                    "text": "Second text",
                },
                {
                    "doc_id": "D3",
                    "title": "Third",
                    "text": "Third text",
                },
            ],
        )

        documents = (
            self.repository.get_documents(
                "test_dataset",
                [
                    "D3",
                    "D1",
                    "D2",
                ],
            )
        )

        self.assertEqual(
            [
                document["doc_id"]
                for document in documents
            ],
            [
                "D3",
                "D1",
                "D2",
            ],
        )

    def test_queries_can_be_inserted_and_read(self):
        inserted_count = (
            self.repository.bulk_upsert_queries(
                "test_dataset",
                [
                    {
                        "query_id": "Q1",
                        "query": "raw query",
                        "processed_query": (
                            "process queri"
                        ),
                    },
                ],
            )
        )

        self.assertEqual(
            inserted_count,
            1,
        )

        query = self.repository.get_query(
            "test_dataset",
            "Q1",
        )

        self.assertIsNotNone(query)

        self.assertEqual(
            query["raw_query"],
            "raw query",
        )

        self.assertEqual(
            query["processed_query"],
            "process queri",
        )

    def test_qrels_and_counts(self):
        self.repository.bulk_upsert_documents(
            "test_dataset",
            [
                {
                    "doc_id": "D1",
                    "title": "",
                    "text": "Raw document",
                },
            ],
        )

        self.repository.bulk_upsert_queries(
            "test_dataset",
            [
                {
                    "query_id": "Q1",
                    "query": "test query",
                },
            ],
        )

        self.repository.bulk_upsert_qrels(
            "test_dataset",
            [
                {
                    "query_id": "Q1",
                    "doc_id": "D1",
                    "relevance": 2,
                },
            ],
        )

        counts = (
            self.repository.update_dataset_counts(
                "test_dataset",
                mark_imported=True,
            )
        )

        self.assertEqual(
            counts,
            {
                "documents": 1,
                "queries": 1,
                "qrels": 1,
            },
        )

        dataset = self.repository.get_dataset(
            "test_dataset"
        )

        self.assertEqual(
            dataset["document_count"],
            1,
        )

        self.assertEqual(
            dataset["query_count"],
            1,
        )

        self.assertEqual(
            dataset["qrel_count"],
            1,
        )

        self.assertIsNotNone(
            dataset["imported_at"]
        )

    def test_missing_qrel_document_is_detected(
        self,
    ):
        self.repository.bulk_upsert_queries(
            "test_dataset",
            [
                {
                    "query_id": "Q1",
                    "query": "test query",
                },
            ],
        )

        self.repository.bulk_upsert_qrels(
            "test_dataset",
            [
                {
                    "query_id": "Q1",
                    "doc_id": "MISSING-DOC",
                    "relevance": 1,
                },
            ],
        )

        missing = (
            self.repository
            .find_missing_qrel_documents(
                "test_dataset"
            )
        )

        self.assertEqual(
            missing,
            ["MISSING-DOC"],
        )

    def test_missing_qrel_query_is_detected(
        self,
    ):
        self.repository.bulk_upsert_documents(
            "test_dataset",
            [
                {
                    "doc_id": "D1",
                    "title": "",
                    "text": "Raw document",
                },
            ],
        )

        self.repository.bulk_upsert_qrels(
            "test_dataset",
            [
                {
                    "query_id": "MISSING-QUERY",
                    "doc_id": "D1",
                    "relevance": 1,
                },
            ],
        )

        missing = (
            self.repository
            .find_missing_qrel_queries(
                "test_dataset"
            )
        )

        self.assertEqual(
            missing,
            ["MISSING-QUERY"],
        )

    def test_delete_dataset_cascades_records(
        self,
    ):
        self.repository.bulk_upsert_documents(
            "test_dataset",
            [
                {
                    "doc_id": "D1",
                    "title": "",
                    "text": "Raw document",
                },
            ],
        )

        self.repository.bulk_upsert_queries(
            "test_dataset",
            [
                {
                    "query_id": "Q1",
                    "query": "test query",
                },
            ],
        )

        self.repository.bulk_upsert_qrels(
            "test_dataset",
            [
                {
                    "query_id": "Q1",
                    "doc_id": "D1",
                    "relevance": 1,
                },
            ],
        )

        self.repository.delete_dataset(
            "test_dataset"
        )

        self.assertIsNone(
            self.repository.get_dataset(
                "test_dataset"
            )
        )

        self.assertEqual(
            self.repository.get_dataset_counts(
                "test_dataset"
            ),
            {
                "documents": 0,
                "queries": 0,
                "qrels": 0,
            },
        )

    def test_missing_document_returns_none(self):
        document = (
            self.repository.get_document(
                "test_dataset",
                "UNKNOWN",
            )
        )

        self.assertIsNone(document)

    def test_document_without_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.repository.bulk_upsert_documents(
                "test_dataset",
                [
                    {
                        "title": "Missing ID",
                        "text": "Text",
                    },
                ],
            )

    def test_query_without_text_is_rejected(self):
        with self.assertRaises(ValueError):
            self.repository.bulk_upsert_queries(
                "test_dataset",
                [
                    {
                        "query_id": "Q1",
                        "query": "",
                    },
                ],
            )