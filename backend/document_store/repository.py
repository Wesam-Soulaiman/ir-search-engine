import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


class DocumentStoreError(RuntimeError):
    """
    Raised when a document-store operation fails.
    """


class DocumentStoreRepository:
    """
    SQLite repository for raw IR datasets.

    The repository stores:

    - Dataset metadata.
    - Original raw documents.
    - Original queries.
    - Relevance judgments.

    Retrieval indexes remain separate from this database.
    """

    def __init__(
        self,
        database_path: str | Path,
    ):
        self.database_path = Path(
            database_path
        ).expanduser().resolve()

        self.schema_path = (
            Path(__file__).resolve().parent
            / "schema.sql"
        )

    def initialize(self):
        """
        Create the database directory and tables.
        """
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.schema_path.is_file():
            raise FileNotFoundError(
                "Document-store schema file "
                f"not found: {self.schema_path}"
            )

        schema = self.schema_path.read_text(
            encoding="utf-8"
        )

        with self.connection() as connection:
            connection.executescript(schema)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=60.0,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        connection.execute(
            "PRAGMA synchronous = NORMAL"
        )

        connection.execute(
            "PRAGMA temp_store = MEMORY"
        )

        return connection

    @contextmanager
    def connection(
        self,
    ) -> Iterator[sqlite3.Connection]:
        connection = self._connect()

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _json_dump(
        value: Optional[Dict[str, Any]],
    ) -> str:
        return json.dumps(
            value or {},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _json_load(value: str) -> Dict[str, Any]:
        if not value:
            return {}

        return json.loads(value)

    def upsert_dataset(
        self,
        dataset_key: str,
        display_name: str,
        ir_dataset_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        dataset_key = str(
            dataset_key
        ).strip()

        display_name = str(
            display_name
        ).strip()

        ir_dataset_id = str(
            ir_dataset_id
        ).strip()

        if not dataset_key:
            raise ValueError(
                "dataset_key cannot be empty."
            )

        if not display_name:
            raise ValueError(
                "display_name cannot be empty."
            )

        if not ir_dataset_id:
            raise ValueError(
                "ir_dataset_id cannot be empty."
            )

        updated_at = self._utc_now()

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO datasets (
                    dataset_key,
                    display_name,
                    ir_dataset_id,
                    metadata_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)

                ON CONFLICT(dataset_key)
                DO UPDATE SET
                    display_name = excluded.display_name,
                    ir_dataset_id = excluded.ir_dataset_id,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    dataset_key,
                    display_name,
                    ir_dataset_id,
                    self._json_dump(metadata),
                    updated_at,
                ),
            )

    def delete_dataset(
        self,
        dataset_key: str,
    ):
        with self.connection() as connection:
            connection.execute(
                """
                DELETE FROM datasets
                WHERE dataset_key = ?
                """,
                (dataset_key,),
            )

    def bulk_upsert_documents(
        self,
        dataset_key: str,
        documents: Iterable[Dict[str, Any]],
    ) -> int:
        rows = []

        for document in documents:
            doc_id = str(
                document.get(
                    "doc_id",
                    "",
                )
            ).strip()

            if not doc_id:
                raise ValueError(
                    "A document is missing doc_id."
                )

            rows.append((
                dataset_key,
                doc_id,
                str(
                    document.get(
                        "title",
                        "",
                    )
                    or ""
                ),
                str(
                    document.get(
                        "text",
                        document.get(
                            "raw_text",
                            "",
                        ),
                    )
                    or ""
                ),
                self._json_dump(
                    document.get("metadata")
                ),
            ))

        if not rows:
            return 0

        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO documents (
                    dataset_key,
                    doc_id,
                    title,
                    raw_text,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?)

                ON CONFLICT(dataset_key, doc_id)
                DO UPDATE SET
                    title = excluded.title,
                    raw_text = excluded.raw_text,
                    metadata_json = excluded.metadata_json
                """,
                rows,
            )

        return len(rows)

    def bulk_upsert_queries(
        self,
        dataset_key: str,
        queries: Iterable[Dict[str, Any]],
    ) -> int:
        rows = []

        for query in queries:
            query_id = str(
                query.get(
                    "query_id",
                    "",
                )
            ).strip()

            raw_query = str(
                query.get(
                    "query",
                    query.get(
                        "raw_query",
                        "",
                    ),
                )
            ).strip()

            if not query_id:
                raise ValueError(
                    "A query is missing query_id."
                )

            if not raw_query:
                raise ValueError(
                    f"Query '{query_id}' has empty text."
                )

            rows.append((
                dataset_key,
                query_id,
                raw_query,
                str(
                    query.get(
                        "processed_query",
                        "",
                    )
                    or ""
                ),
                self._json_dump(
                    query.get("metadata")
                ),
            ))

        if not rows:
            return 0

        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO queries (
                    dataset_key,
                    query_id,
                    raw_query,
                    processed_query,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?)

                ON CONFLICT(dataset_key, query_id)
                DO UPDATE SET
                    raw_query = excluded.raw_query,
                    processed_query = excluded.processed_query,
                    metadata_json = excluded.metadata_json
                """,
                rows,
            )

        return len(rows)

    def bulk_upsert_qrels(
        self,
        dataset_key: str,
        qrels: Iterable[Dict[str, Any]],
    ) -> int:
        rows = []

        for qrel in qrels:
            query_id = str(
                qrel.get(
                    "query_id",
                    "",
                )
            ).strip()

            doc_id = str(
                qrel.get(
                    "doc_id",
                    "",
                )
            ).strip()

            if not query_id or not doc_id:
                raise ValueError(
                    "A qrel is missing query_id "
                    "or doc_id."
                )

            relevance = int(
                qrel.get(
                    "relevance",
                    0,
                )
            )

            iteration = str(
                qrel.get(
                    "iteration",
                    "0",
                )
            ).strip() or "0"

            rows.append((
                dataset_key,
                query_id,
                doc_id,
                relevance,
                iteration,
            ))

        if not rows:
            return 0

        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO qrels (
                    dataset_key,
                    query_id,
                    doc_id,
                    relevance,
                    iteration
                )
                VALUES (?, ?, ?, ?, ?)

                ON CONFLICT(
                    dataset_key,
                    query_id,
                    doc_id,
                    iteration
                )
                DO UPDATE SET
                    relevance = excluded.relevance
                """,
                rows,
            )

        return len(rows)

    def update_dataset_counts(
        self,
        dataset_key: str,
        mark_imported: bool = False,
    ) -> Dict[str, int]:
        counts = self.get_dataset_counts(
            dataset_key
        )

        imported_at = (
            self._utc_now()
            if mark_imported
            else None
        )

        with self.connection() as connection:
            connection.execute(
                """
                UPDATE datasets
                SET
                    document_count = ?,
                    query_count = ?,
                    qrel_count = ?,
                    imported_at = COALESCE(
                        ?,
                        imported_at
                    ),
                    updated_at = ?
                WHERE dataset_key = ?
                """,
                (
                    counts["documents"],
                    counts["queries"],
                    counts["qrels"],
                    imported_at,
                    self._utc_now(),
                    dataset_key,
                ),
            )

        return counts

    def get_dataset_counts(
        self,
        dataset_key: str,
    ) -> Dict[str, int]:
        with self.connection() as connection:
            document_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM documents
                WHERE dataset_key = ?
                """,
                (dataset_key,),
            ).fetchone()[0]

            query_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM queries
                WHERE dataset_key = ?
                """,
                (dataset_key,),
            ).fetchone()[0]

            qrel_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM qrels
                WHERE dataset_key = ?
                """,
                (dataset_key,),
            ).fetchone()[0]

        return {
            "documents": int(document_count),
            "queries": int(query_count),
            "qrels": int(qrel_count),
        }

    def get_dataset(
        self,
        dataset_key: str,
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM datasets
                WHERE dataset_key = ?
                """,
                (dataset_key,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["metadata"] = self._json_load(
            result.pop("metadata_json")
        )

        return result

    def get_document(
        self,
        dataset_key: str,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    dataset_key,
                    doc_id,
                    title,
                    raw_text,
                    metadata_json
                FROM documents
                WHERE dataset_key = ?
                  AND doc_id = ?
                """,
                (
                    dataset_key,
                    str(doc_id),
                ),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["metadata"] = self._json_load(
            result.pop("metadata_json")
        )

        return result

    def get_documents(
        self,
        dataset_key: str,
        doc_ids: Iterable[str],
    ) -> List[Dict[str, Any]]:
        ordered_ids = [
            str(doc_id)
            for doc_id in doc_ids
        ]

        if not ordered_ids:
            return []

        unique_ids = list(
            dict.fromkeys(ordered_ids)
        )

        placeholders = ",".join(
            "?"
            for _ in unique_ids
        )

        parameters = [
            dataset_key,
            *unique_ids,
        ]

        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    dataset_key,
                    doc_id,
                    title,
                    raw_text,
                    metadata_json
                FROM documents
                WHERE dataset_key = ?
                  AND doc_id IN ({placeholders})
                """,
                parameters,
            ).fetchall()

        documents_by_id = {}

        for row in rows:
            result = dict(row)
            result["metadata"] = (
                self._json_load(
                    result.pop(
                        "metadata_json"
                    )
                )
            )

            documents_by_id[
                result["doc_id"]
            ] = result

        # Preserve the retrieval ranking.
        return [
            documents_by_id[doc_id]
            for doc_id in ordered_ids
            if doc_id in documents_by_id
        ]

    def get_query(
        self,
        dataset_key: str,
        query_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    dataset_key,
                    query_id,
                    raw_query,
                    processed_query,
                    metadata_json
                FROM queries
                WHERE dataset_key = ?
                  AND query_id = ?
                """,
                (
                    dataset_key,
                    str(query_id),
                ),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["metadata"] = self._json_load(
            result.pop("metadata_json")
        )

        return result

    def find_missing_qrel_documents(
        self,
        dataset_key: str,
        limit: int = 100,
    ) -> List[str]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT q.doc_id
                FROM qrels AS q

                LEFT JOIN documents AS d
                  ON d.dataset_key = q.dataset_key
                 AND d.doc_id = q.doc_id

                WHERE q.dataset_key = ?
                  AND d.doc_id IS NULL

                ORDER BY q.doc_id
                LIMIT ?
                """,
                (
                    dataset_key,
                    int(limit),
                ),
            ).fetchall()

        return [
            str(row["doc_id"])
            for row in rows
        ]

    def find_missing_qrel_queries(
        self,
        dataset_key: str,
        limit: int = 100,
    ) -> List[str]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT r.query_id
                FROM qrels AS r

                LEFT JOIN queries AS q
                  ON q.dataset_key = r.dataset_key
                 AND q.query_id = r.query_id

                WHERE r.dataset_key = ?
                  AND q.query_id IS NULL

                ORDER BY r.query_id
                LIMIT ?
                """,
                (
                    dataset_key,
                    int(limit),
                ),
            ).fetchall()

        return [
            str(row["query_id"])
            for row in rows
        ]