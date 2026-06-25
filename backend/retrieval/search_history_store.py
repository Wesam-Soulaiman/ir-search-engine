import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List


class SearchHistoryStore:
    """
    Local SQLite store for anonymous search history.
    """

    def __init__(
        self,
        database_path: str | Path,
    ):
        self.database_path = Path(
            database_path
        ).expanduser().resolve()

    def initialize(self):
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    query TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_search_history_user_dataset_created
                ON search_history (
                    user_id,
                    dataset,
                    created_at DESC,
                    id DESC
                )
                """
            )

    @contextmanager
    def connection(
        self,
    ) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
        )

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

    def record_query(
        self,
        user_id: str,
        dataset: str,
        query: str,
    ):
        normalized_user_id = str(user_id or "").strip()
        normalized_dataset = str(dataset or "").strip()
        normalized_query = str(query or "").strip()

        if (
            not normalized_user_id
            or not normalized_dataset
            or not normalized_query
        ):
            return

        self.initialize()

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO search_history (
                    user_id,
                    dataset,
                    query,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    normalized_user_id,
                    normalized_dataset,
                    normalized_query,
                    self._utc_now(),
                ),
            )

    def get_recent_queries(
        self,
        user_id: str,
        dataset: str,
        limit: int = 20,
    ) -> List[str]:
        normalized_user_id = str(user_id or "").strip()
        normalized_dataset = str(dataset or "").strip()

        if not normalized_user_id or not normalized_dataset:
            return []

        result_limit = max(
            1,
            int(limit),
        )

        self.initialize()

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT query
                FROM search_history
                WHERE user_id = ?
                  AND dataset = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (
                    normalized_user_id,
                    normalized_dataset,
                    result_limit,
                ),
            ).fetchall()

        return [
            str(row[0])
            for row in rows
        ]
