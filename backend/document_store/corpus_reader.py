from dataclasses import dataclass
from typing import Dict, Iterator, List

from document_store.repository import (
    DocumentStoreRepository,
)


@dataclass(frozen=True)
class CorpusDocument:
    """
    One raw document read from the SQLite document store.
    """

    doc_id: str
    title: str
    raw_text: str

    @property
    def search_text(self) -> str:
        """
        Return the text supplied to lexical preprocessing.
        """
        return (
            f"{self.title} {self.raw_text}"
        ).strip()


class CorpusReader:
    """
    Stream complete datasets from SQLite in bounded batches.

    This reader avoids loading all documents into memory and will be used
    by the scalable TF-IDF, BM25, embedding, and clustering pipelines.
    """

    def __init__(
        self,
        repository: DocumentStoreRepository,
        dataset_key: str,
        batch_size: int = 1000,
    ):
        self.repository = repository
        self.dataset_key = str(
            dataset_key
        ).strip()

        self.batch_size = int(
            batch_size
        )

        if not self.dataset_key:
            raise ValueError(
                "dataset_key cannot be empty."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self._validate_dataset()

    def _validate_dataset(self):
        dataset = self.repository.get_dataset(
            self.dataset_key
        )

        if dataset is None:
            raise ValueError(
                f"Dataset '{self.dataset_key}' "
                "is not present in the document store."
            )

    def count_documents(self) -> int:
        counts = (
            self.repository.get_dataset_counts(
                self.dataset_key
            )
        )

        return int(
            counts["documents"]
        )

    def iter_documents(
        self,
    ) -> Iterator[CorpusDocument]:
        """
        Yield documents one at a time while fetching bounded SQLite
        batches internally.
        """
        with self.repository.connection() as connection:
            cursor = connection.execute(
                """
                SELECT
                    doc_id,
                    title,
                    raw_text
                FROM documents
                WHERE dataset_key = ?
                ORDER BY rowid
                """,
                (self.dataset_key,),
            )

            while True:
                rows = cursor.fetchmany(
                    self.batch_size
                )

                if not rows:
                    break

                for row in rows:
                    yield CorpusDocument(
                        doc_id=str(
                            row["doc_id"]
                        ),
                        title=str(
                            row["title"] or ""
                        ),
                        raw_text=str(
                            row["raw_text"] or ""
                        ),
                    )

    def iter_document_batches(
        self,
    ) -> Iterator[List[CorpusDocument]]:
        """
        Yield complete batches for indexing services that operate in
        batches.
        """
        with self.repository.connection() as connection:
            cursor = connection.execute(
                """
                SELECT
                    doc_id,
                    title,
                    raw_text
                FROM documents
                WHERE dataset_key = ?
                ORDER BY rowid
                """,
                (self.dataset_key,),
            )

            while True:
                rows = cursor.fetchmany(
                    self.batch_size
                )

                if not rows:
                    break

                yield [
                    CorpusDocument(
                        doc_id=str(
                            row["doc_id"]
                        ),
                        title=str(
                            row["title"] or ""
                        ),
                        raw_text=str(
                            row["raw_text"] or ""
                        ),
                    )
                    for row in rows
                ]

    def iter_search_texts(
        self,
    ) -> Iterator[str]:
        """
        Yield title + raw text for vectorizer-style APIs.
        """
        for document in self.iter_documents():
            yield document.search_text

    def iter_document_ids(
        self,
    ) -> Iterator[str]:
        """
        Yield document IDs in stable index order.
        """
        with self.repository.connection() as connection:
            cursor = connection.execute(
                """
                SELECT doc_id
                FROM documents
                WHERE dataset_key = ?
                ORDER BY rowid
                """,
                (self.dataset_key,),
            )

            while True:
                rows = cursor.fetchmany(
                    self.batch_size
                )

                if not rows:
                    break

                for row in rows:
                    yield str(
                        row["doc_id"]
                    )

    def get_summary(self) -> Dict:
        """
        Return information useful for logs and index manifests.
        """
        dataset = self.repository.get_dataset(
            self.dataset_key
        )

        counts = (
            self.repository.get_dataset_counts(
                self.dataset_key
            )
        )

        return {
            "dataset_key": self.dataset_key,
            "display_name": dataset[
                "display_name"
            ],
            "ir_dataset_id": dataset[
                "ir_dataset_id"
            ],
            "document_count": counts[
                "documents"
            ],
            "query_count": counts[
                "queries"
            ],
            "qrel_count": counts[
                "qrels"
            ],
            "batch_size": self.batch_size,
            "database_path": str(
                self.repository.database_path
            ),
        }