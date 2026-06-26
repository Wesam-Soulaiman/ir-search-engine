import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from django.conf import settings

from document_store.repository import (
    DocumentStoreRepository,
)
from indexing.distributed_bm25_index import (
    DEFAULT_DISTRIBUTED_NUM_SHARDS,
    DEFAULT_DISTRIBUTED_RRF_K,
    MERGE_METHOD,
    SHARDING_STRATEGY,
    validate_num_shards,
)
from retrieval.bm25_service import (
    BM25RetrievalService,
)


DEFAULT_DISTRIBUTED_SHARD_TOP_K_MINIMUM = 100
MAX_DISTRIBUTED_SHARD_TOP_K = 100_000


class DistributedBm25IndexError(RuntimeError):
    """
    Raised when the distributed BM25 index is missing or invalid.
    """


class DistributedBM25RetrievalService:
    """
    Local sharded Distributed IR coordinator for BM25.

    Each shard is an independent BM25 inverted index. The coordinator
    queries every shard in parallel and merges shard-local rankings with
    Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        dataset_key: str = "sample_dataset",
        num_shards: int | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        default_shard_top_k: int | None = None,
        default_rrf_k: int | None = None,
        indexes_root: str | Path | None = None,
        repository: DocumentStoreRepository | None = None,
        shard_services: Dict[int, Any] | None = None,
    ):
        self.dataset_key = str(
            dataset_key
        ).strip()
        self.requested_num_shards = (
            validate_num_shards(num_shards)
            if num_shards is not None
            else None
        )
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.default_shard_top_k = (
            self._validate_shard_top_k(
                default_shard_top_k
            )
            if default_shard_top_k is not None
            else None
        )
        self.default_rrf_k = int(
            default_rrf_k
            if default_rrf_k is not None
            else DEFAULT_DISTRIBUTED_RRF_K
        )

        if self.default_rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        if not self.dataset_key:
            raise ValueError(
                "dataset_key cannot be empty."
            )

        if self.bm25_k1 <= 0.0:
            raise ValueError(
                "BM25 k1 must be greater than zero."
            )

        if not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError(
                "BM25 b must be between 0 and 1."
            )

        root = Path(
            indexes_root
            if indexes_root is not None
            else getattr(
                settings,
                "INDEXES_DIR",
                settings.BASE_DIR.parent
                / "indexes",
            )
        ).expanduser().resolve()

        self.index_dir = (
            root
            / self.dataset_key
            / "distributed_bm25"
        )
        self.manifest_path = (
            self.index_dir / "manifest.json"
        )
        self.shards_dir = (
            self.index_dir / "shards"
        )

        self.repository = repository
        self.manifest = self._load_manifest()
        self.num_shards = int(
            self.manifest["num_shards"]
        )

        if (
            self.requested_num_shards is not None
            and self.requested_num_shards
            != self.num_shards
        ):
            raise ValueError(
                "Requested num_shards does not match "
                "the built distributed BM25 index. "
                f"Requested: {self.requested_num_shards}; "
                f"manifest: {self.num_shards}."
            )

        self.shard_document_counts = {
            str(shard_name): int(count)
            for shard_name, count
            in self.manifest.get(
                "shard_document_counts",
                {},
            ).items()
        }

        self.shard_services = (
            shard_services
            if shard_services is not None
            else self._load_shard_services()
        )

        self._validate_loaded_shards()
        self.last_search_metadata: Dict[
            str,
            Any,
        ] = {}

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                "Distributed BM25 index is missing. "
                "Build it first with "
                "'python backend/scripts/"
                "build_distributed_bm25_index.py "
                f"--dataset {self.dataset_key} "
                "--num-shards 4'. Missing manifest: "
                f"{self.manifest_path}"
            )

        manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(manifest, dict):
            raise DistributedBm25IndexError(
                "Distributed BM25 manifest must contain "
                "a JSON object."
            )

        if manifest.get("dataset") != self.dataset_key:
            raise DistributedBm25IndexError(
                "Distributed BM25 manifest dataset "
                "mismatch. "
                f"Expected '{self.dataset_key}', found "
                f"'{manifest.get('dataset')}'."
            )

        if (
            manifest.get("model")
            != "distributed_bm25"
        ):
            raise DistributedBm25IndexError(
                "Distributed BM25 manifest has an "
                "unsupported model: "
                f"{manifest.get('model')}"
            )

        manifest_num_shards = validate_num_shards(
            manifest.get(
                "num_shards",
                DEFAULT_DISTRIBUTED_NUM_SHARDS,
            )
        )

        total_documents = int(
            manifest.get(
                "total_documents",
                0,
            )
        )

        if total_documents <= 0:
            raise DistributedBm25IndexError(
                "Distributed BM25 manifest has an "
                "invalid total_documents value."
            )

        shard_document_counts = manifest.get(
            "shard_document_counts"
        )

        if not isinstance(
            shard_document_counts,
            dict,
        ):
            raise DistributedBm25IndexError(
                "Distributed BM25 manifest is missing "
                "shard_document_counts."
            )

        expected_shards = {
            f"shard_{shard_id}"
            for shard_id in range(
                manifest_num_shards
            )
        }

        if set(shard_document_counts) != expected_shards:
            raise DistributedBm25IndexError(
                "Distributed BM25 manifest shard "
                "counts do not match num_shards."
            )

        observed_total = sum(
            int(count)
            for count in shard_document_counts.values()
        )

        if observed_total != total_documents:
            raise DistributedBm25IndexError(
                "Distributed BM25 shard counts do "
                "not sum to total_documents."
            )

        if (
            manifest.get("sharding_strategy")
            != SHARDING_STRATEGY
        ):
            raise DistributedBm25IndexError(
                "Unsupported distributed BM25 "
                "sharding strategy: "
                f"{manifest.get('sharding_strategy')}"
            )

        if manifest.get("merge_method") != MERGE_METHOD:
            raise DistributedBm25IndexError(
                "Unsupported distributed BM25 merge "
                "method: "
                f"{manifest.get('merge_method')}"
            )

        return manifest

    def _load_shard_services(
        self,
    ) -> Dict[int, BM25RetrievalService]:
        shard_services: Dict[
            int,
            BM25RetrievalService,
        ] = {}

        missing_paths = []

        for shard_id in range(
            self.num_shards
        ):
            shard_dir = (
                self.shards_dir
                / f"shard_{shard_id}"
            )
            shard_manifest_path = (
                shard_dir
                / "shard_manifest.json"
            )
            bm25_manifest_path = (
                shard_dir
                / "manifest.json"
            )

            for path in [
                shard_dir,
                shard_manifest_path,
                bm25_manifest_path,
            ]:
                if not path.exists():
                    missing_paths.append(
                        path
                    )

            if missing_paths:
                continue

            self._validate_shard_manifest(
                shard_id=shard_id,
                shard_manifest_path=(
                    shard_manifest_path
                ),
            )

            shard_services[shard_id] = (
                BM25RetrievalService(
                    dataset_key=self.dataset_key,
                    k1=self.bm25_k1,
                    b=self.bm25_b,
                    use_saved_index=True,
                    index_dir=shard_dir,
                    validate_document_store_count=False,
                )
            )

        if missing_paths:
            formatted = "\n".join(
                f"- {path}"
                for path in missing_paths
            )

            raise FileNotFoundError(
                "Distributed BM25 shard artifacts "
                f"are missing:\n{formatted}"
            )

        return shard_services

    def _validate_shard_manifest(
        self,
        shard_id: int,
        shard_manifest_path: Path,
    ):
        manifest = json.loads(
            shard_manifest_path.read_text(
                encoding="utf-8"
            )
        )

        shard_name = f"shard_{shard_id}"

        if manifest.get("dataset") != self.dataset_key:
            raise DistributedBm25IndexError(
                f"{shard_name} dataset mismatch."
            )

        if manifest.get("model") != "distributed_bm25":
            raise DistributedBm25IndexError(
                f"{shard_name} has unsupported model "
                f"{manifest.get('model')}."
            )

        if int(manifest.get("shard_id", -1)) != shard_id:
            raise DistributedBm25IndexError(
                f"{shard_name} shard_id mismatch."
            )

        if int(manifest.get("num_shards", 0)) != self.num_shards:
            raise DistributedBm25IndexError(
                f"{shard_name} num_shards mismatch."
            )

        expected_count = int(
            self.shard_document_counts[
                shard_name
            ]
        )

        if (
            int(manifest.get("document_count", -1))
            != expected_count
        ):
            raise DistributedBm25IndexError(
                f"{shard_name} document count "
                "mismatch."
            )

    def _validate_loaded_shards(self):
        expected_ids = set(
            range(self.num_shards)
        )

        loaded_ids = {
            int(shard_id)
            for shard_id in self.shard_services
        }

        if loaded_ids != expected_ids:
            raise DistributedBm25IndexError(
                "Distributed BM25 did not load every "
                "shard. Expected shard IDs "
                f"{sorted(expected_ids)}, loaded "
                f"{sorted(loaded_ids)}."
            )

    @staticmethod
    def _validate_shard_top_k(
        shard_top_k: int,
    ) -> int:
        try:
            parsed = int(shard_top_k)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "shard_top_k must be an integer."
            ) from error

        if parsed <= 0:
            raise ValueError(
                "shard_top_k must be greater than zero."
            )

        if parsed > MAX_DISTRIBUTED_SHARD_TOP_K:
            raise ValueError(
                "shard_top_k must not exceed "
                f"{MAX_DISTRIBUTED_SHARD_TOP_K}."
            )

        return parsed

    @staticmethod
    def _validate_rrf_k(
        rrf_k: int,
    ) -> int:
        try:
            parsed = int(rrf_k)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "rrf_k must be an integer."
            ) from error

        if parsed <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        return parsed

    def _resolve_shard_top_k(
        self,
        top_k: int,
        shard_top_k: int | None,
    ) -> int:
        if shard_top_k is not None:
            return self._validate_shard_top_k(
                shard_top_k
            )

        if self.default_shard_top_k is not None:
            return self.default_shard_top_k

        return self._validate_shard_top_k(
            max(
                int(top_k) * 10,
                DEFAULT_DISTRIBUTED_SHARD_TOP_K_MINIMUM,
            )
        )

    def _get_repository(
        self,
    ) -> DocumentStoreRepository:
        if self.repository is None:
            database_path = getattr(
                settings,
                "CORPUS_DATABASE_PATH",
                None,
            )

            if database_path is None:
                raise DistributedBm25IndexError(
                    "CORPUS_DATABASE_PATH is not "
                    "configured."
                )

            self.repository = DocumentStoreRepository(
                database_path
            )
            self.repository.initialize()

        return self.repository

    def search(
        self,
        query: str,
        top_k: int = 10,
        shard_top_k: int | None = None,
        rrf_k: int | None = None,
        hydrate: bool = True,
        snippet_length: int = 250,
    ) -> List[Dict[str, Any]]:
        if not isinstance(query, str):
            raise ValueError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            self.last_search_metadata = (
                self._build_search_metadata(
                    shard_top_k=0,
                    rrf_k=(
                        self.default_rrf_k
                    ),
                    shard_result_counts={
                        f"shard_{shard_id}": 0
                        for shard_id in range(
                            self.num_shards
                        )
                    },
                    shards_queried=0,
                )
            )
            return []

        try:
            parsed_top_k = int(top_k)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "top_k must be an integer."
            ) from error

        if parsed_top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        parsed_shard_top_k = (
            self._resolve_shard_top_k(
                parsed_top_k,
                shard_top_k,
            )
        )

        parsed_rrf_k = self._validate_rrf_k(
            rrf_k
            if rrf_k is not None
            else self.default_rrf_k
        )

        shard_results = self._query_all_shards(
            query=query,
            shard_top_k=parsed_shard_top_k,
        )

        fused_results = (
            self._reciprocal_rank_fusion(
                shard_results=shard_results,
                top_k=parsed_top_k,
                rrf_k=parsed_rrf_k,
            )
        )

        if hydrate:
            fused_results = self._hydrate_results(
                results=fused_results,
                snippet_length=snippet_length,
            )

        shard_result_counts = {
            shard_name: len(results)
            for shard_name, results
            in shard_results.items()
        }

        self.last_search_metadata = (
            self._build_search_metadata(
                shard_top_k=parsed_shard_top_k,
                rrf_k=parsed_rrf_k,
                shard_result_counts=(
                    shard_result_counts
                ),
                shards_queried=len(
                    shard_results
                ),
            )
        )

        return fused_results

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        query_batch_size: int = 64,
        hydrate: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        if not isinstance(queries, list):
            raise ValueError(
                "queries must be a list of strings."
            )

        if query_batch_size <= 0:
            raise ValueError(
                "query_batch_size must be greater than zero."
            )

        results = []

        for query in queries:
            if not isinstance(query, str):
                raise ValueError(
                    "Every query must be a string."
                )

            results.append(
                self.search(
                    query=query,
                    top_k=top_k,
                    hydrate=hydrate,
                )
                if query.strip()
                else []
            )

        return results

    def _query_all_shards(
        self,
        query: str,
        shard_top_k: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        with ThreadPoolExecutor(
            max_workers=self.num_shards
        ) as executor:
            futures = {
                shard_id: executor.submit(
                    self._query_shard,
                    shard_id,
                    query,
                    shard_top_k,
                )
                for shard_id in range(
                    self.num_shards
                )
            }

            return {
                f"shard_{shard_id}": futures[
                    shard_id
                ].result()
                for shard_id in range(
                    self.num_shards
                )
            }

    def _query_shard(
        self,
        shard_id: int,
        query: str,
        shard_top_k: int,
    ) -> List[Dict[str, Any]]:
        service = self.shard_services[
            shard_id
        ]

        try:
            results = service.search(
                query=query,
                top_k=shard_top_k,
                hydrate=False,
            )
        except TypeError:
            results = service.search(
                query=query,
                top_k=shard_top_k,
            )

        normalized_results = []

        for position, result in enumerate(
            results,
            start=1,
        ):
            local_rank = int(
                result.get(
                    "rank",
                    position,
                )
                or position
            )

            normalized_result = dict(
                result
            )
            normalized_result[
                "shard_id"
            ] = shard_id
            normalized_result[
                "local_rank"
            ] = local_rank
            normalized_result[
                "local_score"
            ] = result.get("score")
            normalized_results.append(
                normalized_result
            )

        return normalized_results

    def _reciprocal_rank_fusion(
        self,
        shard_results: Dict[
            str,
            List[Dict[str, Any]],
        ],
        top_k: int,
        rrf_k: int,
    ) -> List[Dict[str, Any]]:
        fused_scores: Dict[str, float] = {}
        best_metadata: Dict[
            str,
            Dict[str, Any],
        ] = {}
        contributions: Dict[
            str,
            List[Dict[str, Any]],
        ] = {}

        for shard_name in sorted(
            shard_results,
            key=lambda name: int(
                name.split("_", 1)[1]
            ),
        ):
            shard_id = int(
                shard_name.split("_", 1)[1]
            )

            for position, result in enumerate(
                shard_results[shard_name],
                start=1,
            ):
                doc_id = str(
                    result.get("doc_id", "")
                    or ""
                ).strip()

                if not doc_id:
                    continue

                local_rank = int(
                    result.get(
                        "local_rank",
                        result.get(
                            "rank",
                            position,
                        ),
                    )
                    or position
                )

                if local_rank <= 0:
                    local_rank = position

                local_score = result.get(
                    "local_score",
                    result.get("score"),
                )

                contribution = (
                    1.0
                    / (
                        float(rrf_k)
                        + float(local_rank)
                    )
                )

                fused_scores[doc_id] = (
                    fused_scores.get(
                        doc_id,
                        0.0,
                    )
                    + contribution
                )

                contributions.setdefault(
                    doc_id,
                    [],
                ).append({
                    "shard_id": shard_id,
                    "shard_name": shard_name,
                    "local_rank": local_rank,
                    "local_score": local_score,
                    "rrf_contribution": round(
                        float(contribution),
                        8,
                    ),
                })

                current_best = (
                    best_metadata.get(doc_id)
                )

                candidate_best = {
                    "doc_id": doc_id,
                    "title": result.get(
                        "title"
                    ),
                    "snippet": result.get(
                        "snippet"
                    ),
                    "shard_id": shard_id,
                    "local_rank": local_rank,
                    "local_score": local_score,
                }

                if current_best is None:
                    best_metadata[
                        doc_id
                    ] = candidate_best
                    continue

                current_rank = int(
                    current_best.get(
                        "local_rank",
                        local_rank,
                    )
                )
                current_shard = int(
                    current_best.get(
                        "shard_id",
                        shard_id,
                    )
                )

                if (
                    local_rank,
                    shard_id,
                    doc_id,
                ) < (
                    current_rank,
                    current_shard,
                    doc_id,
                ):
                    best_metadata[
                        doc_id
                    ] = candidate_best

        ranked_doc_ids = sorted(
            fused_scores,
            key=lambda doc_id: (
                -fused_scores[doc_id],
                int(
                    best_metadata[doc_id][
                        "local_rank"
                    ]
                ),
                int(
                    best_metadata[doc_id][
                        "shard_id"
                    ]
                ),
                doc_id,
            ),
        )

        results = []

        for rank, doc_id in enumerate(
            ranked_doc_ids[:top_k],
            start=1,
        ):
            metadata = best_metadata[
                doc_id
            ]

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": metadata.get(
                    "title"
                ),
                "snippet": metadata.get(
                    "snippet"
                ),
                "score": round(
                    float(
                        fused_scores[doc_id]
                    ),
                    8,
                ),
                "model": "distributed_bm25",
                "distributed": True,
                "shard_id": metadata[
                    "shard_id"
                ],
                "local_rank": metadata[
                    "local_rank"
                ],
                "local_score": metadata[
                    "local_score"
                ],
                "merge_method": MERGE_METHOD,
                "model_details": {
                    "distributed_bm25": {
                        "merge_method": (
                            MERGE_METHOD
                        ),
                        "rrf_k": rrf_k,
                        "shard_contributions": (
                            contributions.get(
                                doc_id,
                                [],
                            )
                        ),
                    }
                },
            })

        return results

    def _hydrate_results(
        self,
        results: List[Dict[str, Any]],
        snippet_length: int,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        if snippet_length <= 0:
            raise ValueError(
                "snippet_length must be greater than zero."
            )

        repository = self._get_repository()

        doc_ids = [
            str(result["doc_id"])
            for result in results
        ]

        documents = repository.get_documents(
            dataset_key=self.dataset_key,
            doc_ids=doc_ids,
        )

        documents_by_id = {
            document["doc_id"]: document
            for document in documents
        }

        missing_doc_ids = [
            doc_id
            for doc_id in doc_ids
            if doc_id not in documents_by_id
        ]

        if missing_doc_ids:
            raise DistributedBm25IndexError(
                "Distributed BM25 returned document "
                "IDs missing from the document store. "
                f"Examples: {missing_doc_ids[:10]}"
            )

        hydrated_results = []

        for result in results:
            hydrated = dict(result)
            document = documents_by_id[
                hydrated["doc_id"]
            ]
            raw_text = str(
                document.get(
                    "raw_text",
                    "",
                )
                or ""
            )

            hydrated["title"] = str(
                document.get(
                    "title",
                    "",
                )
                or ""
            )
            hydrated["snippet"] = raw_text[
                :snippet_length
            ]
            hydrated_results.append(
                hydrated
            )

        return hydrated_results

    def _build_search_metadata(
        self,
        shard_top_k: int,
        rrf_k: int,
        shard_result_counts: Dict[str, int],
        shards_queried: int,
    ) -> Dict[str, Any]:
        return {
            "distributed": True,
            "num_shards": self.num_shards,
            "shards_queried": shards_queried,
            "shard_top_k": shard_top_k,
            "merge_method": MERGE_METHOD,
            "rrf_k": rrf_k,
            "shard_result_counts": (
                shard_result_counts
            ),
            "shard_document_counts": (
                self.shard_document_counts
            ),
        }

    def get_last_search_metadata(
        self,
    ) -> Dict[str, Any]:
        return dict(
            self.last_search_metadata
        )
