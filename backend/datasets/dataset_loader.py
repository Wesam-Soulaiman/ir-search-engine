import csv
import json
import os
from typing import Dict, List

from datasets.dataset_registry import get_dataset_config


class DatasetLoader:
    @staticmethod
    def load_documents(dataset_key: str) -> List[Dict]:
        config = get_dataset_config(dataset_key)
        documents_path = config["documents_path"]
        dataset_format = config["format"]

        if not os.path.exists(documents_path):
            raise FileNotFoundError(
                f"Documents file not found for dataset '{dataset_key}': "
                f"{documents_path}"
            )

        if dataset_format == "json":
            return DatasetLoader._load_json_documents(documents_path)

        if dataset_format == "jsonl":
            return DatasetLoader._load_jsonl_documents(documents_path)

        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    @staticmethod
    def _load_json_documents(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as file:
            documents = json.load(file)

        return [DatasetLoader._normalize_document(doc) for doc in documents]

    @staticmethod
    def _load_jsonl_documents(path: str) -> List[Dict]:
        documents = []

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                doc = json.loads(line)
                documents.append(DatasetLoader._normalize_document(doc))

        return documents

    @staticmethod
    def _normalize_document(doc: Dict) -> Dict:
        doc_id = (
            doc.get("doc_id")
            or doc.get("id")
            or doc.get("_id")
            or doc.get("docno")
        )

        title = doc.get("title") or ""
        text = doc.get("text") or doc.get("contents") or doc.get("body") or ""

        if not doc_id:
            raise ValueError(f"Document is missing an id field: {doc}")

        return {
            "doc_id": str(doc_id),
            "title": str(title),
            "text": str(text),
        }

    @staticmethod
    def load_queries(dataset_key: str) -> List[Dict]:
        config = get_dataset_config(dataset_key)
        queries_path = config.get("queries_path")

        if not queries_path:
            return []

        if not os.path.exists(queries_path):
            raise FileNotFoundError(
                f"Queries file not found for dataset '{dataset_key}': "
                f"{queries_path}"
            )

        queries = []

        with open(queries_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")

            for row in reader:
                if len(row) < 2:
                    continue

                queries.append({
                    "query_id": row[0],
                    "query": row[1],
                })

        return queries

    @staticmethod
    def load_qrels(dataset_key: str) -> Dict[str, Dict[str, int]]:
        config = get_dataset_config(dataset_key)
        qrels_path = config.get("qrels_path")

        if not qrels_path:
            return {}

        if not os.path.exists(qrels_path):
            raise FileNotFoundError(
                f"Qrels file not found for dataset '{dataset_key}': "
                f"{qrels_path}"
            )

        qrels = {}

        with open(qrels_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue

                query_id = row[0]

                # Common qrels formats:
                # qid doc_id relevance
                # qid 0 doc_id relevance
                if len(row) == 3:
                    doc_id = row[1]
                    relevance = int(row[2])
                else:
                    doc_id = row[2]
                    relevance = int(row[3])

                if query_id not in qrels:
                    qrels[query_id] = {}

                qrels[query_id][doc_id] = relevance

        return qrels