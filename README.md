# IR Search Engine

A full Information Retrieval search engine built with **Django REST Framework** and **React**.
The system supports multiple retrieval models, large-scale indexing, hybrid ranking, query refinement, biomedical retrieval, distributed retrieval simulation, and supervised Learning-to-Rank reranking.

This project was built as a final-year Information Retrieval project and includes complete indexing, searching, evaluation, and a web interface.

---

## Features

### Retrieval Models

The system supports the following retrieval models:

* **TF-IDF / Vector Space Model**
* **BM25**
* **Dense Embedding Retrieval**

  * Sentence Transformers
  * FAISS vector search
* **Hybrid Serial Retrieval**

  * BM25 candidate generation
  * Embedding reranking
* **Hybrid Parallel Retrieval**

  * TF-IDF, BM25, and Embedding search in parallel
  * Weighted Reciprocal Rank Fusion
* **Biomedical PubMedBERT Retrieval**

  * Clinical Trials only
  * Biomedical embedding model
  * FAISS vector index
* **Distributed BM25**

  * Local distributed IR simulation
  * Corpus split into multiple shards
  * Coordinator queries all shards
  * Results merged using RRF
* **Learning-to-Rank (LTR)**

  * Candidate generation from BM25, TF-IDF, Embedding, and optionally Biomedical retrieval
  * Feature extraction
  * Supervised ML reranking using qrels
  * Trained model loaded from disk

---

## Extra IR Features

The system also includes:

* Query preprocessing
* Dataset-aware normalization
* Pseudo Relevance Feedback
* Offline spelling correction
* Search personalization using local search history
* Topic detection
* Document clustering
* Rule-based search agent
* Evaluation using standard IR metrics
* React UI with advanced search controls

---

## Datasets

The project supports two large datasets:

### 1. Quora

* Documents: more than 500,000
* Queries: 10,000
* Qrels: available
* Main use case: general question retrieval

### 2. Clinical Trials

* Documents: more than 240,000
* Queries: 50
* Qrels: available
* Main use case: biomedical / clinical search

---

## Evaluation Metrics

The project evaluates models using:

* MAP
* Precision@10
* Recall@K
* nDCG@10
* Query time
* Queries per second

---

## Latest Evaluation Results

### Quora

| Model           |      MAP | Precision@10 |   Recall |  nDCG@10 |
| --------------- | -------: | -----------: | -------: | -------: |
| TF-IDF          | 0.689935 |     0.111890 | 0.952802 | 0.731330 |
| BM25            | 0.720676 |     0.116750 | 0.963569 | 0.761794 |
| Embedding       | 0.794472 |     0.128190 | 0.979004 | 0.831295 |
| Hybrid Parallel | 0.774977 |     0.124330 | 0.988270 | 0.813707 |
| Hybrid Serial   | 0.839377 |     0.132550 | 0.987909 | 0.872116 |
| LTR             | 0.824619 |     0.131600 | 0.997103 | 0.860768 |

Notes:

* LTR was fully trained on Quora.
* LTR evaluation was completed over 10,000 queries.
* LTR used candidate_count = 500.
* LTR achieved very strong performance and the highest recall among the listed Quora configurations.

---

### Clinical Trials

| Model               |      MAP | Precision@10 |   Recall |  nDCG@10 |
| ------------------- | -------: | -----------: | -------: | -------: |
| BM25                | 0.265665 |        0.440 | 0.742024 | 0.419820 |
| Distributed BM25    | 0.251163 |        0.438 | 0.740266 | 0.385522 |
| Hybrid + Biomedical | 0.199293 |        0.364 | 0.719760 | 0.320801 |
| LTR                 | 0.352012 |        0.550 | 0.785058 | 0.570328 |

Notes:

* LTR achieved the best Clinical Trials results among the tested configurations.
* Distributed BM25 preserved results close to centralized BM25 while demonstrating a distributed retrieval architecture.
* Biomedical PubMedBERT improved the system with a domain-specific retrieval component.

---

## Architecture

The project follows a **modular Service-Oriented Architecture** inside a Django backend.

Each major retrieval component is implemented as a separate service module with a clear responsibility.

```text
React Frontend
    |
    v
Django REST API
    |
    v
Search Controller / API View
    |
    +--> TF-IDF Service
    +--> BM25 Service
    +--> Embedding Service
    +--> Biomedical Embedding Service
    +--> Hybrid Serial Service
    +--> Hybrid Parallel Service
    +--> Distributed BM25 Service
    +--> LTR Service
    +--> Query Refinement Services
    +--> Evaluation Service
    |
    v
Indexes / FAISS / SQLite Document Store / Model Artifacts
```

The current implementation is a modular SOA inside one backend application.
It can be extended into full microservices by exposing each retrieval service as an independent REST or gRPC service.

---

## Main Backend Components

```text
backend/
  evaluation/
    evaluator.py

  indexing/
    distributed_bm25_index.py
    ...

  query_refinement/
    spelling_correction_service.py
    ...

  retrieval/
    bm25_service.py
    tfidf_service.py
    embedding_service.py
    biomedical_embedding_service.py
    hybrid_serial_service.py
    hybrid_parallel_service.py
    distributed_bm25_service.py
    ltr_feature_extractor.py
    ltr_service.py
    rag_answer_generator.py
    rag_llm_client.py
    rag_service.py
    personalization_service.py
    views.py
    tests.py

  scripts/
    train_ltr_model.py
    run_all_evaluations.py
    build_distributed_bm25_index.py
    build_biomedical_embedding_index.py
    ...
```

---

## Frontend

The frontend is built with React and provides:

* Dataset selector
* Retrieval model selector
* Advanced search controls
* Hybrid weights
* BM25 parameters
* LTR candidate model controls
* Biomedical option for Clinical Trials
* Query refinement options
* Search result explanations and metadata

```text
frontend/
  src/
    App.jsx
    api/
    components/
    config/
```

---

## Requirements

Recommended environment:

* Python 3.12 64-bit
* Node.js 18+
* npm
* Windows PowerShell or compatible terminal

Install Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
```

---

## Important Artifact Notice

Large generated files are not included in Git.

The following folders may be required for full local search functionality:

```text
artifacts/database/
artifacts/models/
indexes/
```

These may include:

* SQLite document store
* FAISS indexes
* TF-IDF indexes
* BM25 indexes
* Biomedical embedding model
* LTR trained models
* Distributed BM25 shard indexes

If these folders are missing, the application code can run, but some search models may return errors such as:

```text
index not found
model not found
document store not found
LTR model is not trained
```

To run the full system on another machine, copy prepared `artifacts/` and `indexes/` folders into the project root.

---

## Running the Backend

From the project root:

```powershell
.\.venv\Scripts\activate
python .\backend\manage.py migrate
python .\backend\manage.py runserver
```

Backend URL:

```text
http://127.0.0.1:8000
```

---

## Running the Frontend

In a second terminal:

```powershell
cd frontend
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

## Search API Example

Endpoint:

```text
POST /api/search/
```

Example BM25 request:

```json
{
  "dataset": "quora",
  "model": "bm25",
  "query": "what causes nightmares",
  "top_k": 10
}
```

Example LTR request:

```json
{
  "dataset": "quora",
  "model": "ltr",
  "query": "what causes nightmares",
  "top_k": 5,
  "candidate_count": 500,
  "ltr_candidate_models": ["bm25", "tfidf", "embedding"]
}
```

Example Clinical LTR request with Biomedical candidates:

```json
{
  "dataset": "clinical_trials",
  "model": "ltr",
  "query": "diabetes insulin treatment",
  "top_k": 5,
  "candidate_count": 1000,
  "ltr_candidate_models": ["bm25", "tfidf", "embedding"],
  "include_biomedical": true
}
```

---

## Optional Local LLM RAG with Ollama

RAG is available through `model: "rag"`.

The default generation mode remains:

```text
extractive_offline
```

This mode uses deterministic offline extractive answer synthesis and does not require Ollama or any LLM server.

An optional local LLM mode is also available:

```text
local_llm
```

Local LLM RAG uses a local Ollama server and the default model:

```text
llama3.2:3b
```

Setup:

```powershell
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

Local Ollama API:

```text
http://localhost:11434
```

No cloud API key is required. The project still runs without Ollama; only `model: "rag"` with `rag_generation_mode: "local_llm"` requires the local Ollama server and model to be available.

Example local LLM RAG request:

```json
{
  "dataset": "quora",
  "model": "rag",
  "query": "what causes nightmares",
  "top_k": 10,
  "rag_retriever_model": "hybrid_serial",
  "rag_generation_mode": "local_llm",
  "rag_llm_provider": "ollama",
  "rag_llm_model": "llama3.2:3b",
  "rag_llm_base_url": "http://localhost:11434",
  "rag_llm_temperature": 0.0,
  "rag_llm_max_tokens": 350
}
```

---

## Training LTR Models

LTR models are not trained automatically during API requests.

### Train LTR on Quora

```powershell
.\.venv\Scripts\python.exe .\backend\scripts\train_ltr_model.py `
  --dataset quora `
  --candidate-count 500 `
  --output .\artifacts\models\ltr\quora_ltr.joblib
```

### Train LTR on Clinical Trials

```powershell
.\.venv\Scripts\python.exe .\backend\scripts\train_ltr_model.py `
  --dataset clinical_trials `
  --candidate-count 1000 `
  --include-biomedical `
  --output .\artifacts\models\ltr\clinical_trials_ltr.joblib
```

Generated LTR artifacts:

```text
artifacts/models/ltr/quora_ltr.joblib
artifacts/models/ltr/quora_ltr_metadata.json
artifacts/models/ltr/clinical_trials_ltr.joblib
artifacts/models/ltr/clinical_trials_ltr_metadata.json
```

These files should not be committed to Git.

---

## Running Evaluation

### Evaluate Quora LTR

```powershell
.\.venv\Scripts\python.exe .\backend\scripts\run_all_evaluations.py `
  --datasets quora `
  --models ltr `
  --retrieval-depth 500 `
  --precision-k 10 `
  --recall-k 500 `
  --ndcg-k 10 `
  --candidate-count 500 `
  --query-batch-size 1 `
  --ltr-model-path .\artifacts\models\ltr\quora_ltr.joblib `
  --output .\reports\evaluation\quora_ltr_c500.csv
```

### Evaluate Clinical Trials LTR

```powershell
.\.venv\Scripts\python.exe .\backend\scripts\run_all_evaluations.py `
  --datasets clinical_trials `
  --models ltr `
  --retrieval-depth 1000 `
  --precision-k 10 `
  --recall-k 1000 `
  --ndcg-k 10 `
  --candidate-count 1000 `
  --include-biomedical `
  --query-batch-size 1 `
  --ltr-model-path .\artifacts\models\ltr\clinical_trials_ltr.joblib `
  --output .\reports\evaluation\clinical_trials_ltr.csv
```

### Evaluate Distributed BM25

```powershell
.\.venv\Scripts\python.exe .\backend\scripts\run_all_evaluations.py `
  --datasets clinical_trials `
  --models distributed_bm25 `
  --retrieval-depth 1000 `
  --precision-k 10 `
  --recall-k 1000 `
  --ndcg-k 10 `
  --candidate-count 1000 `
  --num-shards 4 `
  --shard-top-k 1000 `
  --rrf-k 60 `
  --output .\reports\evaluation\clinical_trials_distributed_bm25.csv
```

---

## Distributed BM25

Distributed BM25 simulates Distributed Information Retrieval locally.

Process:

```text
Corpus
  |
  v
Hash-based sharding
  |
  +--> Shard 0 BM25 Index
  +--> Shard 1 BM25 Index
  +--> Shard 2 BM25 Index
  +--> Shard 3 BM25 Index
  |
  v
Coordinator queries all shards
  |
  v
RRF merge
  |
  v
Final ranked results
```

Build distributed BM25 index:

```powershell
.\.venv\Scripts\python.exe .\backend\scripts\build_distributed_bm25_index.py `
  --dataset clinical_trials `
  --num-shards 4
```

---

## Biomedical PubMedBERT Retrieval

Biomedical retrieval is designed for the Clinical Trials dataset.

It uses a biomedical embedding model and FAISS vector search.

Search example:

```json
{
  "dataset": "clinical_trials",
  "model": "biomedical_embedding",
  "query": "diabetes insulin treatment",
  "top_k": 10
}
```

Biomedical retrieval is not available for Quora.

---

## Query Refinement

The system supports several query refinement techniques:

* Pseudo Relevance Feedback
* Offline spelling correction
* Search personalization

The spelling correction module works offline and does not require external APIs.

Personalization uses local search history and can influence future searches for the same user/session.

---

## Testing

Run backend tests:

```powershell
.\run_backend_tests.ps1
```

Or manually:

```powershell
cd backend
..\.venv\Scripts\python.exe manage.py test
```

Run frontend checks:

```powershell
cd frontend
npm run lint
npm run build
```

---

## Git Notes

Do not commit generated artifacts:

```text
artifacts/models/
artifacts/database/
indexes/
reports/evaluation/
hub/
*.joblib
*.csv
*.sqlite3
```

Recommended commit command for code changes:

```powershell
git add backend frontend README.md .gitignore requirements.txt
git commit -m "Update project documentation"
git push origin main
```

Avoid:

```powershell
git add .
```

because it may accidentally stage generated models, indexes, or evaluation reports.

---

## Project Status

Implemented:

* TF-IDF retrieval
* BM25 retrieval
* Embedding retrieval
* Hybrid Serial retrieval
* Hybrid Parallel retrieval
* Biomedical PubMedBERT retrieval
* Distributed BM25
* Learning-to-Rank
* Spelling correction
* Personalization
* Query refinement
* Evaluation pipeline
* React web interface

Future possible extensions:

* Full RAG answer generation
* Full multilingual retrieval
* Real remote microservices deployment
* Web crawling pipeline
* Advanced neural reranking
* Online user feedback learning

---

## Summary

This project demonstrates a complete Information Retrieval system with traditional lexical models, dense retrieval, hybrid fusion, biomedical retrieval, distributed retrieval, and supervised Learning-to-Rank. It includes large-scale datasets, local indexing, a React search interface, and a full evaluation pipeline using standard IR metrics.
