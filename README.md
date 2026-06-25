# Information Retrieval Search Engine

A full-stack Information Retrieval system built with **Django REST Framework** and **React**.
The project supports large-scale offline search over two real datasets, multiple retrieval models, query refinement, hybrid retrieval, FAISS vector search, document clustering, topic detection, and an automatic retrieval strategy agent.

---

## 1. Project Overview

This project implements a complete Information Retrieval pipeline:

* Dataset loading and preprocessing
* Offline raw document storage
* Efficient indexing
* Multiple retrieval models
* Hybrid retrieval
* Query refinement
* Evaluation using official qrels
* REST API
* Dark-mode React interface
* Additional IR features integrated into the system

The system is designed to run locally and offline after the required datasets, indexes, and models are prepared.

---

## 2. Supported Datasets

The system currently supports two datasets.

| Dataset           |                      Domain | Approx. Documents | Purpose                                             |
| ----------------- | --------------------------: | ----------------: | --------------------------------------------------- |
| `quora`           |           General questions |           522,931 | General natural-language question retrieval         |
| `clinical_trials` | Biomedical / medical trials |           241,006 | Clinical-trial and biomedical information retrieval |

Each dataset has its own:

* corpus
* queries
* qrels
* preprocessing configuration
* indexes
* evaluation reports

Retrieval and evaluation are dataset-scoped. The user selects one dataset before searching to avoid mixing different domains and to keep evaluation consistent with the dataset qrels.

---

## 3. Main Features

### Core Requirements

* Full corpus retrieval over two datasets
* Text preprocessing
* TF-IDF retrieval
* BM25 retrieval
* Embedding retrieval
* Hybrid Serial retrieval
* Hybrid Parallel retrieval
* Query refinement using pseudo-relevance feedback
* Evaluation with MAP, Precision@10, Recall, and nDCG
* Web interface for search and parameter control
* REST API
* Original raw document display from the local document store

### Additional Features

* FAISS Vector Store
* Document Clustering
* Topic Detection
* Retrieval Strategy Agent
* Weighted Hybrid Fusion controls
* Charts for cluster/topic analysis
* Dark-mode interactive frontend

---

## 4. Architecture

The project follows a service-oriented structure. Each major responsibility is separated into its own module.

```text
ir-search-engine/
├── backend/
│   ├── agents/
│   │   └── retrieval_strategy_agent.py
│   ├── clustering/
│   │   └── clustering_service.py
│   ├── datasets/
│   │   ├── dataset_loader.py
│   │   └── dataset_registry.py
│   ├── document_store/
│   │   └── repository.py
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── indexing/
│   │   ├── scalable_tfidf_index.py
│   │   ├── scalable_bm25_index.py
│   │   └── scalable_embedding_index.py
│   ├── preprocessing/
│   │   └── text_preprocessor.py
│   ├── query_refinement/
│   │   └── pseudo_relevance_feedback.py
│   ├── retrieval/
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── topic_views.py
│   │   ├── tfidf_service.py
│   │   ├── bm25_service.py
│   │   ├── embedding_service.py
│   │   ├── hybrid_serial_service.py
│   │   └── hybrid_parallel_service.py
│   └── scripts/
│       ├── build_scalable_tfidf_index.py
│       ├── build_scalable_bm25_index.py
│       ├── build_scalable_embedding_index.py
│       ├── build_document_clusters.py
│       ├── build_cluster_topics.py
│       └── run_all_evaluations.py
├── frontend/
│   └── src/
│       ├── api/
│       ├── components/
│       ├── config/
│       ├── App.jsx
│       └── App.css
├── artifacts/
├── data/
├── indexes/
├── reports/
└── README.md
```

---

## 5. Data Storage and Caching

The system uses offline artifacts to avoid loading or processing the entire corpus on every request.

### Raw Document Store

Original documents are stored in a local SQLite database:

```text
artifacts/database/corpus.sqlite3
```

The search API enriches ranked document IDs with:

* `doc_id`
* title
* snippet
* raw text if requested
* document source metadata

### Indexes

Indexes are stored under:

```text
indexes/<dataset>/
```

Examples:

```text
indexes/quora/tfidf/
indexes/quora/bm25/
indexes/quora/embedding/
indexes/clinical_trials/tfidf/
indexes/clinical_trials/bm25/
indexes/clinical_trials/embedding/
```

### Caching

The API uses cached service constructors so expensive retrieval services are not reloaded on every request.

Examples:

* TF-IDF service cache
* BM25 service cache
* Embedding service cache
* Hybrid retrieval service cache
* Query refinement service cache
* Retrieval strategy agent cache

---

## 6. Preprocessing

The preprocessing layer is dataset-aware.

### Quora

Quora contains short general-language questions, so preprocessing focuses on:

* lowercasing
* tokenization
* stopword handling
* stemming
* preserving meaningful negations

### Clinical Trials

Clinical Trials contains biomedical terminology, so preprocessing is more conservative:

* biomedical terms are preserved
* short medical tokens are kept when useful
* aggressive stemming is avoided
* terms such as gene names, mutations, and drug expressions are preserved where possible

This distinction is important because medical retrieval often depends on exact biomedical terms.

---

## 7. Retrieval Models

### 7.1 TF-IDF

TF-IDF is the lexical Vector Space Model baseline.

It represents documents and queries as weighted term vectors and ranks documents using similarity between the query vector and document vectors.

Use case:

* strong lexical baseline
* useful for exact vocabulary overlap
* interpretable term-based retrieval

---

### 7.2 BM25

BM25 is a probabilistic lexical retrieval model.

It is especially strong for Clinical Trials because exact biomedical terms are often important.

Main parameters:

| Parameter | Meaning                                |
| --------- | -------------------------------------- |
| `bm25_k1` | controls term-frequency saturation     |
| `bm25_b`  | controls document-length normalization |

Default values:

```text
bm25_k1 = 1.5
bm25_b = 0.75
```

---

### 7.3 Embedding Retrieval

Embedding retrieval uses dense vector representations and FAISS indexes.

The system uses a local saved embedding model and saved FAISS indexes so it can run offline after setup.

Use case:

* semantic retrieval
* useful when query and document use different words with similar meaning
* important for general natural-language search

---

### 7.4 Hybrid Serial Retrieval

Hybrid Serial retrieval works in two stages:

```text
Stage 1: BM25 retrieves candidate documents.
Stage 2: Embedding retrieval reranks the BM25 candidates.
```

This combines lexical candidate selection with semantic reranking.

This is useful when:

* exact terms matter
* semantic similarity is still needed
* the search space should be reduced before embedding reranking

---

### 7.5 Hybrid Parallel Retrieval

Hybrid Parallel retrieval runs multiple retrieval models independently:

```text
TF-IDF
BM25
Embedding
```

Then it fuses the ranked lists using Reciprocal Rank Fusion.

The current system supports **Weighted RRF**, so each retrieval model can have a different influence.

Weighted RRF:

```text
final_score(document) += model_weight / (rrf_k + rank_in_model)
```

Available weights:

```text
tfidf_weight
bm25_weight
embedding_weight
```

This allows the user to control the fusion behavior from the frontend.

Examples:

* increase `bm25_weight` for stronger lexical exact-match influence
* increase `embedding_weight` for stronger semantic influence
* increase `tfidf_weight` to strengthen the vector-space baseline contribution

---

## 8. Query Refinement

The system supports pseudo-relevance feedback.

When query refinement is enabled:

1. BM25 retrieves initial feedback documents.
2. Important terms are extracted from the top feedback documents.
3. The original query is expanded with selected terms.
4. The selected retrieval model runs using the refined query.

Parameters:

| Parameter              | Meaning                                   |
| ---------------------- | ----------------------------------------- |
| `use_query_refinement` | enable or disable PRF                     |
| `feedback_docs`        | number of top documents used for feedback |
| `expansion_terms`      | number of terms added to the query        |

Query refinement is useful as an enhancement, but it may also cause query drift, especially in specialized domains. Therefore, the system exposes it as an optional setting.

---

## 9. Retrieval Strategy Agent

The project includes a lightweight rule-based retrieval strategy agent.

The agent chooses the retrieval model based on:

* dataset
* query length
* query language
* domain characteristics

Example behavior:

| Case                       | Agent Decision                                                             |
| -------------------------- | -------------------------------------------------------------------------- |
| Clinical Trials query      | BM25                                                                       |
| Short Quora query          | Hybrid Serial                                                              |
| Longer general Quora query | Embedding                                                                  |
| Arabic query               | Multilingual mode, with fallback until multilingual service is implemented |

The API exposes this through:

```text
model = agent
```

The response includes:

```text
requested_model
executed_model
agent_selected_model
agent_reason
agent_features
agent_fallback
```

This makes the decision explainable in the interface.

---

## 10. Document Clustering

The project includes document clustering as an additional feature.

The clustering pipeline uses saved document embeddings and MiniBatchKMeans to group similar documents.

Generated reports:

```text
reports/clustering/quora_cluster_summary.csv
reports/clustering/clinical_trials_cluster_summary.csv
```

The API exposes clustering results:

```text
GET /api/clusters/quora/
GET /api/clusters/clinical_trials/
```

The frontend displays:

* cluster IDs
* document counts
* representative document IDs
* bar charts for cluster size distribution

Purpose:

* understand corpus structure
* inspect dominant groups of documents
* support exploratory analysis beyond ranked search

---

## 11. Topic Detection

Topic detection labels the document clusters using representative terms.

Generated reports:

```text
reports/topics/quora_cluster_topics.csv
reports/topics/clinical_trials_cluster_topics.csv
```

The API exposes topic detection results:

```text
GET /api/topics/quora/
GET /api/topics/clinical_trials/
```

The frontend displays:

* topic labels
* top terms
* document counts
* representative documents
* topic size charts

Purpose:

* explain what each cluster represents
* make clustering interpretable
* support visual analysis of the corpus

---

## 12. Evaluation

Evaluation is performed using the official query and qrels files for each dataset.

The evaluator does not use manually invented queries.
It evaluates the retrieval models using the dataset qrels.

Metrics:

| Metric       | Meaning                                       |
| ------------ | --------------------------------------------- |
| MAP          | Mean Average Precision                        |
| Precision@10 | Precision in top 10 results                   |
| Recall       | Fraction of relevant documents retrieved      |
| nDCG         | Ranking quality with graded position discount |

---

## 13. Evaluation Results

### Quora

| Model           |      MAP |     P@10 |   Recall |     nDCG |      Time |
| --------------- | -------: | -------: | -------: | -------: | --------: |
| TF-IDF          | 0.689935 | 0.111890 | 0.952802 | 0.731330 |  106.097s |
| BM25            | 0.720676 | 0.116750 | 0.963569 | 0.761794 |  225.972s |
| Embedding       | 0.794472 | 0.128190 | 0.979004 | 0.831295 |  152.174s |
| Hybrid Serial   | 0.839377 | 0.132550 | 0.987909 | 0.872116 |  695.127s |
| Hybrid Parallel | 0.774977 | 0.124330 | 0.988270 | 0.813707 | 8468.611s |

Best Quora result:

```text
Hybrid Serial achieved the best MAP and nDCG on Quora.
```

---

### Clinical Trials

| Model           |      MAP |  P@10 |   Recall |     nDCG |     Time |
| --------------- | -------: | ----: | -------: | -------: | -------: |
| TF-IDF          | 0.123258 | 0.272 | 0.318945 | 0.233755 |        - |
| BM25            | 0.227589 | 0.440 | 0.410081 | 0.419820 |   5.071s |
| Embedding       | 0.017345 | 0.092 | 0.068110 | 0.071522 |   2.119s |
| Hybrid Serial   | 0.030604 | 0.110 | 0.144656 | 0.079190 |  16.570s |
| Hybrid Parallel | 0.158198 | 0.352 | 0.360931 | 0.304596 | 114.286s |

Best Clinical Trials result:

```text
BM25 achieved the best MAP, Precision@10, and nDCG on Clinical Trials.
```

This makes sense because Clinical Trials contains specialized biomedical terminology where exact term matching is highly important.

---

## 14. Backend Setup

### 14.1 Create and activate virtual environment

```powershell
cd "C:\Users\LONOVO\Desktop\Final Year 5\IR\ir-search-engine"

python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### 14.2 Install dependencies

```powershell
pip install -r requirements.txt
```

If your environment has separate backend requirements, use:

```powershell
pip install -r backend\requirements.txt
```

---

## 15. Build Offline Artifacts

The project depends on local artifacts such as document stores and indexes.

Run the relevant scripts from the repository root.

### 15.1 Build TF-IDF indexes

```powershell
python .\backend\scripts\build_scalable_tfidf_index.py --dataset quora

python .\backend\scripts\build_scalable_tfidf_index.py --dataset clinical_trials
```

### 15.2 Build BM25 indexes

```powershell
python .\backend\scripts\build_scalable_bm25_index.py --dataset quora

python .\backend\scripts\build_scalable_bm25_index.py --dataset clinical_trials
```

### 15.3 Build embedding / FAISS indexes

```powershell
python .\backend\scripts\build_scalable_embedding_index.py --dataset quora

python .\backend\scripts\build_scalable_embedding_index.py --dataset clinical_trials
```

### 15.4 Build document clusters

```powershell
python .\backend\scripts\build_document_clusters.py --dataset quora

python .\backend\scripts\build_document_clusters.py --dataset clinical_trials
```

### 15.5 Build cluster topics

```powershell
python .\backend\scripts\build_cluster_topics.py --dataset quora

python .\backend\scripts\build_cluster_topics.py --dataset clinical_trials
```

---

## 16. Run Backend

```powershell
cd "C:\Users\LONOVO\Desktop\Final Year 5\IR\ir-search-engine"

.\.venv\Scripts\Activate.ps1

python .\backend\manage.py runserver
```

Default backend URL:

```text
http://127.0.0.1:8000/
```

---

## 17. Run Frontend

```powershell
cd "C:\Users\LONOVO\Desktop\Final Year 5\IR\ir-search-engine\frontend"

npm install

npm run dev
```

Default frontend URL:

```text
http://127.0.0.1:5173/
```

---

## 18. API Endpoints

### Search

```text
POST /api/search/
```

Example:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/search/" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "dataset": "quora",
    "model": "agent",
    "query": "what causes nightmares",
    "top_k": 5
  }'
```

---

### Search with Hybrid Parallel Weights

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/search/" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "dataset": "quora",
    "model": "hybrid_parallel",
    "query": "what causes nightmares",
    "top_k": 5,
    "candidate_count": 1000,
    "rrf_k": 60,
    "tfidf_weight": 1.0,
    "bm25_weight": 2.0,
    "embedding_weight": 1.0
  }'
```

---

### Get Original Document

```text
GET /api/documents/<dataset>/<doc_id>/
```

Example:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/documents/clinical_trials/NCT00405587/" `
  -Method Get
```

---

### Get Cluster Summary

```text
GET /api/clusters/<dataset>/
```

Example:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/clusters/quora/" `
  -Method Get
```

---

### Get Cluster Topics

```text
GET /api/topics/<dataset>/
```

Example:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/topics/quora/" `
  -Method Get
```

---

## 19. Frontend Features

The React interface supports:

* selecting dataset
* selecting retrieval model
* searching by query
* setting `top_k`
* controlling BM25 parameters
* controlling hybrid candidate count
* controlling RRF parameter
* controlling hybrid parallel weights
* enabling query refinement
* setting feedback documents and expansion terms
* enabling raw text display
* viewing agent decision
* viewing search results with document IDs and scores
* viewing clusters and topics
* viewing cluster/topic charts

Available models in the UI:

```text
agent
tfidf
bm25
embedding
hybrid_serial
hybrid_parallel
```

---

## 20. Demo Checklist

Before presentation, run this checklist.

### Backend

```powershell
python -m py_compile `
  .\backend\retrieval\views.py `
  .\backend\retrieval\topic_views.py `
  .\backend\retrieval\hybrid_parallel_service.py `
  .\backend\agents\retrieval_strategy_agent.py
```

### API Tests

Test each endpoint:

```text
/api/search/
/api/clusters/quora/
/api/topics/quora/
/api/clusters/clinical_trials/
/api/topics/clinical_trials/
```

### Frontend Tests

From the UI, test:

* `tfidf`
* `bm25`
* `embedding`
* `hybrid_serial`
* `hybrid_parallel`
* `agent`
* query refinement on/off
* raw text on/off
* hybrid parallel weights
* topic detection panel
* clustering panel

---

## 21. Important Notes

* Retrieval and evaluation are performed per dataset.
* The two datasets are not mixed during evaluation.
* The system uses dataset qrels for evaluation.
* Query refinement is optional because it can improve or harm performance depending on the dataset.
* BM25 performs best on Clinical Trials because biomedical search depends heavily on exact specialized terms.
* Hybrid Serial performs best on Quora because it combines lexical candidate retrieval with semantic reranking.
* Clustering and Topic Detection are analysis features, not primary ranking models.
* Agent mode is explainable and returns the selected and executed retrieval strategy.

---

## 22. Tech Stack

### Backend

* Python
* Django
* Django REST Framework
* scikit-learn
* rank-bm25
* sentence-transformers
* FAISS
* NumPy
* SciPy
* pandas
* SQLite

### Frontend

* React
* Vite
* Axios
* CSS modules / component-based CSS structure

---

## 23. Summary

This project implements a complete offline Information Retrieval system over two large datasets. It supports multiple retrieval models, efficient indexes, raw document storage, query refinement, hybrid retrieval, vector search, clustering, topic detection, weighted hybrid controls, and an explainable retrieval strategy agent. The system is accessible through a Django REST API and a dark-mode React interface designed for live demonstration and experimentation.
