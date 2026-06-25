PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_key TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    ir_dataset_id TEXT NOT NULL,
    document_count INTEGER NOT NULL DEFAULT 0,
    query_count INTEGER NOT NULL DEFAULT 0,
    qrel_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    imported_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    dataset_key TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    raw_text TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',

    PRIMARY KEY (dataset_key, doc_id),

    FOREIGN KEY (dataset_key)
        REFERENCES datasets(dataset_key)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS queries (
    dataset_key TEXT NOT NULL,
    query_id TEXT NOT NULL,
    raw_query TEXT NOT NULL,
    processed_query TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',

    PRIMARY KEY (dataset_key, query_id),

    FOREIGN KEY (dataset_key)
        REFERENCES datasets(dataset_key)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS qrels (
    dataset_key TEXT NOT NULL,
    query_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    relevance INTEGER NOT NULL,
    iteration TEXT NOT NULL DEFAULT '0',

    PRIMARY KEY (
        dataset_key,
        query_id,
        doc_id,
        iteration
    ),

    FOREIGN KEY (dataset_key)
        REFERENCES datasets(dataset_key)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_dataset
ON documents(dataset_key);

CREATE INDEX IF NOT EXISTS idx_queries_dataset
ON queries(dataset_key);

CREATE INDEX IF NOT EXISTS idx_qrels_query
ON qrels(dataset_key, query_id);

CREATE INDEX IF NOT EXISTS idx_qrels_document
ON qrels(dataset_key, doc_id);

CREATE INDEX IF NOT EXISTS idx_qrels_relevance
ON qrels(dataset_key, relevance);