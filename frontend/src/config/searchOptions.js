export const DATASETS = [
  {
    value: "quora",
    label: "Quora",
    description: "General question-answer retrieval dataset",
    accent: "General",
  },
  {
    value: "clinical_trials",
    label: "Clinical Trials",
    description: "Biomedical clinical-trial retrieval dataset",
    accent: "Medical",
  },
];

export const MODELS = [
  {
    value: "agent",
    label: "Auto Agent",
    description: "Automatically chooses the best retrieval strategy.",
    tag: "Smart",
  },
  {
    value: "tfidf",
    label: "TF-IDF",
    description: "Vector Space Model lexical retrieval.",
    tag: "Lexical",
  },
  {
    value: "bm25",
    label: "BM25",
    description: "Probabilistic lexical ranking model.",
    tag: "Strong baseline",
  },
  {
    value: "embedding",
    label: "Embedding",
    description: "Semantic search using FAISS vector store.",
    tag: "Semantic",
  },
  {
    value: "hybrid_serial",
    label: "Hybrid Serial",
    description: "BM25 candidates with embedding reranking.",
    tag: "Best Quora",
  },
  {
    value: "hybrid_parallel",
    label: "Hybrid Parallel",
    description: "Weighted fusion of TF-IDF, BM25, and Embedding results.",
    tag: "Weighted fusion",
  },
];

export const EXAMPLE_QUERIES = {
  quora: [
    "what causes nightmares",
    "best programming language to learn",
    "how to improve memory",
  ],
  clinical_trials: [
    "melanoma BRAF V600E clinical trial",
    "EGFR lung cancer trial",
    "diabetes insulin glucose treatment",
  ],
};

export const DEFAULT_SEARCH_FORM = {
  query: "",
  dataset: "quora",
  model: "agent",
  topK: 10,
  bm25K1: 1.5,
  bm25B: 0.75,
  candidateCount: 1000,
  rrfK: 60,
  tfidfWeight: 1.0,
  bm25Weight: 1.0,
  embeddingWeight: 1.0,
  useQueryRefinement: false,
  feedbackDocs: 3,
  expansionTerms: 5,
  snippetLength: 500,
  includeRawText: false,
  usePersonalization: false,
  maxPersonalizationTerms: 3,
};
