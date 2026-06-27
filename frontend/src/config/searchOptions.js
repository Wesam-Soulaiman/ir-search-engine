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
    value: "distributed_bm25",
    label: "Distributed BM25",
    description: "Local sharded BM25 with RRF coordinator merging.",
    tag: "Distributed",
  },
  {
    value: "embedding",
    label: "Embedding",
    description: "Semantic search using FAISS vector store.",
    tag: "Semantic",
  },
  {
    value: "biomedical_embedding",
    label: "Biomedical PubMedBERT",
    description: "Biomedical semantic search for Clinical Trials.",
    tag: "Biomedical",
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
  {
    value: "ltr",
    label: "Learning to Rank (LTR)",
    description: "Trained reranker over lexical and semantic candidates.",
    tag: "Reranker",
  },
  {
    value: "rag",
    label: "RAG Answer",
    description: "Grounded offline answer synthesis over retrieved documents.",
    tag: "Answer",
  },
];

export const RAG_RETRIEVER_MODELS = [
  {
    value: "bm25",
    label: "BM25",
  },
  {
    value: "tfidf",
    label: "TF-IDF",
  },
  {
    value: "embedding",
    label: "Embedding",
  },
  {
    value: "hybrid_serial",
    label: "Hybrid Serial",
  },
  {
    value: "hybrid_parallel",
    label: "Hybrid Parallel",
  },
  {
    value: "ltr",
    label: "LTR",
  },
  {
    value: "distributed_bm25",
    label: "Distributed BM25",
  },
  {
    value: "biomedical_embedding",
    label: "Biomedical PubMedBERT",
    clinicalOnly: true,
  },
];

export const RAG_GENERATION_MODES = [
  {
    value: "extractive_offline",
    label: "Extractive offline",
  },
  {
    value: "local_llm",
    label: "Local LLM via Ollama",
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
  numShards: 4,
  shardTopK: 100,
  ltrCandidateModels: {
    bm25: true,
    tfidf: true,
    embedding: true,
  },
  includeBiomedical: false,
  ragRetrieverModel: "hybrid_serial",
  ragContextDocs: 5,
  ragAnswerSentences: 4,
  includeSources: true,
  ragGenerationMode: "extractive_offline",
  ragLlmProvider: "ollama",
  ragLlmModel: "llama3.2:3b",
  ragLlmBaseUrl: "http://localhost:11434",
  ragLlmTemperature: 0,
  ragLlmMaxTokens: 350,
  tfidfWeight: 1.0,
  bm25Weight: 1.0,
  embeddingWeight: 1.0,
  biomedicalWeight: 0,
  useSpellingCorrection: false,
  useQueryRefinement: false,
  feedbackDocs: 3,
  expansionTerms: 5,
  snippetLength: 500,
  includeRawText: false,
  usePersonalization: false,
  maxPersonalizationTerms: 3,
};
