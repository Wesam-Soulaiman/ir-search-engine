import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000/api";
const SESSION_STORAGE_KEY = "ir_search_session_id";

let fallbackSessionId = "";

function createSessionId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }

  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function getAnonymousSessionId() {
  if (fallbackSessionId) {
    return fallbackSessionId;
  }

  try {
    const storedSessionId = window.localStorage.getItem(SESSION_STORAGE_KEY);

    if (storedSessionId) {
      fallbackSessionId = storedSessionId;
      return storedSessionId;
    }

    const newSessionId = createSessionId();
    window.localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);
    fallbackSessionId = newSessionId;

    return newSessionId;
  } catch {
    fallbackSessionId = createSessionId();
    return fallbackSessionId;
  }
}

export async function searchDocuments(form) {
  const payload = {
    query: form.query,
    user_id: getAnonymousSessionId(),
    dataset: form.dataset,
    model: form.model,
    top_k: Number(form.topK),
    bm25_k1: Number(form.bm25K1),
    bm25_b: Number(form.bm25B),
    candidate_count: Number(form.candidateCount),
    rrf_k: Number(form.rrfK),
    tfidf_weight: Number(form.tfidfWeight),
    bm25_weight: Number(form.bm25Weight),
    embedding_weight: Number(form.embeddingWeight),
    biomedical_weight: Number(form.biomedicalWeight),
    use_spelling_correction: Boolean(form.useSpellingCorrection),
    use_query_refinement: Boolean(form.useQueryRefinement),
    feedback_docs: Number(form.feedbackDocs),
    expansion_terms: Number(form.expansionTerms),
    snippet_length: Number(form.snippetLength),
    include_raw_text: Boolean(form.includeRawText),
    use_personalization: Boolean(form.usePersonalization),
    max_personalization_terms: Number(form.maxPersonalizationTerms),
  };

  if (form.model === "distributed_bm25") {
    payload.num_shards = Number(form.numShards);
    payload.shard_top_k = Number(form.shardTopK);
  }

  if (form.model === "ltr") {
    payload.ltr_candidate_models = Object.entries(form.ltrCandidateModels || {})
      .filter(([, enabled]) => Boolean(enabled))
      .map(([modelName]) => modelName);
    payload.include_biomedical = Boolean(form.includeBiomedical);
  }

  if (form.model === "rag") {
    payload.rag_retriever_model = form.ragRetrieverModel;
    payload.rag_context_docs = Number(form.ragContextDocs);
    payload.rag_answer_sentences = Number(form.ragAnswerSentences);
    payload.include_sources = Boolean(form.includeSources);
    payload.rag_generation_mode = form.ragGenerationMode;
    payload.rag_llm_provider = form.ragLlmProvider;
    payload.rag_llm_model = form.ragLlmModel;
    payload.rag_llm_base_url = form.ragLlmBaseUrl;
    payload.rag_llm_temperature = Number(form.ragLlmTemperature);
    payload.rag_llm_max_tokens = Number(form.ragLlmMaxTokens);
  }

  const response = await axios.post(`${API_BASE_URL}/search/`, payload);
  return response.data;
}

export async function fetchRawDocument(dataset, docId) {
  const response = await axios.get(
    `${API_BASE_URL}/documents/${encodeURIComponent(dataset)}/${encodeURIComponent(docId)}/`,
  );

  return response.data;
}

export async function fetchClusterSummary(dataset) {
  const response = await axios.get(
    `${API_BASE_URL}/clusters/${encodeURIComponent(dataset)}/`,
  );

  return response.data;
}

export async function fetchClusterTopics(dataset) {
  const response = await axios.get(
    `${API_BASE_URL}/topics/${encodeURIComponent(dataset)}/`,
  );

  return response.data;
}
