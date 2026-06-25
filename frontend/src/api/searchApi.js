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
    use_query_refinement: Boolean(form.useQueryRefinement),
    feedback_docs: Number(form.feedbackDocs),
    expansion_terms: Number(form.expansionTerms),
    snippet_length: Number(form.snippetLength),
    include_raw_text: Boolean(form.includeRawText),
    use_personalization: Boolean(form.usePersonalization),
    max_personalization_terms: Number(form.maxPersonalizationTerms),
  };

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
