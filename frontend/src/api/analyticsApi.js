import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000/api";

export async function fetchEvaluationAnalytics() {
  const response = await axios.get(`${API_BASE_URL}/analytics/evaluation/`);
  return response.data;
}

export async function fetchEvaluationFiles() {
  const response = await axios.get(
    `${API_BASE_URL}/analytics/evaluation/files/`,
  );

  return response.data;
}

export async function fetchClusteringAnalytics(dataset) {
  const response = await axios.get(
    `${API_BASE_URL}/analytics/clustering/${encodeURIComponent(dataset)}/`,
  );

  return response.data;
}

export async function fetchTopicDetectionAnalytics(dataset) {
  const response = await axios.get(
    `${API_BASE_URL}/analytics/topics/${encodeURIComponent(dataset)}/`,
  );

  return response.data;
}
