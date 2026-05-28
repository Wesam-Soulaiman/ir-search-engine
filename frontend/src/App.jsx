import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [dataset, setDataset] = useState("dataset1");
  const [model, setModel] = useState("bm25");
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState([]);
  const [responseInfo, setResponseInfo] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) {
      alert("Please enter a search query.");
      return;
    }

    setLoading(true);
    setResults([]);
    setResponseInfo(null);

    try {
      const response = await axios.post("http://127.0.0.1:8000/api/search/", {
        query: query,
        dataset: dataset,
        model: model,
        top_k: Number(topK),
      });

      setResults(response.data.results || []);
      setResponseInfo({
        query: response.data.query,
        dataset: response.data.dataset,
        model: response.data.model,
        top_k: response.data.top_k,
      });
    } catch (error) {
      console.error(error);
      alert("Error while connecting to Django API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="container">
        <h1>Information Retrieval Search Engine</h1>
        <p className="subtitle">
          React frontend connected to Django backend API
        </p>

        <div className="search-card">
          <label>Search Query</label>
          <input
            type="text"
            placeholder="Example: machine learning"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />

          <div className="grid">
            <div>
              <label>Dataset</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
              >
                <option value="dataset1">Dataset 1</option>
                <option value="dataset2">Dataset 2</option>
              </select>
            </div>

            <div>
              <label>Retrieval Model</label>
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                <option value="tfidf">TF-IDF</option>
                <option value="bm25">BM25</option>
                <option value="embedding">Embedding</option>
                <option value="hybrid_serial">Hybrid Serial</option>
                <option value="hybrid_parallel">Hybrid Parallel</option>
              </select>
            </div>

            <div>
              <label>Top K</label>
              <input
                type="number"
                min="1"
                max="100"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
              />
            </div>
          </div>

          <button onClick={handleSearch} disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        {responseInfo && (
          <div className="info">
            <strong>Query:</strong> {responseInfo.query} |{" "}
            <strong>Dataset:</strong> {responseInfo.dataset} |{" "}
            <strong>Model:</strong> {responseInfo.model} |{" "}
            <strong>Top K:</strong> {responseInfo.top_k}
          </div>
        )}

        <div className="results">
          {results.map((item) => (
            <div className="result-card" key={item.doc_id}>
              <div className="result-header">
                <span className="rank">#{item.rank}</span>
                <h3>{item.title}</h3>
              </div>
              <p>{item.snippet}</p>
              <div className="meta">
                <span>Doc ID: {item.doc_id}</span>
                <span>Score: {item.score}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
