import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [dataset, setDataset] = useState("sample_dataset");
  const [model, setModel] = useState("tfidf");
  const [topK, setTopK] = useState(10);
  const [bm25K1, setBm25K1] = useState(1.5);
  const [bm25B, setBm25B] = useState(0.75);
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
        bm25_k1: Number(bm25K1),
        bm25_b: Number(bm25B),
      });

      setResults(response.data.results || []);
      setResponseInfo({
        query: response.data.query,
        dataset: response.data.dataset,
        model: response.data.model,
        top_k: response.data.top_k,
        bm25_k1: response.data.bm25_k1,
        bm25_b: response.data.bm25_b,
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
          React frontend connected to Django retrieval backend
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
                <option value="sample_dataset">Sample Dataset</option>
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

          {model === "bm25" && (
            <div className="bm25-panel">
              <h3>BM25 Parameters</h3>

              <div className="grid two">
                <div>
                  <label>k1: {bm25K1}</label>
                  <input
                    type="range"
                    min="0.5"
                    max="3.0"
                    step="0.1"
                    value={bm25K1}
                    onChange={(e) => setBm25K1(e.target.value)}
                  />
                </div>

                <div>
                  <label>b: {bm25B}</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={bm25B}
                    onChange={(e) => setBm25B(e.target.value)}
                  />
                </div>
              </div>

              <p className="hint">
                k1 controls term frequency saturation, while b controls document
                length normalization.
              </p>
            </div>
          )}

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
            {responseInfo.model === "bm25" && (
              <>
                {" | "}
                <strong>k1:</strong> {responseInfo.bm25_k1} |{" "}
                <strong>b:</strong> {responseInfo.bm25_b}
              </>
            )}
          </div>
        )}

        <div className="results">
          {results.map((item) => (
            <div className="result-card" key={`${item.doc_id}-${item.rank}`}>
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
