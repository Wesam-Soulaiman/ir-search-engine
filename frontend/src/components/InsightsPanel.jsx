import { useMemo, useState } from "react";

import {
  fetchClusterSummary,
  fetchClusterTopics,
} from "../api/searchApi";

function toNumber(value) {
  const numberValue = Number(value);
  return Number.isNaN(numberValue) ? 0 : numberValue;
}

function formatNumber(value) {
  return toNumber(value).toLocaleString();
}

function getTopicLabel(row) {
  return row.topic_label || `Cluster ${row.cluster_id}`;
}

function getRepresentativeDoc(row) {
  return row.representative_doc_id || row.representative_doc_ids || "N/A";
}

function InsightsPanel({ dataset }) {
  const [activeTab, setActiveTab] = useState("topics");
  const [topics, setTopics] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const rows = activeTab === "topics" ? topics : clusters;

  const sortedRows = useMemo(
    () =>
      [...rows].sort(
        (firstRow, secondRow) =>
          toNumber(secondRow.document_count) - toNumber(firstRow.document_count),
      ),
    [rows],
  );

  const chartRows = sortedRows.slice(0, 10);
  const listRows = sortedRows.slice(0, 8);
  const maxDocumentCount = Math.max(
    1,
    ...chartRows.map((row) => toNumber(row.document_count)),
  );

  const loadInsights = async (tab = activeTab) => {
    setLoading(true);
    setErrorMessage("");

    try {
      if (tab === "topics") {
        const data = await fetchClusterTopics(dataset);
        setTopics(data.topics || []);
        setMetadata({
          dataset: data.dataset,
          feature: data.feature,
          count: data.topic_count,
        });
      } else {
        const data = await fetchClusterSummary(dataset);
        setClusters(data.clusters || []);
        setMetadata({
          dataset: data.dataset,
          feature: data.feature,
          count: data.cluster_count,
        });
      }
    } catch (error) {
      const apiError = error?.response?.data?.error;
      setErrorMessage(apiError || "Could not load clustering insights.");
    } finally {
      setLoading(false);
    }
  };

  const switchTab = (tab) => {
    setActiveTab(tab);
    loadInsights(tab);
  };

  return (
    <section className="insights-panel">
      <div className="insights-header">
        <div>
          <span className="section-kicker">Additional features</span>
          <h2>Clusters & Topics</h2>
        </div>

        <button
          type="button"
          onClick={() => loadInsights(activeTab)}
          disabled={loading}
        >
          {loading ? "Loading..." : "Load"}
        </button>
      </div>

      <div className="insights-tabs">
        <button
          type="button"
          className={activeTab === "topics" ? "active" : ""}
          onClick={() => switchTab("topics")}
        >
          Topic Detection
        </button>

        <button
          type="button"
          className={activeTab === "clusters" ? "active" : ""}
          onClick={() => switchTab("clusters")}
        >
          Document Clustering
        </button>
      </div>

      {metadata ? (
        <div className="insights-meta">
          <span>{metadata.dataset}</span>
          <span>{metadata.feature}</span>
          <span>{metadata.count} rows</span>
        </div>
      ) : (
        <p className="insights-hint">
          Load cluster insights for the selected dataset.
        </p>
      )}

      {errorMessage ? <div className="insights-error">{errorMessage}</div> : null}

      {chartRows.length ? (
        <div className="cluster-chart-card">
          <div className="cluster-chart-header">
            <strong>
              {activeTab === "topics"
                ? "Topic size chart"
                : "Cluster distribution chart"}
            </strong>
            <small>Top 10 by document count</small>
          </div>

          <div className="cluster-chart">
            {chartRows.map((row) => {
              const documentCount = toNumber(row.document_count);
              const width = Math.max(
                6,
                (documentCount / maxDocumentCount) * 100,
              );

              return (
                <div
                  className="chart-row"
                  key={`chart-${activeTab}-${row.cluster_id}`}
                >
                  <span className="chart-label">C{row.cluster_id}</span>

                  <div className="chart-track">
                    <div
                      className="chart-bar"
                      style={{
                        width: `${width}%`,
                      }}
                    />
                  </div>

                  <span className="chart-value">
                    {formatNumber(documentCount)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

      <div className="insights-list">
        {listRows.map((row) => {
          const clusterId = row.cluster_id;
          const count = row.document_count;
          const label = getTopicLabel(row);
          const terms = row.top_terms || "";
          const representativeDoc = getRepresentativeDoc(row);

          return (
            <article className="insight-row" key={`${activeTab}-${clusterId}`}>
              <div className="cluster-chip">C{clusterId}</div>

              <div>
                <h3>{label}</h3>

                {terms ? <p>{terms}</p> : null}

                <div className="insight-row-meta">
                  <span>{formatNumber(count)} documents</span>
                  <span>Representative: {representativeDoc}</span>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

export default InsightsPanel;
