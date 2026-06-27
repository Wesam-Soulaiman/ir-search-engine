import { useEffect, useMemo, useState } from "react";

import { fetchClusteringAnalytics } from "../../api/analyticsApi";
import ChartCard from "./ChartCard";
import MetricBarChart from "./MetricBarChart";

function formatClusterName(cluster) {
  return `C${cluster.cluster_id}`;
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function ClusteringCharts({
  dataset,
}) {
  const [data, setData] = useState(null);
  const [selectedClusterId, setSelectedClusterId] = useState("");
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let isCurrent = true;

    async function loadClusters() {
      setLoading(true);
      setErrorMessage("");

      try {
        const responseData = await fetchClusteringAnalytics(dataset);

        if (isCurrent) {
          setData(responseData);
          setSelectedClusterId(
            responseData.clusters?.[0]?.cluster_id?.toString() || "",
          );
        }
      } catch (error) {
        const apiMessage = error?.response?.data?.error;

        if (isCurrent) {
          setData(null);
          setSelectedClusterId("");
          setErrorMessage(
            apiMessage
            || "No clustering data available. Build or run clustering first.",
          );
        }
      } finally {
        if (isCurrent) {
          setLoading(false);
        }
      }
    }

    loadClusters();

    return () => {
      isCurrent = false;
    };
  }, [dataset]);

  const clusters = useMemo(
    () => data?.clusters || [],
    [data],
  );
  const selectedCluster = useMemo(
    () => clusters.find(
      (cluster) => cluster.cluster_id?.toString() === selectedClusterId,
    ) || clusters[0],
    [clusters, selectedClusterId],
  );

  const clusterSizeData = clusters.map((cluster) => ({
    name: formatClusterName(cluster),
    Documents: cluster.size,
    label: cluster.label,
  }));

  const topTermData = (selectedCluster?.top_terms || []).map((term) => ({
    name: term.term,
    Weight: term.weight,
  }));

  if (loading) {
    return (
      <div className="analytics-empty">
        Loading clustering analytics...
      </div>
    );
  }

  if (errorMessage) {
    return (
      <div className="analytics-empty danger">
        {errorMessage}
      </div>
    );
  }

  if (!clusters.length) {
    return (
      <div className="analytics-empty">
        No clustering data available. Build or run clustering first.
      </div>
    );
  }

  return (
    <div className="analytics-stack">
      <div className="analytics-meta-row">
        <span>{data.dataset}</span>
        <span>{data.num_clusters} clusters</span>
      </div>

      <div className="interpretation-panel">
        <strong>How to read clustering</strong>
        <p>
          Document clustering groups similar documents together using the
          prepared feature space. The size chart shows how documents are
          distributed, while top terms and examples help interpret each group.
        </p>
      </div>

      <div className="analytics-grid">
        <ChartCard
          title="Cluster Size Distribution"
          subtitle="Document count per cluster"
        >
          <MetricBarChart
            data={clusterSizeData}
            bars={[
              {
                key: "Documents",
                name: "Documents",
                color: "#38bdf8",
              },
            ]}
            emptyMessage="No clustering data available. Build or run clustering first."
          />
        </ChartCard>

        <ChartCard
          title="Top Terms per Selected Cluster"
          subtitle={selectedCluster?.label}
          actions={
            <label className="chart-select">
              <span>Cluster</span>
              <select
                value={selectedCluster?.cluster_id?.toString() || ""}
                onChange={(event) => setSelectedClusterId(event.target.value)}
              >
                {clusters.map((cluster) => (
                  <option
                    key={cluster.cluster_id}
                    value={cluster.cluster_id}
                  >
                    {formatClusterName(cluster)} - {formatNumber(cluster.size)}
                  </option>
                ))}
              </select>
            </label>
          }
        >
          <MetricBarChart
            data={topTermData}
            bars={[
              {
                key: "Weight",
                name: "Relative term weight",
                color: "#a78bfa",
              },
            ]}
            yDomain={[0, 1]}
            emptyMessage="No top terms are available for this cluster."
          />
        </ChartCard>
      </div>

      <ChartCard
        title="Cluster Examples Table"
        subtitle="Cluster labels, term summaries, and representative documents"
      >
        <div className="analytics-table-wrap">
          <table className="analytics-table">
            <thead>
              <tr>
                <th>Cluster ID</th>
                <th>Cluster Label</th>
                <th>Size</th>
                <th>Top Terms</th>
                <th>Example Documents</th>
              </tr>
            </thead>
            <tbody>
              {clusters.map((cluster) => (
                <tr key={cluster.cluster_id}>
                  <td>{formatClusterName(cluster)}</td>
                  <td>{cluster.label}</td>
                  <td>{formatNumber(cluster.size)}</td>
                  <td>
                    {(cluster.top_terms || []).length ? (
                      <div className="term-chip-list">
                        {cluster.top_terms.map((term) => (
                          <span key={`${cluster.cluster_id}-${term.term}`}>
                            {term.term}
                          </span>
                        ))}
                      </div>
                    ) : (
                      "N/A"
                    )}
                  </td>
                  <td>
                    {(cluster.examples || [])
                      .map((example) => (
                        `${example.title || "Document"} (${example.doc_id})`
                      ))
                      .join("; ") || "N/A"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartCard>
    </div>
  );
}

export default ClusteringCharts;
