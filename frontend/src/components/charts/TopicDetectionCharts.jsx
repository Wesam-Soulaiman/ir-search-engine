import { useEffect, useMemo, useState } from "react";

import { fetchTopicDetectionAnalytics } from "../../api/analyticsApi";
import ChartCard from "./ChartCard";
import MetricBarChart from "./MetricBarChart";

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function TopicDetectionCharts({
  dataset,
}) {
  const [data, setData] = useState(null);
  const [selectedTopicId, setSelectedTopicId] = useState("");
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let isCurrent = true;

    async function loadTopics() {
      setLoading(true);
      setErrorMessage("");

      try {
        const responseData = await fetchTopicDetectionAnalytics(dataset);

        if (isCurrent) {
          setData(responseData);
          setSelectedTopicId(
            responseData.topics?.[0]?.cluster_id?.toString() || "",
          );
        }
      } catch (error) {
        const apiMessage = error?.response?.data?.error;

        if (isCurrent) {
          setData(null);
          setSelectedTopicId("");
          setErrorMessage(apiMessage || "No topic detection data available.");
        }
      } finally {
        if (isCurrent) {
          setLoading(false);
        }
      }
    }

    loadTopics();

    return () => {
      isCurrent = false;
    };
  }, [dataset]);

  const topics = useMemo(
    () => data?.topics || [],
    [data],
  );
  const selectedTopicRow = useMemo(
    () => topics.find(
      (topic) => topic.cluster_id?.toString() === selectedTopicId,
    ) || topics[0],
    [selectedTopicId, topics],
  );

  const topicFrequencyData = topics.map((topic) => ({
    name: topic.topic,
    Count: topic.count,
  }));

  const topTermData = (selectedTopicRow?.top_terms || []).map((term) => ({
    name: term.term,
    Weight: term.weight,
  }));

  if (loading) {
    return (
      <div className="analytics-empty">
        Loading topic detection analytics...
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

  if (!topics.length) {
    return (
      <div className="analytics-empty">
        No topic detection data available.
      </div>
    );
  }

  return (
    <div className="analytics-stack">
      <div className="analytics-meta-row">
        <span>{data.dataset}</span>
        <span>{data.topic_count} topics</span>
      </div>

      <div className="interpretation-panel">
        <strong>How to read topic detection</strong>
        <p>
          Topic detection explains what the discovered groups are about. Topic
          frequency shows coverage, and the selected-topic term chart shows the
          representative words behind each label.
        </p>
      </div>

      <div className="analytics-grid">
        <ChartCard
          title="Topic Frequency Chart"
          subtitle="Document count per detected topic"
        >
          <MetricBarChart
            data={topicFrequencyData}
            bars={[
              {
                key: "Count",
                name: "Documents",
                color: "#38bdf8",
              },
            ]}
            emptyMessage="No topic detection data available."
          />
        </ChartCard>

        <ChartCard
          title="Top Topic Terms Chart"
          subtitle={selectedTopicRow?.topic}
          actions={
            <label className="chart-select">
              <span>Topic</span>
              <select
                value={selectedTopicRow?.cluster_id?.toString() || ""}
                onChange={(event) => setSelectedTopicId(event.target.value)}
              >
                {topics.map((topic) => (
                  <option
                    key={`${topic.cluster_id}-${topic.topic}`}
                    value={topic.cluster_id}
                  >
                    {topic.topic}
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
            emptyMessage="No top terms are available for this topic."
          />
        </ChartCard>
      </div>

      <ChartCard
        title="Topic Summary Table"
        subtitle="Topic counts, representative terms, and example documents"
      >
        <div className="analytics-table-wrap">
          <table className="analytics-table">
            <thead>
              <tr>
                <th>Topic</th>
                <th>Count</th>
                <th>Top Terms</th>
                <th>Example Documents</th>
              </tr>
            </thead>
            <tbody>
              {topics.map((topic) => (
                <tr key={`${topic.cluster_id}-${topic.topic}`}>
                  <td>{topic.topic}</td>
                  <td>{formatNumber(topic.count)}</td>
                  <td>
                    {(topic.top_terms || []).length ? (
                      <div className="term-chip-list">
                        {topic.top_terms.map((term) => (
                          <span key={`${topic.cluster_id}-${term.term}`}>
                            {term.term}
                          </span>
                        ))}
                      </div>
                    ) : (
                      "N/A"
                    )}
                  </td>
                  <td>
                    {(topic.examples || [])
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

export default TopicDetectionCharts;
