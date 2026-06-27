import { DATASETS } from "../config/searchOptions";
import ClusteringCharts from "./charts/ClusteringCharts";
import EvaluationCharts from "./charts/EvaluationCharts";
import TopicDetectionCharts from "./charts/TopicDetectionCharts";
import Badge from "./ui/Badge";
import PageHeader from "./ui/PageHeader";

const ANALYTICS_TABS = [
  {
    value: "evaluation",
    label: "Evaluation Dashboard",
  },
  {
    value: "clustering",
    label: "Clustering Analytics",
  },
  {
    value: "topics",
    label: "Topic Detection Analytics",
  },
];

function AnalyticsDashboard({
  dataset,
  activeTab,
  onDatasetChange,
  onTabChange,
}) {
  const activeLabel = ANALYTICS_TABS.find(
    (tab) => tab.value === activeTab,
  )?.label;

  return (
    <section className="analytics-section" id="analytics">
      <PageHeader
        eyebrow="Analytics"
        title={activeLabel}
        description="Visualize evaluation metrics, clustering structure, and detected topics using backend report data and Recharts."
        meta={(
          <>
            <Badge tone="info">Recharts</Badge>
            <Badge tone="success">CSV-backed</Badge>
            <Badge>{dataset}</Badge>
          </>
        )}
        actions={(
          <label className="analytics-dataset-select">
            <span>Dataset</span>
            <select
              value={dataset}
              onChange={(event) => onDatasetChange(event.target.value)}
            >
              {DATASETS.map((datasetOption) => (
                <option
                  key={datasetOption.value}
                  value={datasetOption.value}
                >
                  {datasetOption.label}
                </option>
              ))}
            </select>
          </label>
        )}
      />

      <div className="analytics-tabs" role="tablist">
        {ANALYTICS_TABS.map((tab) => (
          <button
            key={tab.value}
            type="button"
            className={activeTab === tab.value ? "active" : ""}
            onClick={() => onTabChange(tab.value)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "evaluation" ? (
        <EvaluationCharts dataset={dataset} />
      ) : null}

      {activeTab === "clustering" ? (
        <ClusteringCharts dataset={dataset} />
      ) : null}

      {activeTab === "topics" ? (
        <TopicDetectionCharts dataset={dataset} />
      ) : null}
    </section>
  );
}

export default AnalyticsDashboard;
