import { DATASETS } from "../config/searchOptions";
import ClusteringCharts from "./charts/ClusteringCharts";
import EvaluationCharts from "./charts/EvaluationCharts";
import TopicDetectionCharts from "./charts/TopicDetectionCharts";

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
  return (
    <section className="analytics-section" id="analytics">
      <div className="analytics-header">
        <div>
          <span className="section-kicker">Analytics</span>
          <h2>
            {ANALYTICS_TABS.find((tab) => tab.value === activeTab)?.label}
          </h2>
        </div>

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
      </div>

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
