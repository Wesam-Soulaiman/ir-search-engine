import { useEffect, useMemo, useState } from "react";

import {
  fetchEvaluationAnalytics,
  fetchEvaluationFiles,
} from "../../api/analyticsApi";
import ChartCard from "./ChartCard";
import MetricBarChart from "./MetricBarChart";

const QUALITY_BARS = [
  {
    key: "MAP",
    name: "MAP",
    color: "#38bdf8",
  },
  {
    key: "Precision@10",
    name: "Precision@10",
    color: "#a78bfa",
  },
  {
    key: "Recall",
    name: "Recall",
    color: "#34d399",
  },
  {
    key: "nDCG",
    name: "nDCG",
    color: "#f59e0b",
  },
];

const SPEED_METRICS = [
  {
    key: "wall_time_seconds",
    label: "Wall time",
    chartName: "Wall time seconds",
  },
  {
    key: "average_latency_ms",
    label: "Latency",
    chartName: "Average latency ms",
  },
  {
    key: "qps",
    label: "QPS",
    chartName: "Queries per second",
  },
];

const EMPTY_SECTION = {
  label: "",
  description: "",
  rows: [],
  sources: [],
};

function formatModelName(value) {
  return String(value || "unknown")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatScenario(value) {
  return String(value || "run")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function toMetric(value) {
  return typeof value === "number" ? Number(value.toFixed(6)) : null;
}

function hasNumber(row, key) {
  return typeof row?.[key] === "number";
}

function formatCell(value, digits = 3) {
  if (typeof value !== "number") {
    return "N/A";
  }

  return value.toFixed(digits);
}

function getSection(dashboard, key) {
  return dashboard?.sections?.[key] || {
    ...EMPTY_SECTION,
    key,
  };
}

function filterRowsForDataset(section, dataset) {
  return (section.rows || []).filter((row) => row.dataset === dataset);
}

function getComparisonLabel(row) {
  return `${formatModelName(row.model)} ${formatScenario(row.scenario)}`;
}

function SourceCsvList({
  sources,
}) {
  if (!sources?.length) {
    return (
      <div className="source-csv-list muted">
        No source CSV files selected for this section.
      </div>
    );
  }

  return (
    <div className="source-csv-list" aria-label="Source CSV files">
      {sources.map((source) => (
        <span key={source.relative_path || source.name}>
          {source.relative_path || source.name}
        </span>
      ))}
    </div>
  );
}

function AvailableCsvFilesPanel({
  inventory,
  errorMessage,
}) {
  const files = inventory?.files || [];

  return (
    <ChartCard
      title="Available CSV Files"
      subtitle="Inspection only; files listed here are not loaded into charts unless selected in report_manifest.json"
    >
      {errorMessage ? (
        <div className="analytics-empty danger">
          {errorMessage}
        </div>
      ) : null}

      {!errorMessage && !files.length ? (
        <div className="analytics-empty">
          No evaluation CSV files found.
        </div>
      ) : null}

      {files.length ? (
        <div className="analytics-table-wrap">
          <table className="analytics-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Relative Path</th>
                <th>Rows</th>
                <th>Size</th>
                <th>Modified</th>
                <th>Detected Columns</th>
              </tr>
            </thead>
            <tbody>
              {files.map((file) => (
                <tr key={file.relative_path}>
                  <td>{file.name}</td>
                  <td>{file.relative_path}</td>
                  <td>{file.row_count}</td>
                  <td>{Number(file.size || 0).toLocaleString()} bytes</td>
                  <td>{file.modified_at}</td>
                  <td>
                    {(file.detected_columns || []).join(", ") || "N/A"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </ChartCard>
  );
}

function EvaluationCharts({
  dataset,
}) {
  const [dashboard, setDashboard] = useState(null);
  const [fileInventory, setFileInventory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");
  const [filesErrorMessage, setFilesErrorMessage] = useState("");
  const [speedMetric, setSpeedMetric] = useState("wall_time_seconds");

  useEffect(() => {
    let isCurrent = true;

    async function loadDashboard() {
      setLoading(true);
      setErrorMessage("");
      setFilesErrorMessage("");

      const dashboardPromise = fetchEvaluationAnalytics();
      const filesPromise = fetchEvaluationFiles();

      try {
        const [dashboardData, filesData] = await Promise.all([
          dashboardPromise,
          filesPromise,
        ]);

        if (isCurrent) {
          setDashboard(dashboardData);
          setFileInventory(filesData);
        }
      } catch (error) {
        const apiMessage = error?.response?.data?.error;

        if (isCurrent) {
          setErrorMessage(
            apiMessage || "Could not load evaluation analytics.",
          );
          setDashboard(null);
        }

        try {
          const filesData = await filesPromise;

          if (isCurrent) {
            setFileInventory(filesData);
          }
        } catch (filesError) {
          const filesApiMessage = filesError?.response?.data?.error;

          if (isCurrent) {
            setFilesErrorMessage(
              filesApiMessage || "Could not inspect evaluation CSV files.",
            );
          }
        }
      } finally {
        if (isCurrent) {
          setLoading(false);
        }
      }
    }

    loadDashboard();

    return () => {
      isCurrent = false;
    };
  }, []);

  const modelSection = getSection(dashboard, "model_comparison");
  const refinementSection = getSection(
    dashboard,
    "query_refinement_before_after",
  );
  const runtimeSection = getSection(dashboard, "runtime_comparison");
  const extraFeaturesSection = getSection(dashboard, "extra_features");

  const modelRows = useMemo(
    () => filterRowsForDataset(modelSection, dataset),
    [modelSection, dataset],
  );
  const refinementRows = useMemo(
    () => filterRowsForDataset(refinementSection, dataset),
    [refinementSection, dataset],
  );
  const runtimeRows = useMemo(
    () => filterRowsForDataset(runtimeSection, dataset),
    [runtimeSection, dataset],
  );
  const extraFeatureRows = useMemo(
    () => filterRowsForDataset(extraFeaturesSection, dataset),
    [extraFeaturesSection, dataset],
  );

  const availableSpeedMetrics = useMemo(
    () => SPEED_METRICS.filter(
      (metric) => runtimeRows.some((row) => hasNumber(row, metric.key)),
    ),
    [runtimeRows],
  );

  const effectiveSpeedMetric = availableSpeedMetrics.some(
    (metric) => metric.key === speedMetric,
  )
    ? speedMetric
    : availableSpeedMetrics[0]?.key || speedMetric;

  const selectedSpeedMetric = SPEED_METRICS.find(
    (metric) => metric.key === effectiveSpeedMetric,
  ) || SPEED_METRICS[0];

  const qualityChartData = modelRows.map((row) => ({
    name: formatModelName(row.model),
    MAP: toMetric(row.map),
    "Precision@10": toMetric(row.precision_at_10),
    Recall: toMetric(row.recall),
    nDCG: toMetric(row.ndcg),
  }));

  const beforeAfterChartData = refinementRows.map((row) => ({
    name: getComparisonLabel(row),
    MAP: toMetric(row.map),
    "Precision@10": toMetric(row.precision_at_10),
    Recall: toMetric(row.recall),
    nDCG: toMetric(row.ndcg),
  }));

  const speedChartData = runtimeRows
    .filter((row) => hasNumber(row, selectedSpeedMetric.key))
    .map((row) => ({
      name: `${formatModelName(row.model)} ${formatScenario(row.scenario)}`,
      [selectedSpeedMetric.chartName]: toMetric(row[selectedSpeedMetric.key]),
    }));

  const selectedRows = [
    ...modelRows,
    ...refinementRows,
    ...runtimeRows,
    ...extraFeatureRows,
  ];

  if (loading) {
    return (
      <div className="analytics-empty">
        Loading evaluation dashboard...
      </div>
    );
  }

  if (errorMessage) {
    return (
      <div className="analytics-stack">
        <div className="analytics-empty danger">
          {errorMessage}
        </div>

        <AvailableCsvFilesPanel
          inventory={fileInventory}
          errorMessage={filesErrorMessage}
        />
      </div>
    );
  }

  if (!dashboard?.manifest_found) {
    return (
      <div className="analytics-stack">
        <div className="analytics-empty">
          {dashboard?.message
            || "No report manifest found. Create reports/evaluation/report_manifest.json to select final report CSV files."}
        </div>

        <AvailableCsvFilesPanel
          inventory={fileInventory}
          errorMessage={filesErrorMessage}
        />
      </div>
    );
  }

  return (
    <div className="analytics-stack">
      <div className="analytics-meta-row">
        <span>{dashboard.file_count || 0} manifest CSV files</span>
        <span>{selectedRows.length} selected rows for {dataset}</span>
        <span>{dashboard.manifest_path}</span>
      </div>

      {dashboard.errors?.length ? (
        <div className="analytics-empty danger">
          {dashboard.errors.length} manifest issue(s) were found. Check
          report_manifest.json and the Available CSV Files panel.
        </div>
      ) : null}

      <div className="metric-explainer-grid">
        <article>
          <strong>MAP</strong>
          <p>Mean Average Precision summarizes ranking quality across queries.</p>
        </article>
        <article>
          <strong>Precision@10</strong>
          <p>How many of the top ten retrieved documents are relevant.</p>
        </article>
        <article>
          <strong>Recall</strong>
          <p>How much of the known relevant set is recovered at the reported cutoff.</p>
        </article>
        <article>
          <strong>nDCG</strong>
          <p>Rewards relevant documents appearing higher in the ranked list.</p>
        </article>
        <article>
          <strong>Runtime / QPS</strong>
          <p>Shows evaluation wall time, average latency, or throughput when available.</p>
        </article>
      </div>

      <div className="analytics-grid">
        <ChartCard
          title={modelSection.label || "Model Quality Comparison"}
          subtitle={modelSection.description}
        >
          <SourceCsvList sources={modelSection.sources} />
          <MetricBarChart
            data={qualityChartData}
            bars={QUALITY_BARS}
            yDomain={[0, 1]}
            emptyMessage="No manifest-selected model comparison rows are available for this dataset."
          />
        </ChartCard>

        <ChartCard
          title={refinementSection.label || "Query Refinement Before/After"}
          subtitle={refinementSection.description}
        >
          <SourceCsvList sources={refinementSection.sources} />
          <MetricBarChart
            data={beforeAfterChartData}
            bars={QUALITY_BARS}
            yDomain={[0, 1]}
            emptyMessage="No manifest-selected before/after rows are available for this dataset."
          />
        </ChartCard>
      </div>

      <ChartCard
        title={runtimeSection.label || "Runtime Comparison"}
        subtitle={runtimeSection.description}
        actions={
          availableSpeedMetrics.length ? (
            <div className="segmented-control">
              {availableSpeedMetrics.map((metric) => (
                <button
                  key={metric.key}
                  type="button"
                  className={effectiveSpeedMetric === metric.key ? "active" : ""}
                  onClick={() => setSpeedMetric(metric.key)}
                >
                  {metric.label}
                </button>
              ))}
            </div>
          ) : null
        }
      >
        <SourceCsvList sources={runtimeSection.sources} />
        <MetricBarChart
          data={speedChartData}
          bars={[
            {
              key: selectedSpeedMetric.chartName,
              name: selectedSpeedMetric.chartName,
              color: "#22c55e",
            },
          ]}
          emptyMessage="No manifest-selected runtime rows are available for this dataset."
        />
      </ChartCard>

      <ChartCard
        title={extraFeaturesSection.label || "Extra Features"}
        subtitle={extraFeaturesSection.description}
      >
        <SourceCsvList sources={extraFeaturesSection.sources} />
        <EvaluationRowsTable rows={extraFeatureRows} showSection={false} />
      </ChartCard>

      <ChartCard
        title="Evaluation Table"
        subtitle="Only rows selected by report_manifest.json are shown here"
      >
        <EvaluationRowsTable rows={selectedRows} showSection />
      </ChartCard>

      <AvailableCsvFilesPanel
        inventory={fileInventory}
        errorMessage={filesErrorMessage}
      />
    </div>
  );
}

function EvaluationRowsTable({
  rows,
  showSection,
}) {
  if (!rows.length) {
    return (
      <div className="analytics-empty">
        No manifest-selected rows are available for this dataset.
      </div>
    );
  }

  return (
    <div className="analytics-table-wrap">
      <table className="analytics-table">
        <thead>
          <tr>
            {showSection ? <th>Section</th> : null}
            <th>Dataset</th>
            <th>Model</th>
            <th>Scenario</th>
            <th>MAP</th>
            <th>Precision@10</th>
            <th>Recall</th>
            <th>nDCG</th>
            <th>Time/Latency</th>
            <th>Source CSV</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.section_key}-${row.source_csv}-${row.model}-${index}`}>
              {showSection ? <td>{formatScenario(row.section_key)}</td> : null}
              <td>{row.dataset}</td>
              <td>{formatModelName(row.model)}</td>
              <td>{formatScenario(row.scenario)}</td>
              <td>{formatCell(row.map)}</td>
              <td>{formatCell(row.precision_at_10)}</td>
              <td>{formatCell(row.recall)}</td>
              <td>{formatCell(row.ndcg)}</td>
              <td>
                {typeof row.wall_time_seconds === "number"
                  ? `${formatCell(row.wall_time_seconds, 2)} s`
                  : null}
                {typeof row.wall_time_seconds !== "number"
                  && typeof row.average_latency_ms === "number"
                  ? `${formatCell(row.average_latency_ms, 2)} ms`
                  : null}
                {typeof row.wall_time_seconds !== "number"
                  && typeof row.average_latency_ms !== "number"
                  && typeof row.qps === "number"
                  ? `${formatCell(row.qps, 2)} qps`
                  : null}
                {typeof row.wall_time_seconds !== "number"
                  && typeof row.average_latency_ms !== "number"
                  && typeof row.qps !== "number"
                  ? "N/A"
                  : null}
              </td>
              <td>{row.source_csv}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default EvaluationCharts;
