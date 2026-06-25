import { DATASETS, MODELS } from "../config/searchOptions";

function ControlPanel({
  form,
  onFieldChange,
}) {
  const selectedDataset = DATASETS.find(
    (dataset) => dataset.value === form.dataset
  );

  const selectedModel = MODELS.find(
    (model) => model.value === form.model
  );

  return (
    <section className="control-panel" id="controls">
      <div className="panel-header">
        <div>
          <span className="section-kicker">Configuration</span>
          <h2>Retrieval setup</h2>
        </div>

        <div className="active-pill">
          {selectedModel?.tag || "Model"}
        </div>
      </div>

      <div className="control-layout">
        <label className="input-group">
          <span>Dataset</span>
          <select
            value={form.dataset}
            onChange={(event) => onFieldChange("dataset", event.target.value)}
          >
            {DATASETS.map((dataset) => (
              <option key={dataset.value} value={dataset.value}>
                {dataset.label}
              </option>
            ))}
          </select>
          <small>{selectedDataset?.description}</small>
        </label>

        <label className="input-group">
          <span>Model</span>
          <select
            value={form.model}
            onChange={(event) => onFieldChange("model", event.target.value)}
          >
            {MODELS.map((model) => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
          <small>{selectedModel?.description}</small>
        </label>

        <label className="input-group compact">
          <span>Top K</span>
          <input
            type="number"
            min="1"
            max="1000"
            value={form.topK}
            onChange={(event) => onFieldChange("topK", event.target.value)}
          />
          <small>Returned documents</small>
        </label>
      </div>
    </section>
  );
}

export default ControlPanel;
