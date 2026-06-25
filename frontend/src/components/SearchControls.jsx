import { DATASETS, MODELS } from "../config/searchOptions";

function SearchControls({
  form,
  onFieldChange,
}) {
  const selectedModel = MODELS.find((model) => model.value === form.model);

  return (
    <section className="control-card" aria-label="Search controls">
      <div className="control-grid">
        <label className="field">
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
          <small>
            {DATASETS.find((dataset) => dataset.value === form.dataset)?.description}
          </small>
        </label>

        <label className="field">
          <span>Retrieval Model</span>
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

        <label className="field compact-field">
          <span>Top K</span>
          <input
            type="number"
            min="1"
            max="1000"
            value={form.topK}
            onChange={(event) => onFieldChange("topK", event.target.value)}
          />
          <small>Number of results</small>
        </label>
      </div>

      {selectedModel?.badge ? (
        <div className="model-badge-line">
          <span className="model-badge">{selectedModel.badge}</span>
          <span>Agent mode explains which model it executes.</span>
        </div>
      ) : null}
    </section>
  );
}

export default SearchControls;
