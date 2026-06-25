function AdvancedOptions({
  form,
  onFieldChange,
}) {
  const showHybridOptions = (
    form.model === "hybrid_serial"
    || form.model === "hybrid_parallel"
    || form.model === "agent"
  );

  return (
    <details className="advanced-panel">
      <summary>
        Advanced options
        <span>BM25, hybrid fusion, query refinement, and raw text display</span>
      </summary>

      <div className="advanced-grid">
        <label className="field">
          <span>BM25 k1</span>
          <input
            type="number"
            min="0.000001"
            max="100"
            step="0.1"
            value={form.bm25K1}
            onChange={(event) => onFieldChange("bm25K1", event.target.value)}
          />
          <small>Term-frequency saturation</small>
        </label>

        <label className="field">
          <span>BM25 b</span>
          <input
            type="number"
            min="0"
            max="1"
            step="0.05"
            value={form.bm25B}
            onChange={(event) => onFieldChange("bm25B", event.target.value)}
          />
          <small>Document-length normalization</small>
        </label>

        {showHybridOptions ? (
          <label className="field">
            <span>Candidate Count</span>
            <input
              type="number"
              min="1"
              max="10000"
              value={form.candidateCount}
              onChange={(event) => onFieldChange("candidateCount", event.target.value)}
            />
            <small>Used by hybrid models</small>
          </label>
        ) : null}

        {(form.model === "hybrid_parallel" || form.model === "agent") ? (
          <label className="field">
            <span>RRF K</span>
            <input
              type="number"
              min="1"
              max="100000"
              value={form.rrfK}
              onChange={(event) => onFieldChange("rrfK", event.target.value)}
            />
            <small>Fusion smoothing parameter</small>
          </label>
        ) : null}

        <label className="field">
          <span>Snippet Length</span>
          <input
            type="number"
            min="1"
            max="5000"
            value={form.snippetLength}
            onChange={(event) => onFieldChange("snippetLength", event.target.value)}
          />
          <small>Characters shown per result</small>
        </label>
      </div>

      <div className="toggle-grid">
        <label className="toggle-row">
          <input
            type="checkbox"
            checked={form.useQueryRefinement}
            onChange={(event) => onFieldChange("useQueryRefinement", event.target.checked)}
          />
          <span>
            <strong>Query Refinement</strong>
            <small>Pseudo-relevance feedback expansion</small>
          </span>
        </label>

        <label className="toggle-row">
          <input
            type="checkbox"
            checked={form.includeRawText}
            onChange={(event) => onFieldChange("includeRawText", event.target.checked)}
          />
          <span>
            <strong>Include Raw Text</strong>
            <small>Show original stored document text when top_k ≤ 50</small>
          </span>
        </label>
      </div>

      {form.useQueryRefinement ? (
        <div className="advanced-grid refinement-grid">
          <label className="field">
            <span>Feedback Docs</span>
            <input
              type="number"
              min="1"
              max="100"
              value={form.feedbackDocs}
              onChange={(event) => onFieldChange("feedbackDocs", event.target.value)}
            />
          </label>

          <label className="field">
            <span>Expansion Terms</span>
            <input
              type="number"
              min="1"
              max="100"
              value={form.expansionTerms}
              onChange={(event) => onFieldChange("expansionTerms", event.target.value)}
            />
          </label>
        </div>
      ) : null}
    </details>
  );
}

export default AdvancedOptions;
