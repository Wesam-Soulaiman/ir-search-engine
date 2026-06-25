function AdvancedPanel({
  form,
  onFieldChange,
}) {
  const showHybrid = (
    form.model === "hybrid_serial"
    || form.model === "hybrid_parallel"
    || form.model === "agent"
  );

  const showParallelFusion = (
    form.model === "hybrid_parallel"
    || form.model === "agent"
  );

  return (
    <section className="advanced-card">
      <details>
        <summary>
          <span>
            Advanced retrieval parameters
          </span>
          <small>
            Fine-tune BM25, hybrid retrieval, snippets, and query refinement
          </small>
        </summary>

        <div className="advanced-body">
          <div className="advanced-grid">
            <label className="input-group">
              <span>BM25 k1</span>
              <input
                type="number"
                min="0.000001"
                max="100"
                step="0.1"
                value={form.bm25K1}
                onChange={(event) => onFieldChange(
                  "bm25K1",
                  event.target.value,
                )}
              />
            </label>

            <label className="input-group">
              <span>BM25 b</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={form.bm25B}
                onChange={(event) => onFieldChange(
                  "bm25B",
                  event.target.value,
                )}
              />
            </label>

            {showHybrid ? (
              <label className="input-group">
                <span>Candidate Count</span>
                <input
                  type="number"
                  min="1"
                  max="10000"
                  value={form.candidateCount}
                  onChange={(event) => onFieldChange(
                    "candidateCount",
                    event.target.value,
                  )}
                />
              </label>
            ) : null}

            {showParallelFusion ? (
              <label className="input-group">
                <span>RRF K</span>
                <input
                  type="number"
                  min="1"
                  max="100000"
                  value={form.rrfK}
                  onChange={(event) => onFieldChange(
                    "rrfK",
                    event.target.value,
                  )}
                />
              </label>
            ) : null}

            <label className="input-group">
              <span>Snippet Length</span>
              <input
                type="number"
                min="1"
                max="5000"
                value={form.snippetLength}
                onChange={(event) => onFieldChange(
                  "snippetLength",
                  event.target.value,
                )}
              />
            </label>
          </div>

          {showParallelFusion ? (
            <div className="fusion-weight-card">
              <div className="fusion-weight-header">
                <strong>Weighted Hybrid Parallel Fusion</strong>
                <small>
                  Weighted RRF: higher weight means stronger influence in fusion.
                </small>
              </div>

              <div className="advanced-grid refinement-options">
                <label className="input-group">
                  <span>TF-IDF Weight</span>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    value={form.tfidfWeight}
                    onChange={(event) => onFieldChange(
                      "tfidfWeight",
                      event.target.value,
                    )}
                  />
                </label>

                <label className="input-group">
                  <span>BM25 Weight</span>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    value={form.bm25Weight}
                    onChange={(event) => onFieldChange(
                      "bm25Weight",
                      event.target.value,
                    )}
                  />
                </label>

                <label className="input-group">
                  <span>Embedding Weight</span>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    value={form.embeddingWeight}
                    onChange={(event) => onFieldChange(
                      "embeddingWeight",
                      event.target.value,
                    )}
                  />
                </label>
              </div>
            </div>
          ) : null}

          <div className="switch-grid">
            <label className="switch-card">
              <input
                type="checkbox"
                checked={form.useQueryRefinement}
                onChange={(event) => onFieldChange(
                  "useQueryRefinement",
                  event.target.checked,
                )}
              />
              <span>
                <strong>Query refinement</strong>
                <small>Pseudo-relevance feedback expansion</small>
              </span>
            </label>

            <label className="switch-card">
              <input
                type="checkbox"
                checked={form.includeRawText}
                onChange={(event) => onFieldChange(
                  "includeRawText",
                  event.target.checked,
                )}
              />
              <span>
                <strong>Raw document text</strong>
                <small>Show original raw text for top results</small>
              </span>
            </label>

            <label className="switch-card">
              <input
                type="checkbox"
                checked={form.usePersonalization}
                onChange={(event) => onFieldChange(
                  "usePersonalization",
                  event.target.checked,
                )}
              />
              <span>
                <strong>Personalization</strong>
                <small>Anonymous local query profile</small>
              </span>
            </label>
          </div>

          {form.useQueryRefinement ? (
            <div className="advanced-grid refinement-options">
              <label className="input-group">
                <span>Feedback Docs</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={form.feedbackDocs}
                  onChange={(event) => onFieldChange(
                    "feedbackDocs",
                    event.target.value,
                  )}
                />
              </label>

              <label className="input-group">
                <span>Expansion Terms</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={form.expansionTerms}
                  onChange={(event) => onFieldChange(
                    "expansionTerms",
                    event.target.value,
                  )}
                />
              </label>
            </div>
          ) : null}

          {form.usePersonalization ? (
            <div className="advanced-grid refinement-options">
              <label className="input-group">
                <span>Profile Terms</span>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={form.maxPersonalizationTerms}
                  onChange={(event) => onFieldChange(
                    "maxPersonalizationTerms",
                    event.target.value,
                  )}
                />
              </label>
            </div>
          ) : null}
        </div>
      </details>
    </section>
  );
}

export default AdvancedPanel;
