import {
  RAG_GENERATION_MODES,
  RAG_RETRIEVER_MODELS,
} from "../config/searchOptions";

function AdvancedGroup({
  title,
  description,
  children,
}) {
  return (
    <section className="advanced-group">
      <div className="advanced-group-header">
        <strong>{title}</strong>
        <small>{description}</small>
      </div>
      {children}
    </section>
  );
}

function SwitchCard({
  checked,
  onChange,
  title,
  description,
}) {
  return (
    <label className="switch-card">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>
        <strong>{title}</strong>
        <small>{description}</small>
      </span>
    </label>
  );
}

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
  const showDistributed = form.model === "distributed_bm25";
  const showLtr = form.model === "ltr";
  const showRag = form.model === "rag";
  const showBiomedicalFusion = (
    form.dataset === "clinical_trials"
    && form.model === "hybrid_parallel"
  );
  const showLtrBiomedical = (
    form.dataset === "clinical_trials"
    && showLtr
  );
  const ragRetrieverModels = RAG_RETRIEVER_MODELS.filter(
    (model) => !model.clinicalOnly || form.dataset === "clinical_trials",
  );
  const showRagLocalLlm = (
    showRag
    && form.ragGenerationMode === "local_llm"
  );

  const updateLtrCandidateModel = (modelName, enabled) => {
    onFieldChange(
      "ltrCandidateModels",
      {
        ...(form.ltrCandidateModels || {}),
        [modelName]: enabled,
      },
    );
  };

  return (
    <section className="advanced-card">
      <details>
        <summary>
          <span>Advanced options</span>
          <small>Ranking, fusion, query refinement, RAG, and output controls</small>
        </summary>

        <div className="advanced-body">
          <AdvancedGroup
            title="Ranking parameters"
            description="Core retrieval settings used by lexical, hybrid, and result-display workflows."
          >
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
                <small>Term-frequency saturation</small>
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
                <small>Length normalization</small>
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
                  <small>Pool before fusion/rerank</small>
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
                  <small>Rank fusion smoothing</small>
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
                <small>Preview characters</small>
              </label>
            </div>
          </AdvancedGroup>

          {showParallelFusion ? (
            <AdvancedGroup
              title="Hybrid parallel fusion"
              description="Weighted RRF combines multiple retrieval signals without changing model outputs."
            >
              <div className="advanced-grid">
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

                {showBiomedicalFusion ? (
                  <label className="input-group">
                    <span>Biomedical Weight</span>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      step="0.1"
                      value={form.biomedicalWeight}
                      onChange={(event) => onFieldChange(
                        "biomedicalWeight",
                        event.target.value,
                      )}
                    />
                  </label>
                ) : null}
              </div>
            </AdvancedGroup>
          ) : null}

          {showDistributed ? (
            <AdvancedGroup
              title="Distributed BM25"
              description="Local shard fan-out with an RRF coordinator merge."
            >
              <div className="advanced-grid">
                <label className="input-group">
                  <span>Shards</span>
                  <input
                    type="number"
                    min="1"
                    max="1024"
                    value={form.numShards}
                    onChange={(event) => onFieldChange(
                      "numShards",
                      event.target.value,
                    )}
                  />
                </label>

                <label className="input-group">
                  <span>Shard Top K</span>
                  <input
                    type="number"
                    min="1"
                    max="100000"
                    value={form.shardTopK}
                    onChange={(event) => onFieldChange(
                      "shardTopK",
                      event.target.value,
                    )}
                  />
                </label>

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
              </div>
            </AdvancedGroup>
          ) : null}

          {showLtr ? (
            <AdvancedGroup
              title="Learning to Rank"
              description="Rerank candidates from selected base retrieval models."
            >
              <div className="advanced-grid refinement-options">
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
              </div>

              <div className="switch-grid">
                <SwitchCard
                  checked={Boolean(form.ltrCandidateModels?.bm25)}
                  onChange={(enabled) => updateLtrCandidateModel("bm25", enabled)}
                  title="BM25"
                  description="Lexical candidate source"
                />
                <SwitchCard
                  checked={Boolean(form.ltrCandidateModels?.tfidf)}
                  onChange={(enabled) => updateLtrCandidateModel("tfidf", enabled)}
                  title="TF-IDF"
                  description="Vector-space candidate source"
                />
                <SwitchCard
                  checked={Boolean(form.ltrCandidateModels?.embedding)}
                  onChange={(enabled) => updateLtrCandidateModel("embedding", enabled)}
                  title="Embedding"
                  description="Semantic candidate source"
                />
                {showLtrBiomedical ? (
                  <SwitchCard
                    checked={Boolean(form.includeBiomedical)}
                    onChange={(enabled) => onFieldChange("includeBiomedical", enabled)}
                    title="Biomedical PubMedBERT"
                    description="Clinical Trials candidate source"
                  />
                ) : null}
              </div>
            </AdvancedGroup>
          ) : null}

          {showRag ? (
            <AdvancedGroup
              title="RAG options"
              description="Configure grounded answer synthesis over retrieved evidence."
            >
              <div className="advanced-grid">
                <label className="input-group">
                  <span>Retriever</span>
                  <select
                    value={form.ragRetrieverModel}
                    onChange={(event) => onFieldChange(
                      "ragRetrieverModel",
                      event.target.value,
                    )}
                  >
                    {ragRetrieverModels.map((model) => (
                      <option key={model.value} value={model.value}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Context Docs</span>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={form.ragContextDocs}
                    onChange={(event) => onFieldChange(
                      "ragContextDocs",
                      event.target.value,
                    )}
                  />
                </label>

                <label className="input-group">
                  <span>Answer Sentences</span>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={form.ragAnswerSentences}
                    onChange={(event) => onFieldChange(
                      "ragAnswerSentences",
                      event.target.value,
                    )}
                  />
                </label>

                <label className="input-group">
                  <span>Generation Mode</span>
                  <select
                    value={form.ragGenerationMode}
                    onChange={(event) => onFieldChange(
                      "ragGenerationMode",
                      event.target.value,
                    )}
                  >
                    {RAG_GENERATION_MODES.map((mode) => (
                      <option key={mode.value} value={mode.value}>
                        {mode.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {showRagLocalLlm ? (
                <div className="advanced-grid refinement-options">
                  <label className="input-group">
                    <span>LLM Provider</span>
                    <select
                      value={form.ragLlmProvider}
                      onChange={(event) => onFieldChange(
                        "ragLlmProvider",
                        event.target.value,
                      )}
                    >
                      <option value="ollama">Ollama</option>
                    </select>
                  </label>

                  <label className="input-group">
                    <span>LLM Model</span>
                    <input
                      type="text"
                      value={form.ragLlmModel}
                      onChange={(event) => onFieldChange(
                        "ragLlmModel",
                        event.target.value,
                      )}
                    />
                  </label>

                  <label className="input-group">
                    <span>Base URL</span>
                    <input
                      type="text"
                      value={form.ragLlmBaseUrl}
                      onChange={(event) => onFieldChange(
                        "ragLlmBaseUrl",
                        event.target.value,
                      )}
                    />
                  </label>

                  <label className="input-group">
                    <span>Temperature</span>
                    <input
                      type="number"
                      min="0"
                      max="2"
                      step="0.1"
                      value={form.ragLlmTemperature}
                      onChange={(event) => onFieldChange(
                        "ragLlmTemperature",
                        event.target.value,
                      )}
                    />
                  </label>

                  <label className="input-group">
                    <span>Max Tokens</span>
                    <input
                      type="number"
                      min="1"
                      max="4096"
                      value={form.ragLlmMaxTokens}
                      onChange={(event) => onFieldChange(
                        "ragLlmMaxTokens",
                        event.target.value,
                      )}
                    />
                  </label>
                </div>
              ) : null}

              <div className="switch-grid">
                <SwitchCard
                  checked={Boolean(form.includeSources)}
                  onChange={(enabled) => onFieldChange("includeSources", enabled)}
                  title="Sources"
                  description="Show cited documents used for the answer"
                />
              </div>
            </AdvancedGroup>
          ) : null}

          <AdvancedGroup
            title="Query processing and output"
            description="Optional query cleanup, expansion, personalization, and raw text display."
          >
            <div className="switch-grid">
              <SwitchCard
                checked={form.useSpellingCorrection}
                onChange={(enabled) => onFieldChange("useSpellingCorrection", enabled)}
                title="Spelling correction"
                description="Offline conservative query cleanup"
              />
              <SwitchCard
                checked={form.useQueryRefinement}
                onChange={(enabled) => onFieldChange("useQueryRefinement", enabled)}
                title="Query refinement"
                description="Pseudo-relevance feedback expansion"
              />
              <SwitchCard
                checked={form.includeRawText}
                onChange={(enabled) => onFieldChange("includeRawText", enabled)}
                title="Raw document text"
                description="Show original raw text for top results"
              />
              <SwitchCard
                checked={form.usePersonalization}
                onChange={(enabled) => onFieldChange("usePersonalization", enabled)}
                title="Personalization"
                description="Anonymous local query profile"
              />
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
          </AdvancedGroup>
        </div>
      </details>
    </section>
  );
}

export default AdvancedPanel;
