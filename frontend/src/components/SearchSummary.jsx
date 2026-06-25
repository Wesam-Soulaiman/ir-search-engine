function SearchSummary({
  info,
}) {
  if (!info) {
    return null;
  }

  const model = info.executed_model || info.model;
  const isWeightedHybrid = model === "hybrid_parallel";

  return (
    <section className="search-summary">
      <div className="summary-grid">
        <div>
          <small>Dataset</small>
          <strong>{info.dataset}</strong>
        </div>

        <div>
          <small>Requested</small>
          <strong>{info.requested_model || info.model}</strong>
        </div>

        <div>
          <small>Executed</small>
          <strong>{model}</strong>
        </div>

        <div>
          <small>Results</small>
          <strong>{info.result_count}</strong>
        </div>
      </div>

      {isWeightedHybrid ? (
        <div className="summary-note">
          <span>Weighted hybrid fusion</span>
          <p>
            TF-IDF weight: {info.tfidf_weight}
            {" · "}
            BM25 weight: {info.bm25_weight}
            {" · "}
            Embedding weight: {info.embedding_weight}
            {" · "}
            Fusion: {info.fusion_method || "Weighted RRF"}
          </p>
        </div>
      ) : null}

      {info.original_query !== info.refined_query ? (
        <div className="summary-note">
          <span>Refined query</span>
          <p>{info.refined_query}</p>
        </div>
      ) : null}

      {info.personalization_used ? (
        <div className="summary-note">
          <span>Personalized query</span>
          <p>
            Original: {info.original_query}
            {" آ· "}
            Personalized: {info.personalized_query}
            {" آ· "}
            Added: {(info.personalization_terms || []).join(", ")}
          </p>
        </div>
      ) : null}

      {info.requested_model === "agent" ? (
        <div className="agent-decision">
          <div className="agent-decision-header">
            <span className="agent-icon">AI</span>
            <div>
              <strong>Agent decision</strong>
              <small>
                selected {info.agent_selected_model || "N/A"} · executed {model}
              </small>
            </div>
          </div>

          {info.agent_reason ? (
            <p>{info.agent_reason}</p>
          ) : null}

          {info.agent_fallback ? (
            <p className="fallback-message">{info.agent_fallback}</p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

export default SearchSummary;
