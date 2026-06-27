function SearchSummary({
  info,
}) {
  if (!info) {
    return null;
  }

  const model = info.executed_model || info.model;
  const isWeightedHybrid = model === "hybrid_parallel";
  const isDistributed = Boolean(info.distributed) || model === "distributed_bm25";
  const isLtr = Boolean(info.ltr) || model === "ltr";
  const isRag = Boolean(info.rag) || model === "rag";

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
            {Number(info.biomedical_weight) > 0 ? (
              <>
                {" آ· "}
                Biomedical PubMedBERT weight: {info.biomedical_weight}
              </>
            ) : null}
            {" · "}
            Fusion: {info.fusion_method || "Weighted RRF"}
          </p>
        </div>
      ) : null}

      {isDistributed ? (
        <div className="summary-note">
          <span>Distributed: Yes</span>
          <p>
            Shards queried: {info.shards_queried}
            {" | "}
            Merge method: {info.merge_method || "RRF"}
            {" | "}
            Shard top-k: {info.shard_top_k}
            {" | "}
            RRF k: {info.rrf_k}
          </p>
        </div>
      ) : null}

      {isLtr ? (
        <div className="summary-note">
          <span>LTR: Yes</span>
          <p>
            Candidate count: {info.candidate_count}
            {" | "}
            Candidate models: {(info.candidate_models || []).join(", ")}
            {" | "}
            Include biomedical: {String(Boolean(info.include_biomedical))}
            {info.feature_count ? (
              <>
                {" | "}
                Features: {info.feature_count}
              </>
            ) : null}
            {info.ltr_model_path ? (
              <>
                {" | "}
                Model: {info.ltr_model_path}
              </>
            ) : null}
          </p>
        </div>
      ) : null}

      {isRag ? (
        <div className="summary-note">
          <span>RAG: Yes</span>
          <p>
            Retriever: {info.rag_retriever_model}
            {" | "}
            Mode: {info.rag_generation_mode || "extractive_offline"}
            {" | "}
            Context docs: {info.rag_context_docs}
            {" | "}
            Answer sentences: {info.rag_answer_sentences}
            {" | "}
            Confidence: {info.answer_confidence}
            {" | "}
            Local LLM: {String(Boolean(info.metadata?.local_llm_used))}
            {" | "}
            External LLM: {String(Boolean(info.metadata?.external_llm_used))}
            {info.rag_generation_mode === "local_llm" ? (
              <>
                {" | "}
                Provider: {info.rag_llm_provider}
                {" | "}
                Model: {info.rag_llm_model}
              </>
            ) : null}
          </p>
        </div>
      ) : null}

      {info.original_query !== info.refined_query ? (
        <div className="summary-note">
          <span>Refined query</span>
          <p>{info.refined_query}</p>
        </div>
      ) : null}

      {info.spelling_correction_used ? (
        <div className="summary-note">
          <span>Corrected query</span>
          <p>
            {info.corrected_query}
            {" | "}
            Corrections: {(info.spelling_corrections || [])
              .map((item) => `${item.original} -> ${item.corrected}`)
              .join(", ")}
          </p>
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
