import ResultCard from "./ResultCard";

function ResultsSection({
  results,
  hasSearched,
  info,
}) {
  const showRagAnswer = Boolean(info?.rag);

  if (!hasSearched) {
    return (
      <section className="placeholder-state" id="results">
        <div className="placeholder-orb" />
        <h2>Start a search to inspect ranked documents</h2>
        <p>
          Results will show document IDs, scores, snippets, raw text, and
          retrieval metadata from the offline document store.
        </p>
      </section>
    );
  }

  if (!results.length && !showRagAnswer) {
    return (
      <section className="placeholder-state" id="results">
        <div className="placeholder-orb danger" />
        <h2>No documents found</h2>
        <p>
          Try another query, switch dataset, or use the Agent/Hybrid models.
        </p>
      </section>
    );
  }

  return (
    <section className="results-section" id="results">
      {showRagAnswer ? (
        <article className="rag-answer-card">
          <div className="rag-answer-header">
            <div>
              <span>RAG answer</span>
              <h2>{info.answer_confidence || "unknown"} confidence</h2>
            </div>

            <strong>{info.rag_retriever_model}</strong>
          </div>

          <p>{info.answer}</p>

          {info.sources?.length ? (
            <div className="rag-source-list">
              {info.sources.map((source) => (
                <div
                  className="rag-source"
                  key={`${source.source_id}-${source.doc_id}`}
                >
                  <span>[{source.source_id}]</span>
                  <div>
                    <strong>
                      {source.title || `Document ${source.doc_id}`}
                    </strong>
                    <small>
                      Rank {source.rank} | Doc ID {source.doc_id}
                    </small>
                  </div>
                </div>
              ))}
            </div>
          ) : null}
        </article>
      ) : null}

      {results.map((result) => (
        <ResultCard
          key={`${result.rank}-${result.doc_id}`}
          result={result}
        />
      ))}
    </section>
  );
}

export default ResultsSection;
