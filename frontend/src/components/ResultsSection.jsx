import ResultCard from "./ResultCard";
import EmptyState from "./ui/EmptyState";

function ResultsSection({
  results,
  hasSearched,
  info,
}) {
  const showRagAnswer = Boolean(info?.rag);

  if (!hasSearched) {
    return (
      <EmptyState
        id="results"
        title="Start a search to inspect ranked documents"
        message="Results will show rank, score, document identifiers, snippets, and retrieval metadata from the selected model."
      />
    );
  }

  if (!results.length && !showRagAnswer) {
    return (
      <EmptyState
        id="results"
        tone="danger"
        title="No documents found"
        message="Try another query, switch dataset, or use the Agent or Hybrid models."
      />
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
