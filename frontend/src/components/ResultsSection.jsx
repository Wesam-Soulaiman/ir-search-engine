import ResultCard from "./ResultCard";

function ResultsSection({
  results,
  hasSearched,
}) {
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

  if (!results.length) {
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
