import ResultCard from "./ResultCard";

function ResultsList({
  results,
  hasSearched,
}) {
  if (!hasSearched) {
    return (
      <section className="empty-state" id="results">
        <h2>Ready to search</h2>
        <p>
          Choose a dataset, select a retrieval model, and run an offline IR search.
        </p>
      </section>
    );
  }

  if (!results.length) {
    return (
      <section className="empty-state" id="results">
        <h2>No results found</h2>
        <p>
          Try a different query, another dataset, or a hybrid retrieval model.
        </p>
      </section>
    );
  }

  return (
    <section className="results-list" id="results" aria-label="Search results">
      {results.map((result) => (
        <ResultCard
          key={`${result.doc_id}-${result.rank}`}
          result={result}
        />
      ))}
    </section>
  );
}

export default ResultsList;
