function formatScore(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) {
    return "N/A";
  }

  return Number(score).toFixed(6);
}

function ResultCard({
  result,
}) {
  const title = result.title?.trim() || `Document ${result.doc_id}`;
  const preview = result.raw_text || result.snippet || "No preview available.";

  return (
    <article className="result-card">
      <aside className="rank-badge">
        {result.rank}
      </aside>

      <div className="result-content">
        <div className="result-source">
          <span>{result.document_source || "retrieval_index"}</span>
          <span>Doc ID: {result.doc_id}</span>
        </div>

        <h3>{title}</h3>

        <p className="result-preview">
          {preview}
        </p>

        <div className="result-metrics">
          <span>Score {formatScore(result.score)}</span>

          {result.bm25_rank !== null && result.bm25_rank !== undefined ? (
            <span>BM25 rank {result.bm25_rank}</span>
          ) : null}

          {result.bm25_score !== null && result.bm25_score !== undefined ? (
            <span>BM25 score {formatScore(result.bm25_score)}</span>
          ) : null}

          {result.hybrid_method ? (
            <span>{result.hybrid_method}</span>
          ) : null}
        </div>
      </div>
    </article>
  );
}

export default ResultCard;
