function LoadingResults() {
  return (
    <section className="loading-stack" aria-label="Loading results">
      {Array.from({ length: 4 }).map((_, index) => (
        <div className="loading-card" key={index}>
          <div className="loading-line tiny" />
          <div className="loading-line title" />
          <div className="loading-line" />
          <div className="loading-line short" />
        </div>
      ))}
    </section>
  );
}

export default LoadingResults;
