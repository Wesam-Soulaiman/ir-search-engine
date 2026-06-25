function LoadingSkeleton() {
  return (
    <section className="skeleton-list" aria-label="Loading search results">
      {Array.from({ length: 4 }).map((_, index) => (
        <div className="skeleton-card" key={index}>
          <div className="skeleton-line short" />
          <div className="skeleton-line title" />
          <div className="skeleton-line" />
          <div className="skeleton-line" />
        </div>
      ))}
    </section>
  );
}

export default LoadingSkeleton;
