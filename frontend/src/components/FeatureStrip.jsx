const FEATURES = [
  "TF-IDF",
  "BM25",
  "Embeddings",
  "FAISS Vector Store",
  "Hybrid Serial",
  "Hybrid Parallel",
  "Query Refinement",
  "Clustering",
  "Topic Detection",
  "Agents",
];

function FeatureStrip() {
  return (
    <section id="features" className="feature-strip" aria-label="Implemented features">
      {FEATURES.map((feature) => (
        <span key={feature}>{feature}</span>
      ))}
    </section>
  );
}

export default FeatureStrip;
