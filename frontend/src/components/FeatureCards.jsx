const FEATURES = [
  {
    title: "Vector Store",
    text: "FAISS-based embedding indexes for semantic retrieval.",
  },
  {
    title: "Hybrid Search",
    text: "Serial reranking and parallel result fusion.",
  },
  {
    title: "Query Refinement",
    text: "Pseudo-relevance feedback for query expansion.",
  },
  {
    title: "Agents",
    text: "Rule-based strategy agent chooses the best retrieval model.",
  },
  {
    title: "Clustering",
    text: "Document clustering built from saved embeddings.",
  },
  {
    title: "Topic Detection",
    text: "Cluster labels extracted from representative documents.",
  },
];

function FeatureCards() {
  return (
    <section className="feature-cards" aria-label="Implemented features">
      {FEATURES.map((feature) => (
        <article key={feature.title}>
          <h3>{feature.title}</h3>
          <p>{feature.text}</p>
        </article>
      ))}
    </section>
  );
}

export default FeatureCards;
