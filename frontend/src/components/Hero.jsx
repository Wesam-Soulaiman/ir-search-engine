import SearchInput from "./SearchInput";
import ExampleQueries from "./ExampleQueries";

function Hero({
  query,
  dataset,
  loading,
  onQueryChange,
  onSubmit,
  onPickExample,
}) {
  return (
    <section className="hero-section" id="search">
      <div className="hero-glow" aria-hidden="true" />

      <div className="hero-content">
        <div className="eyebrow">
          <span className="pulse-dot" />
          Retrieval-ready · Offline · Evaluated
        </div>

        <h1>
          Explore large-scale IR datasets with intelligent retrieval.
        </h1>

        <p>
          Search Quora and Clinical Trials using BM25, TF-IDF, embeddings,
          hybrid retrieval, query refinement, and the retrieval strategy agent.
        </p>

        <SearchInput
          query={query}
          loading={loading}
          onQueryChange={onQueryChange}
          onSubmit={onSubmit}
        />

        <ExampleQueries
          dataset={dataset}
          onPickExample={onPickExample}
        />
      </div>
    </section>
  );
}

export default Hero;
