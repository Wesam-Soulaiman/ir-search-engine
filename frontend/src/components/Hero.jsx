import ExampleQueries from "./ExampleQueries";
import SearchInput from "./SearchInput";
import Badge from "./ui/Badge";

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
      <div className="hero-content">
        <div className="eyebrow">
          Retrieval-ready | Offline | Evaluated
        </div>

        <h1>
          Information Retrieval search lab.
        </h1>

        <p>
          Search Quora and Clinical Trials using BM25, TF-IDF, embeddings,
          hybrid fusion, query refinement, Learning-to-Rank, RAG, clustering,
          topic detection, and CSV-backed evaluation charts.
        </p>

        <div className="hero-badges" aria-label="Implemented system features">
          <Badge tone="info">Django REST</Badge>
          <Badge tone="success">React UI</Badge>
          <Badge>Multiple ranking models</Badge>
          <Badge>Report analytics</Badge>
        </div>

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
