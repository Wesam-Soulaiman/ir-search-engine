import { EXAMPLE_QUERIES } from "../config/searchOptions";

function ExampleQueries({
  dataset,
  onPickExample,
}) {
  const examples = EXAMPLE_QUERIES[dataset] || [];

  return (
    <div className="example-row">
      <span>Examples</span>

      {examples.map((query) => (
        <button
          type="button"
          key={query}
          onClick={() => onPickExample(query)}
        >
          {query}
        </button>
      ))}
    </div>
  );
}

export default ExampleQueries;
