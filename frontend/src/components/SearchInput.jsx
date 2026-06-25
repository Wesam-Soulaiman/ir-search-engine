function SearchInput({
  query,
  loading,
  onQueryChange,
  onSubmit,
}) {
  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="search-input-card">
      <div className="search-input-row">
        <span className="search-symbol" aria-hidden="true">
          ⌘
        </span>

        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a query, e.g. melanoma BRAF V600E clinical trial"
          autoComplete="off"
          autoFocus
        />

        {query ? (
          <button
            type="button"
            className="ghost-icon-button"
            onClick={() => onQueryChange("")}
            aria-label="Clear query"
          >
            ×
          </button>
        ) : null}

        <button
          type="button"
          className="search-submit"
          onClick={onSubmit}
          disabled={loading}
        >
          {loading ? "Searching" : "Search"}
        </button>
      </div>
    </div>
  );
}

export default SearchInput;
