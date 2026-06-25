function SearchBox({
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
    <div className="search-box-wrap">
      <div className="search-box">
        <span className="search-icon" aria-hidden="true">⌕</span>

        <input
          id="main-query-input"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Search Quora or Clinical Trials..."
          autoComplete="off"
          autoFocus
        />

        {query ? (
          <button
            type="button"
            className="clear-button"
            onClick={() => onQueryChange("")}
            aria-label="Clear search"
          >
            ×
          </button>
        ) : null}
      </div>

      <div className="search-actions">
        <button
          type="button"
          className="primary-button"
          onClick={onSubmit}
          disabled={loading}
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
    </div>
  );
}

export default SearchBox;
