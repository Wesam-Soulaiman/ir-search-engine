const NAV_ITEMS = [
  {
    value: "search",
    label: "Search",
  },
  {
    value: "rag",
    label: "RAG Chat",
  },
  {
    value: "analytics",
    label: "Analytics",
  },
];

function AppHeader({
  activeView,
  onNavigate,
}) {
  return (
    <header className="app-header">
      <button
        className="brand brand-button"
        type="button"
        onClick={() => onNavigate("search")}
        aria-label="IR Search Engine"
      >
        <span className="brand-mark">IR</span>
        <span className="brand-copy">
          <strong>Search Engine</strong>
          <small>Offline Information Retrieval System</small>
        </span>
      </button>

      <nav className="header-nav" aria-label="Application sections">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.value}
            type="button"
            className={activeView === item.value ? "active" : ""}
            onClick={() => onNavigate(item.value)}
          >
            {item.label}
          </button>
        ))}
      </nav>
    </header>
  );
}

export default AppHeader;
