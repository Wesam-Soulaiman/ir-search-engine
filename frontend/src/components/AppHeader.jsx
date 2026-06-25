function AppHeader() {
  return (
    <header className="app-header">
      <a className="brand" href="#search" aria-label="IR Search Engine">
        <span className="brand-mark">IR</span>
        <span className="brand-copy">
          <strong>Search Engine</strong>
          <small>Offline Information Retrieval System</small>
        </span>
      </a>

      <nav className="header-nav" aria-label="Application sections">
        <a href="#search">Search</a>
        <a href="#controls">Models</a>
        <a href="#results">Results</a>
      </nav>
    </header>
  );
}

export default AppHeader;
