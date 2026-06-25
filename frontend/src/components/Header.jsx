function Header() {
  return (
    <header className="top-bar">
      <div className="brand-mini" aria-label="IR Search Engine">
        <span className="brand-dot blue" />
        <span className="brand-dot red" />
        <span className="brand-dot yellow" />
        <span className="brand-dot green" />
        <span className="brand-text">IR Search</span>
      </div>

      <nav className="top-nav" aria-label="Main navigation">
        <a href="#search">Search</a>
        <a href="#features">Features</a>
        <a href="#results">Results</a>
      </nav>
    </header>
  );
}

export default Header;
