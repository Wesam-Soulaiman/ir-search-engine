function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  meta,
}) {
  return (
    <header className="page-header">
      <div className="page-header-copy">
        {eyebrow ? (
          <span className="section-kicker">{eyebrow}</span>
        ) : null}
        <h1>{title}</h1>
        {description ? <p>{description}</p> : null}

        {meta ? (
          <div className="page-header-meta">
            {meta}
          </div>
        ) : null}
      </div>

      {actions ? (
        <div className="page-header-actions">
          {actions}
        </div>
      ) : null}
    </header>
  );
}

export default PageHeader;
