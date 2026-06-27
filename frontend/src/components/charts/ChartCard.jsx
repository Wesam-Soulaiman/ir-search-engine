function ChartCard({
  title,
  subtitle,
  actions,
  children,
}) {
  return (
    <section className="chart-card">
      <div className="chart-card-header">
        <div>
          <h3>{title}</h3>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>

        {actions ? (
          <div className="chart-card-actions">
            {actions}
          </div>
        ) : null}
      </div>

      {children}
    </section>
  );
}

export default ChartCard;
