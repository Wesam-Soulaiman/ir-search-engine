function MetricCard({
  label,
  value,
  detail,
}) {
  return (
    <article className="metric-card">
      <small>{label}</small>
      <strong>{value}</strong>
      {detail ? <span>{detail}</span> : null}
    </article>
  );
}

export default MetricCard;
