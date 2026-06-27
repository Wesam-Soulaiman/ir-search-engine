function EmptyState({
  id,
  title,
  message,
  tone = "default",
  children,
}) {
  return (
    <section className={`empty-state ${tone}`} id={id}>
      <div className="empty-state-icon" aria-hidden="true" />
      <h2>{title}</h2>
      <p>{message}</p>
      {children ? (
        <div className="empty-state-actions">
          {children}
        </div>
      ) : null}
    </section>
  );
}

export default EmptyState;
