function Badge({
  children,
  tone = "default",
}) {
  return (
    <span className={`ui-badge ${tone}`}>
      {children}
    </span>
  );
}

export default Badge;
