function ErrorBanner({
  message,
  onDismiss,
}) {
  if (!message) {
    return null;
  }

  return (
    <div className="error-banner" role="alert">
      <div>
        <strong>Search failed</strong>
        <p>{message}</p>
      </div>

      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss error"
      >
        ×
      </button>
    </div>
  );
}

export default ErrorBanner;
