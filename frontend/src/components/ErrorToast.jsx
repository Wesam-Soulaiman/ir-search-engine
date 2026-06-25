function ErrorToast({
  message,
  onDismiss,
}) {
  if (!message) {
    return null;
  }

  return (
    <div className="error-toast" role="alert">
      <div>
        <strong>Request failed</strong>
        <span>{message}</span>
      </div>

      <button type="button" onClick={onDismiss}>
        ×
      </button>
    </div>
  );
}

export default ErrorToast;
