import { useEffect, useMemo, useState } from "react";

import { searchDocuments } from "../api/searchApi";
import {
  DATASETS,
  RAG_GENERATION_MODES,
  RAG_RETRIEVER_MODELS,
} from "../config/searchOptions";
import Badge from "./ui/Badge";

const STORAGE_KEY = "ir_rag_chat_messages";

function createMessageId() {
  return `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function loadStoredMessages() {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);

    if (!stored) {
      return [];
    }

    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function persistMessages(messages) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  } catch {
    // Chat history persistence is optional.
  }
}

function RagChatPanel({
  form,
  onFieldChange,
}) {
  const [messages, setMessages] = useState(loadStoredMessages);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    persistMessages(messages);
  }, [messages]);

  const ragRetrieverModels = useMemo(
    () => RAG_RETRIEVER_MODELS.filter(
      (model) => !model.clinicalOnly || form.dataset === "clinical_trials",
    ),
    [form.dataset],
  );
  const showLocalLlm = form.ragGenerationMode === "local_llm";

  useEffect(() => {
    const selectedRetriever = ragRetrieverModels.some(
      (model) => model.value === form.ragRetrieverModel,
    );

    if (!selectedRetriever) {
      onFieldChange("ragRetrieverModel", "hybrid_serial");
    }
  }, [form.ragRetrieverModel, onFieldChange, ragRetrieverModels]);

  const clearChat = () => {
    setMessages([]);
  };

  const sendMessage = async () => {
    const query = draft.trim();

    if (!query || loading) {
      return;
    }

    const userMessage = {
      id: createMessageId(),
      role: "user",
      content: query,
      createdAt: new Date().toISOString(),
    };

    setMessages((currentMessages) => [
      ...currentMessages,
      userMessage,
    ]);
    setDraft("");
    setLoading(true);

    try {
      const response = await searchDocuments({
        ...form,
        model: "rag",
        query,
      });

      const assistantMessage = {
        id: createMessageId(),
        role: "assistant",
        content: response.answer || "",
        confidence: response.answer_confidence,
        sources: response.sources || [],
        results: response.results || [],
        metadata: response.metadata || {},
        ragGenerationMode: response.rag_generation_mode,
        ragRetrieverModel: response.rag_retriever_model,
        createdAt: new Date().toISOString(),
      };

      setMessages((currentMessages) => [
        ...currentMessages,
        assistantMessage,
      ]);
    } catch (error) {
      const apiMessage = error?.response?.data?.error;

      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: createMessageId(),
          role: "error",
          content: apiMessage || "RAG chat request failed.",
          createdAt: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const onDraftKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <section className="rag-chat-panel">
      <div className="rag-chat-header">
        <div>
          <span className="section-kicker">RAG Chat</span>
          <h2>Ask against retrieved documents</h2>
          <p>
            Answers are generated after retrieval and include ranked evidence
            when sources are enabled.
          </p>
        </div>

        <button
          className="secondary-action"
          type="button"
          onClick={clearChat}
          disabled={!messages.length || loading}
        >
          Clear chat
        </button>
      </div>

      <div className="rag-status-row" aria-label="Current RAG configuration">
        <Badge tone="info">{form.dataset}</Badge>
        <Badge>{form.ragRetrieverModel}</Badge>
        <Badge tone="success">{form.ragGenerationMode}</Badge>
        <Badge>{form.ragContextDocs} context docs</Badge>
      </div>

      <div className="rag-chat-controls">
        <label className="input-group">
          <span>Dataset</span>
          <select
            value={form.dataset}
            onChange={(event) => onFieldChange("dataset", event.target.value)}
          >
            {DATASETS.map((dataset) => (
              <option key={dataset.value} value={dataset.value}>
                {dataset.label}
              </option>
            ))}
          </select>
        </label>

        <label className="input-group">
          <span>Retriever</span>
          <select
            value={form.ragRetrieverModel}
            onChange={(event) => onFieldChange(
              "ragRetrieverModel",
              event.target.value,
            )}
          >
            {ragRetrieverModels.map((model) => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
        </label>

        <label className="input-group">
          <span>Generation</span>
          <select
            value={form.ragGenerationMode}
            onChange={(event) => onFieldChange(
              "ragGenerationMode",
              event.target.value,
            )}
          >
            {RAG_GENERATION_MODES.map((mode) => (
              <option key={mode.value} value={mode.value}>
                {mode.label}
              </option>
            ))}
          </select>
        </label>

        <label className="input-group">
          <span>Context Docs</span>
          <input
            type="number"
            min="1"
            max="50"
            value={form.ragContextDocs}
            onChange={(event) => onFieldChange(
              "ragContextDocs",
              event.target.value,
            )}
          />
        </label>

        <label className="input-group">
          <span>Answer Sentences</span>
          <input
            type="number"
            min="1"
            max="10"
            value={form.ragAnswerSentences}
            onChange={(event) => onFieldChange(
              "ragAnswerSentences",
              event.target.value,
            )}
          />
        </label>

        <label className="switch-card compact-switch">
          <input
            type="checkbox"
            checked={Boolean(form.includeSources)}
            onChange={(event) => onFieldChange(
              "includeSources",
              event.target.checked,
            )}
          />
          <span>
            <strong>Sources</strong>
            <small>Show citations</small>
          </span>
        </label>
      </div>

      {showLocalLlm ? (
        <div className="rag-chat-controls local-llm-controls">
          <label className="input-group">
            <span>Provider</span>
            <select
              value={form.ragLlmProvider}
              onChange={(event) => onFieldChange(
                "ragLlmProvider",
                event.target.value,
              )}
            >
              <option value="ollama">Ollama</option>
            </select>
          </label>

          <label className="input-group">
            <span>Model</span>
            <input
              type="text"
              value={form.ragLlmModel}
              onChange={(event) => onFieldChange(
                "ragLlmModel",
                event.target.value,
              )}
            />
          </label>

          <label className="input-group">
            <span>Base URL</span>
            <input
              type="text"
              value={form.ragLlmBaseUrl}
              onChange={(event) => onFieldChange(
                "ragLlmBaseUrl",
                event.target.value,
              )}
            />
          </label>

          <label className="input-group">
            <span>Temperature</span>
            <input
              type="number"
              min="0"
              max="2"
              step="0.1"
              value={form.ragLlmTemperature}
              onChange={(event) => onFieldChange(
                "ragLlmTemperature",
                event.target.value,
              )}
            />
          </label>

          <label className="input-group">
            <span>Max Tokens</span>
            <input
              type="number"
              min="1"
              max="4096"
              value={form.ragLlmMaxTokens}
              onChange={(event) => onFieldChange(
                "ragLlmMaxTokens",
                event.target.value,
              )}
            />
          </label>
        </div>
      ) : null}

      <div className="rag-chat-thread" aria-live="polite">
        {!messages.length ? (
          <div className="rag-chat-empty">
            <h3>Start a grounded RAG conversation</h3>
            <p>
              Ask a question and the assistant will answer from retrieved
              documents, with sources and the ranked evidence kept nearby.
            </p>
            <div className="empty-prompt-row">
              <button
                type="button"
                onClick={() => setDraft("What evidence is available for diabetes insulin treatment?")}
              >
                Diabetes insulin treatment
              </button>
              <button
                type="button"
                onClick={() => setDraft("What causes nightmares?")}
              >
                Nightmares evidence
              </button>
            </div>
          </div>
        ) : null}

        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}

        {loading ? (
          <div className="chat-row assistant">
            <div className="chat-bubble assistant">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        ) : null}
      </div>

      <div className="rag-chat-composer">
        <textarea
          value={draft}
          placeholder="Ask a RAG question..."
          rows={2}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onDraftKeyDown}
          disabled={loading}
        />

        <button
          type="button"
          onClick={sendMessage}
          disabled={!draft.trim() || loading}
        >
          Send
        </button>
      </div>
    </section>
  );
}

function ChatMessage({
  message,
}) {
  if (message.role === "user") {
    return (
      <div className="chat-row user">
        <div className="chat-bubble user">
          <small>You</small>
          <p>{message.content}</p>
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="chat-row assistant">
        <div className="chat-bubble error">
          <small>Error</small>
          <p>{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-row assistant">
      <div className="chat-bubble assistant">
        <div className="assistant-meta">
          <small>Assistant</small>
          <Badge tone="success">{message.confidence || "unknown"} confidence</Badge>
          {message.ragGenerationMode ? (
            <Badge>{message.ragGenerationMode}</Badge>
          ) : null}
          {message.ragRetrieverModel ? (
            <Badge tone="info">{message.ragRetrieverModel}</Badge>
          ) : null}
          {typeof message.metadata?.local_llm_used === "boolean" ? (
            <Badge>local LLM {String(message.metadata.local_llm_used)}</Badge>
          ) : null}
        </div>

        <p>{message.content}</p>

        {message.sources?.length ? (
          <div className="chat-source-list">
            <h4>Sources used</h4>
            {message.sources.map((source) => (
              <article
                className="chat-source-card"
                key={`${source.source_id}-${source.doc_id}`}
              >
                <span>[{source.source_id}]</span>
                <div>
                  <strong>
                    {source.title || `Document ${source.doc_id}`}
                  </strong>
                  <small>
                    Rank {source.rank} | Doc ID {source.doc_id}
                  </small>
                  {source.snippet ? (
                    <p>{source.snippet}</p>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        ) : null}

        {message.results?.length ? (
          <details className="chat-retrieved-results">
            <summary>
              Retrieved results ({message.results.length})
            </summary>

            <div>
              {message.results.slice(0, 5).map((result) => (
                <article key={`${result.rank}-${result.doc_id}`}>
                  <strong>
                    #{result.rank} {result.title || `Document ${result.doc_id}`}
                  </strong>
                  <small>
                    Doc ID {result.doc_id} | Score {result.score ?? "N/A"}
                  </small>
                  <p>{result.snippet || result.raw_text || "No preview available."}</p>
                </article>
              ))}
            </div>
          </details>
        ) : null}
      </div>
    </div>
  );
}

export default RagChatPanel;
