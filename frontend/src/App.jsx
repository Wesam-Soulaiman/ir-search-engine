import { useState } from "react";

import { searchDocuments } from "./api/searchApi";
import AdvancedPanel from "./components/AdvancedPanel";
import AnalyticsDashboard from "./components/AnalyticsDashboard";
import AppHeader from "./components/AppHeader";
import ControlPanel from "./components/ControlPanel";
import ErrorToast from "./components/ErrorToast";
import FeatureCards from "./components/FeatureCards";
import Hero from "./components/Hero";
import InsightsPanel from "./components/InsightsPanel";
import LoadingResults from "./components/LoadingResults";
import RagChatPanel from "./components/RagChatPanel";
import ResultsSection from "./components/ResultsSection";
import SearchSummary from "./components/SearchSummary";
import { DEFAULT_SEARCH_FORM } from "./config/searchOptions";

import "./App.css";

function App() {
  const [form, setForm] = useState(DEFAULT_SEARCH_FORM);
  const [activeView, setActiveView] = useState("search");
  const [activeAnalyticsTab, setActiveAnalyticsTab] = useState("evaluation");
  const [results, setResults] = useState([]);
  const [responseInfo, setResponseInfo] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const updateField = (field, value) => {
    setForm((currentForm) => ({
      ...currentForm,
      [field]: value,
    }));
  };

  const runSearch = async () => {
    if (!form.query.trim()) {
      setErrorMessage("Please enter a query before searching.");
      return;
    }

    setLoading(true);
    setHasSearched(true);
    setErrorMessage("");
    setResults([]);
    setResponseInfo(null);

    try {
      const data = await searchDocuments(form);

      setResults(data.results || []);
      setResponseInfo({
        query: data.query,
        original_query: data.original_query,
        corrected_query: data.corrected_query,
        use_spelling_correction: data.use_spelling_correction,
        spelling_correction_used: data.spelling_correction_used,
        spelling_corrections: data.spelling_corrections,
        refined_query: data.refined_query,
        dataset: data.dataset,
        requested_model: data.requested_model || data.model,
        model: data.model,
        executed_model: data.executed_model || data.model,
        agent_selected_model: data.agent_selected_model,
        agent_reason: data.agent_reason,
        agent_features: data.agent_features,
        agent_fallback: data.agent_fallback,
        top_k: data.top_k,
        bm25_k1: data.bm25_k1,
        bm25_b: data.bm25_b,
        candidate_count: data.candidate_count,
        rrf_k: data.rrf_k,
        distributed: data.distributed,
        num_shards: data.num_shards,
        shards_queried: data.shards_queried,
        shard_top_k: data.shard_top_k,
        merge_method: data.merge_method,
        shard_result_counts: data.shard_result_counts,
        shard_document_counts: data.shard_document_counts,
        ltr: data.ltr,
        ltr_model_path: data.ltr_model_path,
        candidate_models: data.candidate_models,
        include_biomedical: data.include_biomedical,
        feature_count: data.feature_count,
        training_metadata: data.training_metadata,
        rag: data.rag,
        rag_retriever_model: data.rag_retriever_model,
        rag_context_docs: data.rag_context_docs,
        rag_answer_sentences: data.rag_answer_sentences,
        include_sources: data.include_sources,
        rag_generation_mode: data.rag_generation_mode,
        rag_llm_provider: data.rag_llm_provider,
        rag_llm_model: data.rag_llm_model,
        rag_llm_base_url: data.rag_llm_base_url,
        rag_llm_temperature: data.rag_llm_temperature,
        rag_llm_max_tokens: data.rag_llm_max_tokens,
        answer: data.answer,
        answer_confidence: data.answer_confidence,
        sources: data.sources,
        metadata: data.metadata,
        tfidf_weight: data.tfidf_weight,
        bm25_weight: data.bm25_weight,
        embedding_weight: data.embedding_weight,
        biomedical_weight: data.biomedical_weight,
        fusion_method: data.fusion_method,
        use_query_refinement: data.use_query_refinement,
        feedback_docs: data.feedback_docs,
        expansion_terms: data.expansion_terms,
        use_personalization: data.use_personalization,
        personalization_used: data.personalization_used,
        personalized_query: data.personalized_query,
        personalization_terms: data.personalization_terms,
        max_personalization_terms: data.max_personalization_terms,
        snippet_length: data.snippet_length,
        include_raw_text: data.include_raw_text,
        document_source: data.document_source,
        result_count: data.result_count,
      });
    } catch (error) {
      const apiMessage = error?.response?.data?.error;

      setErrorMessage(
        apiMessage
        || "Could not connect to the Django API. Make sure the backend is running on port 8000.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <AppHeader
        activeView={activeView}
        onNavigate={setActiveView}
      />

      {activeView === "search" ? (
        <>
          <Hero
            query={form.query}
            dataset={form.dataset}
            loading={loading}
            onQueryChange={(value) => updateField("query", value)}
            onSubmit={runSearch}
            onPickExample={(query) => updateField("query", query)}
          />

          <section className="workspace">
            <div className="workspace-main">
              <ControlPanel
                form={form}
                onFieldChange={updateField}
              />

              <AdvancedPanel
                form={form}
                onFieldChange={updateField}
              />

              {form.model === "rag" ? (
                <RagChatPanel
                  form={form}
                  onFieldChange={updateField}
                />
              ) : null}

              <SearchSummary info={responseInfo} />

              {loading ? (
                <LoadingResults />
              ) : (
                <ResultsSection
                  results={results}
                  hasSearched={hasSearched}
                  info={responseInfo}
                />
              )}
            </div>

            <aside className="workspace-aside">
              <FeatureCards />

              <InsightsPanel dataset={form.dataset} />
            </aside>
          </section>
        </>
      ) : null}

      {activeView === "rag" ? (
        <section className="single-workspace">
          <RagChatPanel
            form={form}
            onFieldChange={updateField}
          />
        </section>
      ) : null}

      {activeView === "analytics" ? (
        <AnalyticsDashboard
          dataset={form.dataset}
          activeTab={activeAnalyticsTab}
          onDatasetChange={(value) => updateField("dataset", value)}
          onTabChange={setActiveAnalyticsTab}
        />
      ) : null}

      <ErrorToast
        message={errorMessage}
        onDismiss={() => setErrorMessage("")}
      />
    </main>
  );
}

export default App;
