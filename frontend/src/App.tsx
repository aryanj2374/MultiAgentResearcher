import { useMemo, useState } from "react";

type Paper = {
  paper_id: string;
  title: string;
  authors: string[];
  year?: number | null;
  venue?: string | null;
  url?: string | null;
  abstract?: string | null;
};

type StudyExtraction = {
  paper_id: string;
  claim_summary: string;
  study_type: string;
  population?: string | null;
  sample_size?: number | null;
  intervention_exposure?: string | null;
  comparison?: string | null;
  outcomes?: string | null;
  effect_direction?: string | null;
  effect_size_text?: string | null;
  key_snippet: string;
  limitations: string[];
  apa_citation: string;
  url?: string | null;
};

type Critique = {
  paper_id: string;
  risk_of_bias: string;
  rationale: string[];
  red_flags: string[];
};

type Synthesis = {
  final_answer: string[];
  evidence_consensus: string;
  top_limitations_overall: string[];
  confidence_score: number;
  confidence_rationale: string[];
  citations_used: string[];
};

type Verification = {
  passed: boolean;
  issues: string[];
  revised_synthesis?: Synthesis | null;
};

type RunResponse = {
  run_id: string;
  question: string;
  papers: Paper[];
  extractions: StudyExtraction[];
  critiques: Critique[];
  synthesis: Synthesis;
  verification: Verification;
  logs?: Record<string, unknown> | null;
};

type TabKey = "final" | "evidence" | "logs";

type StepStatus = "idle" | "running" | "done";

type IconSpec = { paths: string[] };

const icons: Record<string, IconSpec> = {
  plus: { paths: ["M12 5v14", "M5 12h14"] },
  search: {
    paths: [
      "M11 19a8 8 0 1 1 0-16 8 8 0 0 1 0 16Z",
      "M21 21l-4.3-4.3",
    ],
  },
  chat: { paths: ["M4 6h16v10H7l-3 3V6Z"] },
  retrieve: { paths: ["M4 7h16", "M4 12h10", "M4 17h6"] },
  extract: { paths: ["M6 4h9l3 3v13H6z", "M9 12h6", "M9 16h6"] },
  critique: { paths: ["M12 5l7 12H5l7-12Z", "M12 10v3", "M12 16h.01"] },
  synthesize: { paths: ["M5 12h14", "M12 5v14", "M7 7l10 10"] },
  verify: { paths: ["M6 12l4 4 8-8"] },
  answer: { paths: ["M5 6h14", "M5 12h10", "M5 18h7"] },
  table: { paths: ["M4 7h16", "M4 12h16", "M4 17h16", "M10 7v10"] },
  logs: { paths: ["M6 6h12", "M6 12h12", "M6 18h8"] },
  send: { paths: ["M4 12l16-7-4 7 4 7-16-7Z"] },
};

const stepItems = [
  { key: "Retrieve", icon: "retrieve" },
  { key: "Extract", icon: "extract" },
  { key: "Critique", icon: "critique" },
  { key: "Synthesize", icon: "synthesize" },
  { key: "Verify", icon: "verify" },
];

const tabItems: Array<{ key: TabKey; label: string; icon: string }> = [
  { key: "final", label: "Answer", icon: "answer" },
  { key: "evidence", label: "Evidence", icon: "table" },
  { key: "logs", label: "Logs", icon: "logs" },
];

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const spec = icons[name];
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {spec.paths.map((path, index) => (
        <path key={index} d={path} />
      ))}
    </svg>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [runs, setRuns] = useState<RunResponse[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>("final");

  const activeRun = runs.find((run) => run.run_id === activeRunId) ?? runs[0] ?? null;
  const stepStatus: StepStatus = loading ? "running" : activeRun ? "done" : "idle";

  const extractionById = useMemo(() => {
    const map = new Map<string, StudyExtraction>();
    activeRun?.extractions.forEach((item) => map.set(item.paper_id, item));
    return map;
  }, [activeRun]);

  const critiqueById = useMemo(() => {
    const map = new Map<string, Critique>();
    activeRun?.critiques.forEach((item) => map.set(item.paper_id, item));
    return map;
  }, [activeRun]);

  const handleRun = async () => {
    if (!question.trim()) {
      setError("Please enter a research question.");
      return;
    }
    setLoading(true);
    setError(null);
    setPendingQuestion(question);
    setActiveTab("final");

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        const detail = payload?.detail ?? "Request failed";
        throw new Error(detail);
      }

      const data: RunResponse = await res.json();
      setRuns((prev) => [data, ...prev]);
      setActiveRunId(data.run_id);
      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
      setPendingQuestion(null);
    }
  };

  const handleSelectRun = (runId: string) => {
    setActiveRunId(runId);
    setActiveTab("final");
  };

  const handleNewChat = () => {
    setActiveRunId(null);
    setPendingQuestion(null);
    setError(null);
    setActiveTab("final");
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo">MA</div>
          <div>
            <p className="brand-title">Multi-Agent Research</p>
            <p className="brand-subtitle">Scientific Assistant</p>
          </div>
        </div>

        <div className="sidebar-actions">
          <button className="icon-row" onClick={handleNewChat} type="button">
            <Icon name="plus" />
            <span>New chat</span>
          </button>
          <button className="icon-row muted" type="button" aria-disabled="true">
            <Icon name="search" />
            <span>Search</span>
          </button>
        </div>

        <div className="history">
          <p className="section-title">History</p>
          {runs.length === 0 && <p className="muted">No runs yet.</p>}
          <ul>
            {runs.map((run) => (
              <li key={run.run_id}>
                <button
                  className={`history-item ${run.run_id === activeRun?.run_id ? "active" : ""}`}
                  onClick={() => handleSelectRun(run.run_id)}
                >
                  <Icon name="chat" />
                  <span>{run.question}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </aside>

      <main className="chat">
        <header className="topbar">
          <div className="topbar-left">
            <p className="topbar-title">Research Workspace</p>
            <span className="topbar-pill">HF · Multi-Agent</span>
          </div>
          <div className="topbar-right">
            <span className={`status ${loading ? "running" : "idle"}`}></span>
          </div>
        </header>

        <div className="chat-scroll">
          {!activeRun && !pendingQuestion && (
            <div className="empty-state">
              <h1>What are you investigating today?</h1>
              <p>Ask a scientific question and get a citation‑grounded synthesis.</p>
            </div>
          )}

          {(pendingQuestion || activeRun) && (
            <div className="thread">
              <div className="message user">
                <div className="bubble">{pendingQuestion ?? activeRun?.question}</div>
              </div>

              <div className="message assistant">
                <div className="bubble">
                  <div className="pipeline" aria-label="Pipeline stages">
                    {stepItems.map((step) => (
                      <div key={step.key} className={`pipeline-step ${stepStatus}`} title={step.key}>
                        <Icon name={step.icon} />
                      </div>
                    ))}
                  </div>

                  {loading && <p className="muted">Agents are working. This usually takes under a minute.</p>}

                  {activeRun && (
                    <>
                      <div className="tabs">
                        {tabItems.map((tab) => (
                          <button
                            key={tab.key}
                            className={activeTab === tab.key ? "active" : ""}
                            onClick={() => setActiveTab(tab.key)}
                          >
                            <Icon name={tab.icon} />
                            <span>{tab.label}</span>
                          </button>
                        ))}
                      </div>

                      {activeTab === "final" && (
                        <div className="final-answer">
                          <h2>Final Answer</h2>
                          <ul>
                            {activeRun.synthesis.final_answer.map((bullet, index) => (
                              <li key={index}>{bullet}</li>
                            ))}
                          </ul>

                          <h3>Evidence Consensus</h3>
                          <p>{activeRun.synthesis.evidence_consensus}</p>

                          <div className="grid-two">
                            <div>
                              <h3>Top Limitations</h3>
                              <ul>
                                {activeRun.synthesis.top_limitations_overall.map((item, index) => (
                                  <li key={index}>{item}</li>
                                ))}
                              </ul>
                            </div>
                            <div>
                              <h3>Confidence</h3>
                              <p className="confidence">{activeRun.synthesis.confidence_score}/100</p>
                              <ul>
                                {activeRun.synthesis.confidence_rationale.map((item, index) => (
                                  <li key={index}>{item}</li>
                                ))}
                              </ul>
                            </div>
                          </div>

                          {!activeRun.verification.passed && activeRun.verification.issues.length > 0 && (
                            <div className="callout">
                              <h3>Verification Issues</h3>
                              <ul>
                                {activeRun.verification.issues.map((issue, index) => (
                                  <li key={index}>{issue}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}

                      {activeTab === "evidence" && (
                        <div className="evidence">
                          <h2>Evidence Table</h2>
                          <div className="table">
                            <div className="table-row header">
                              <div>Paper</div>
                              <div>Study Type</div>
                              <div>Sample Size</div>
                              <div>Effect</div>
                              <div>Bias</div>
                            </div>
                            {activeRun.papers.map((paper) => {
                              const extraction = extractionById.get(paper.paper_id);
                              const critique = critiqueById.get(paper.paper_id);
                              return (
                                <div className="table-row" key={paper.paper_id}>
                                  <div>
                                    <p className="paper-title">{paper.title}</p>
                                    <p className="paper-meta">
                                      {paper.authors.slice(0, 2).join(", ")}
                                      {paper.authors.length > 2 ? " et al." : ""} {paper.year ?? ""}
                                    </p>
                                    {paper.url && (
                                      <a href={paper.url} target="_blank" rel="noreferrer">
                                        View paper
                                      </a>
                                    )}
                                    {extraction && <p className="paper-claim">{extraction.claim_summary}</p>}
                                  </div>
                                  <div>{extraction?.study_type ?? "-"}</div>
                                  <div>{extraction?.sample_size ?? "-"}</div>
                                  <div>{extraction?.effect_direction ?? "-"}</div>
                                  <div className={`bias ${critique?.risk_of_bias ?? "unknown"}`}>
                                    {critique?.risk_of_bias ?? "unknown"}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {activeTab === "logs" && (
                        <div className="logs">
                          <h2>Agent Logs</h2>
                          <pre>{JSON.stringify(activeRun.logs ?? activeRun, null, 2)}</pre>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="input-bar">
          <div className="input-wrapper">
            <button className="icon-button" type="button" aria-label="Add context">
              <Icon name="plus" />
            </button>
            <textarea
              placeholder="Ask a research question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={2}
            />
            <button className="send" onClick={handleRun} disabled={loading}>
              <Icon name="send" />
              <span>{loading ? "Running" : "Send"}</span>
            </button>
          </div>
          {error && <p className="error">{error}</p>}
        </div>
      </main>
    </div>
  );
}

export default App;
