import { useMemo, useState } from "react";
import type { Message, BackendResponse } from "../types";
import Collapsible from "./Collapsible";

type MessageProps = {
  message: Message;
  onRetry?: (messageId: string, question: string) => void;
};

function buildSummary(response: BackendResponse | undefined, content: string): string {
  if (response?.synthesis?.evidence_consensus) return response.synthesis.evidence_consensus;
  if (response?.synthesis?.final_answer?.length) return response.synthesis.final_answer[0];
  return content || "Summary unavailable.";
}

function getConfidenceLevel(score: number): { label: string; className: string } {
  if (score >= 60) return { label: "High", className: "high" };
  if (score >= 30) return { label: "Medium", className: "medium" };
  return { label: "Low", className: "low" };
}

export default function Message({ message, onRetry }: MessageProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const response = message.meta?.response;
  const summary = useMemo(() => buildSummary(response, message.content), [response, message.content]);

  const handleCopy = async () => {
    const textParts = [message.content];
    if (response) {
      textParts.push("\nAnswer:\n" + response.synthesis.final_answer.join("\n"));
      textParts.push("\nEvidence consensus:\n" + response.synthesis.evidence_consensus);
    }
    await navigator.clipboard.writeText(textParts.join("\n\n"));
    setMenuOpen(false);
  };

  const handleRetry = () => {
    if (!onRetry || !message.meta?.request?.question) return;
    onRetry(message.id, message.meta.request.question);
  };

  if (message.meta?.typing) {
    // Progress widget is shown separately in ChatWindow
    return null;
  }

  if (message.meta?.error) {
    return (
      <div className="message-row assistant">
        <div className="message-inner">
          <div className="avatar assistant">MA</div>
          <div className="bubble assistant error-bubble">
            <strong>Request failed</strong>
            <p className="error-text">{message.meta.error}</p>
            {message.meta.request?.question && (
              <button className="retry-btn" onClick={handleRetry} type="button">
                ↻ Try again
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (message.role === "user") {
    return (
      <div className="message-row user">
        <div className="message-inner">
          <div className="avatar user">You</div>
          <div className="bubble user">{message.content}</div>
        </div>
      </div>
    );
  }

  const confidence = response ? getConfidenceLevel(response.synthesis.confidence_score) : null;

  return (
    <div className="message-row assistant">
      <div className="message-inner">
        <div className="avatar assistant">MA</div>
        <div className="bubble assistant">
          <div className="bubble-header">
            <span className="timestamp">{new Date(message.createdAt).toLocaleTimeString()}</span>
            <div className="actions">
              <button className="icon-btn" onClick={() => setMenuOpen((prev) => !prev)} type="button">
                ⋯
              </button>
              {menuOpen && (
                <div className="menu">
                  <button type="button" onClick={handleCopy}>
                    Copy response
                  </button>
                  <button type="button" onClick={() => { setShowRaw((prev) => !prev); setMenuOpen(false); }}>
                    {showRaw ? "Hide JSON" : "Show JSON"}
                  </button>
                </div>
              )}
            </div>
          </div>

          {response ? (
            <div className="assistant-content">
              <p>{summary}</p>

              {confidence && (
                <div className="confidence-row">
                  <span className={`confidence-badge ${confidence.className}`}>
                    Confidence: {response.synthesis.confidence_score}% ({confidence.label})
                  </span>
                  {response.papers.length > 0 && (
                    <span className="confidence-meta">
                      Based on {response.papers.length} paper{response.papers.length !== 1 ? "s" : ""}
                    </span>
                  )}
                </div>
              )}

              <section>
                <h4>Key Findings</h4>
                <ul>
                  {response.synthesis.final_answer.map((bullet, index) => (
                    <li key={index}>{bullet}</li>
                  ))}
                </ul>
              </section>

              {response.papers.length > 0 && (
                <section>
                  <h4>Sources ({response.papers.length})</h4>
                  <ul>
                    {response.papers.map((paper) => (
                      <li key={paper.paper_id}>
                        <strong>{paper.title}</strong>
                        <span className="paper-meta">
                          — {paper.authors.slice(0, 2).join(", ")}{paper.authors.length > 2 ? " et al." : ""} {paper.year ?? ""}
                        </span>
                        {paper.url && (
                          <a href={paper.url} target="_blank" rel="noreferrer" className="paper-link">
                            View paper ↗
                          </a>
                        )}
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {response.extractions.length > 0 && (
                <Collapsible title={`Evidence Table (${response.extractions.length} studies)`}>
                  <div className="table">
                    <div className="table-row header">
                      <div>Paper</div>
                      <div>Study Type</div>
                      <div>Effect</div>
                      <div>Bias Risk</div>
                      <div>Key Finding</div>
                    </div>
                    {response.extractions.map((extraction) => {
                      const critique = response.critiques.find((item) => item.paper_id === extraction.paper_id);
                      const paper = response.papers.find((p) => p.paper_id === extraction.paper_id);
                      return (
                        <div className="table-row" key={extraction.paper_id}>
                          <div>{paper?.title?.slice(0, 40) ?? extraction.paper_id.slice(0, 8)}...</div>
                          <div>{extraction.study_type.replace("_", " ")}</div>
                          <div>{extraction.effect_direction ?? "—"}</div>
                          <div className={`risk ${critique?.risk_of_bias ?? "unknown"}`}>
                            {critique?.risk_of_bias ?? "—"}
                          </div>
                          <div>{extraction.claim_summary.slice(0, 60)}...</div>
                        </div>
                      );
                    })}
                  </div>
                </Collapsible>
              )}

              <Collapsible title="Pipeline Details">
                <div className="agent-stepper">
                  {["Retriever", "Extractor", "Critic", "Synthesizer", "Referee"].map((label, index, arr) => (
                    <div className="agent-stepper-item" key={label}>
                      <div className="agent-stepper-icon">
                        <span className="agent-dot" />
                        {index < arr.length - 1 && <span className="agent-line" />}
                      </div>
                      <div className="agent-stepper-label">{label}</div>
                    </div>
                  ))}
                </div>
                <div className="agent-steps">
                  {[
                    { label: "Retriever", data: response.logs?.retrieve ?? { papers: response.papers.length } },
                    { label: "Extractor", data: response.logs?.extract ?? response.extractions },
                    { label: "Critic", data: response.logs?.critique ?? response.critiques },
                    { label: "Synthesizer", data: response.logs?.synthesize ?? response.synthesis },
                    { label: "Referee", data: response.logs?.verify ?? response.verification },
                  ].map((step) => (
                    <div key={step.label} className="agent-step">
                      <h5>{step.label}</h5>
                      <pre className="json-block">{JSON.stringify(step.data, null, 2)}</pre>
                    </div>
                  ))}
                </div>
              </Collapsible>

              {showRaw && <pre className="json-block">{JSON.stringify(response, null, 2)}</pre>}
            </div>
          ) : (
            <p>{message.content}</p>
          )}
        </div>
      </div>
    </div>
  );
}
