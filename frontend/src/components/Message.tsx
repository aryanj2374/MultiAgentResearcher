import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { BackendResponse, Message } from "../types";
import Collapsible from "./Collapsible";

type MessageProps = {
  message: Message;
  onRetry?: (messageId: string, question: string) => void;
};

const PIPELINE_STEPS = ["Plan", "Retrieve", "Extract", "Critique", "Synthesize", "Verify"];

function buildSummary(response: BackendResponse | undefined, content: string): string {
  if (response?.synthesis?.evidence_consensus) return response.synthesis.evidence_consensus;
  if (response?.synthesis?.final_answer?.length) return response.synthesis.final_answer[0];
  return content || "Summary unavailable.";
}

function getConfidenceLevel(score: number): { label: string; className: string } {
  if (score >= 75) return { label: "High confidence", className: "high" };
  if (score >= 45) return { label: "Moderate confidence", className: "medium" };
  return { label: "Low confidence", className: "low" };
}

function formatLabel(value: string | null | undefined): string {
  if (!value) return "Unknown";
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter: string) => letter.toUpperCase());
}

function InlineText({ text }: { text: string }) {
  const parts = text.split(/(\*\*.*?\*\*|\[[^\]]+\])/g).filter(Boolean);
  return (
    <>
      {parts.map((part, index) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
        }
        if (part.startsWith("[") && part.endsWith("]")) {
          return <span className="inline-citation" key={`${part}-${index}`}>{part}</span>;
        }
        return <span key={`${part}-${index}`}>{part}</span>;
      })}
    </>
  );
}

export default function MessageView({ message, onRetry }: MessageProps) {
  const [showRaw, setShowRaw] = useState(false);
  const [copied, setCopied] = useState(false);
  const response = message.meta?.response;
  const summary = useMemo(() => buildSummary(response, message.content), [response, message.content]);
  const rawPanelRef = useRef<HTMLDivElement | null>(null);
  const rawPanelId = `raw-response-${message.id}`;

  // Toggling the panel changes nothing the chat's auto-scroll effect watches,
  // so nothing brings it into view on its own. It now renders directly under
  // the toggle, and "nearest" only nudges the view when the card sits low
  // enough that the panel would still open past the bottom edge.
  useEffect(() => {
    if (!showRaw) return;
    rawPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [showRaw]);

  const handleCopy = async () => {
    const textParts: string[] = [];
    if (response) {
      textParts.push("Research brief\n" + response.synthesis.final_answer.join("\n"));
      textParts.push("Evidence consensus\n" + response.synthesis.evidence_consensus);
      if (response.synthesis.top_limitations_overall.length) {
        textParts.push("Limitations\n" + response.synthesis.top_limitations_overall.join("\n"));
      }
    } else {
      textParts.push(message.content);
    }

    try {
      await navigator.clipboard.writeText(textParts.join("\n\n"));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  const handleRetry = () => {
    if (!onRetry || !message.meta?.request?.question) return;
    onRetry(message.id, message.meta.request.question);
  };

  if (message.meta?.typing) return null;

  if (message.meta?.error) {
    return (
      <div className="message-row assistant">
        <div className="message-inner">
          <div className="avatar assistant" aria-hidden="true">R</div>
          <div className="bubble assistant error-bubble" role="alert">
            <span className="error-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 8v5M12 17h.01" /><circle cx="12" cy="12" r="9" /></svg>
            </span>
            <div>
              <strong>Research paused</strong>
              <p className="error-text">{message.meta.error}</p>
              {message.meta.request?.question && (
                <button className="retry-btn" onClick={handleRetry} type="button">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5" /><path d="M6.1 9a7 7 0 0 1 11.7-2.6L20 9M4 15l2.2 2.6A7 7 0 0 0 17.9 15" /></svg>
                  Try again
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (message.role === "user") {
    return (
      <div className="message-row user">
        <div className="message-inner">
          <div className="avatar user" aria-hidden="true">You</div>
          <div className="bubble user">{message.content}</div>
        </div>
      </div>
    );
  }

  const confidence = response ? getConfidenceLevel(response.synthesis.confidence_score) : null;
  const time = new Date(message.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

  return (
    <div className="message-row assistant result-message">
      <div className="message-inner">
        <div className="avatar assistant" aria-hidden="true">R</div>
        <article className="bubble assistant result-card">
          <div className="result-header">
            <div>
              <span className="result-kicker"><i aria-hidden="true" /> Research complete</span>
              <h3>Evidence brief</h3>
              <span className="timestamp">Prepared at {time}</span>
            </div>
            <div className="result-actions">
              <button className={`result-action ${copied ? "success" : ""}`} onClick={handleCopy} type="button" aria-label="Copy research brief" title="Copy research brief">
                {copied ? (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="m5 12 4 4L19 6" /></svg>
                ) : (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="2" /><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" /></svg>
                )}
                <span>{copied ? "Copied" : "Copy"}</span>
              </button>
              {response && (
                <button className={`result-action ${showRaw ? "active" : ""}`} onClick={() => setShowRaw((current) => !current)} type="button" aria-expanded={showRaw} aria-controls={rawPanelId} aria-label="Toggle raw response" title="Toggle raw response">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="m8 9-3 3 3 3M16 9l3 3-3 3M14 5l-4 14" /></svg>
                  <span>Raw</span>
                </button>
              )}
            </div>
          </div>

          {response ? (
            <div className="assistant-content">
              {/* Sits directly under the toggle rather than at the foot of the
                  card. The card is several thousand pixels tall with a full
                  evidence table, so a panel appended at the end opens entirely
                  off-screen and the button looks like it does nothing. */}
              {showRaw && (
                <div className="raw-response" id={rawPanelId} ref={rawPanelRef}>
                  <div>
                    <span>Raw response</span>
                    <small>{response.logs_trimmed ? "Developer view · logs dropped when saved" : "Developer view"}</small>
                  </div>
                  <pre className="json-block">{JSON.stringify(response, null, 2)}</pre>
                </div>
              )}

              <div className="consensus-card">
                <div className="consensus-label">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M4 19V9M10 19V5M16 19v-7M22 19V3" /></svg>
                  Evidence consensus
                </div>
                <p><InlineText text={summary} /></p>
              </div>

              {confidence && (
                <div className="metric-grid">
                  <div className="metric-card confidence-metric">
                    <span
                      className={`confidence-ring ${confidence.className}`}
                      style={{ "--score": `${response.synthesis.confidence_score * 3.6}deg` } as CSSProperties}
                      aria-hidden="true"
                    >
                      <strong>{response.synthesis.confidence_score}</strong>
                    </span>
                    <span><small>Confidence</small><strong>{confidence.label}</strong></span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-icon papers" aria-hidden="true">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M6 3h9l4 4v14H6z" /><path d="M14 3v5h5M9 13h7M9 17h5" /></svg>
                    </span>
                    <span><small>Evidence base</small><strong>{response.papers.length} paper{response.papers.length === 1 ? "" : "s"}</strong></span>
                  </div>
                  <div className="metric-card">
                    <span className={`metric-icon ${response.verification.passed ? "verified" : "review"}`} aria-hidden="true">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" /><path d="m9 12 2 2 4-4" /></svg>
                    </span>
                    <span><small>Citation check</small><strong>{response.verification.passed ? "Verified" : "Needs review"}</strong></span>
                  </div>
                </div>
              )}

              <section className="result-section findings-section">
                <div className="section-heading">
                  <span className="section-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M9 18h6M10 22h4M8.5 14.5A7 7 0 1 1 16 14c-.8.7-1 1.3-1 2H9c0-.8-.2-1.1-.5-1.5Z" /></svg>
                  </span>
                  <div><span>Answer</span><h4>Key findings</h4></div>
                </div>
                <ol className="findings-list">
                  {response.synthesis.final_answer.map((bullet, index) => (
                    <li key={`${index}-${bullet.slice(0, 20)}`} style={{ "--item-index": index } as CSSProperties}>
                      <span className="finding-number">{String(index + 1).padStart(2, "0")}</span>
                      <p><InlineText text={bullet} /></p>
                    </li>
                  ))}
                </ol>
              </section>

              {response.synthesis.top_limitations_overall.length > 0 && (
                <section className="limitations-card">
                  <div className="limitations-heading">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M12 9v4M12 17h.01" /><path d="M10.3 3.7 2.5 17.2A2 2 0 0 0 4.2 20h15.6a2 2 0 0 0 1.7-2.8L13.7 3.7a2 2 0 0 0-3.4 0Z" /></svg>
                    <div><span>Read with context</span><h4>Important limitations</h4></div>
                  </div>
                  <ul>
                    {response.synthesis.top_limitations_overall.map((limitation, index) => (
                      <li key={`${index}-${limitation.slice(0, 20)}`}><InlineText text={limitation} /></li>
                    ))}
                  </ul>
                </section>
              )}

              {response.papers.length > 0 && (
                <section className="result-section sources-section">
                  <div className="section-heading">
                    <span className="section-icon" aria-hidden="true">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M4 5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2z" /><path d="M8 7h8M8 11h6" /></svg>
                    </span>
                    <div><span>Literature</span><h4>Sources <small>{response.papers.length}</small></h4></div>
                  </div>
                  <div className="source-list">
                    {response.papers.map((paper, index) => (
                      <article className="source-card" key={`${paper.paper_id}-${index}`}>
                        <span className="source-index">{String(index + 1).padStart(2, "0")}</span>
                        <div className="source-copy">
                          <strong>{paper.title}</strong>
                          <span>{paper.authors.slice(0, 2).join(", ") || "Unknown authors"}{paper.authors.length > 2 ? " et al." : ""} {paper.year ? `· ${paper.year}` : ""}</span>
                          {paper.venue && <small>{paper.venue}</small>}
                        </div>
                        {paper.url && (
                          <a href={paper.url} target="_blank" rel="noreferrer" className="paper-link" aria-label={`Open ${paper.title}`}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M14 5h5v5M10 14 19 5M19 14v5H5V5h5" /></svg>
                          </a>
                        )}
                      </article>
                    ))}
                  </div>
                </section>
              )}

              {response.extractions.length > 0 && (
                <Collapsible title={`Explore evidence table · ${response.extractions.length} studies`}>
                  <div className="table">
                    <div className="table-row header">
                      <div>Paper</div><div>Study type</div><div>Effect</div><div>Bias risk</div><div>Key finding</div>
                    </div>
                    {response.extractions.map((extraction, index) => {
                      const critique = response.critiques.find((item) => item.paper_id === extraction.paper_id);
                      const paper = response.papers.find((item) => item.paper_id === extraction.paper_id);
                      return (
                        <div className="table-row" key={`${extraction.paper_id}-${index}`}>
                          <div data-label="Paper">{paper?.title ?? extraction.paper_id.slice(0, 8)}</div>
                          <div data-label="Study type">{formatLabel(extraction.study_type)}</div>
                          <div data-label="Effect"><span className={`effect ${extraction.effect_direction ?? "unclear"}`}>{formatLabel(extraction.effect_direction)}</span></div>
                          <div data-label="Bias risk"><span className={`risk ${critique?.risk_of_bias ?? "unknown"}`}>{formatLabel(critique?.risk_of_bias)}</span></div>
                          <div data-label="Key finding">{extraction.claim_summary}</div>
                        </div>
                      );
                    })}
                  </div>
                </Collapsible>
              )}

              <Collapsible title="How this answer was built">
                <div className="pipeline-overview">
                  {PIPELINE_STEPS.map((label, index) => (
                    <div className="pipeline-stage" key={label}>
                      <span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="m6 12 4 4 8-8" /></svg></span>
                      <strong>{label}</strong>
                      {index < PIPELINE_STEPS.length - 1 && <i aria-hidden="true" />}
                    </div>
                  ))}
                </div>
                <p className="pipeline-note">Six specialized agents planned the search, evaluated the retrieved evidence, synthesized the findings, and checked citation consistency.</p>
              </Collapsible>

            </div>
          ) : (
            <p>{message.content}</p>
          )}
        </article>
      </div>
    </div>
  );
}
