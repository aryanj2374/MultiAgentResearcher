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
    return (
      <div className="message-row assistant">
        <div className="message-inner">
          <div className="avatar">MA</div>
          <div className="bubble assistant">
            <div className="typing-dots">
              <span />
              <span />
              <span />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (message.meta?.error) {
    return (
      <div className="message-row assistant">
        <div className="message-inner">
          <div className="avatar">MA</div>
          <div className="bubble assistant error-bubble">
            <strong>Request failed.</strong>
            <p>{message.meta.error}</p>
            {message.meta.request?.question && (
              <button className="retry-btn" onClick={handleRetry} type="button">
                Retry
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
          <div className="bubble user">{message.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="message-row assistant">
      <div className="message-inner">
        <div className="avatar">MA</div>
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
                    Copy
                  </button>
                  <button type="button" onClick={() => setShowRaw((prev) => !prev)}>
                    {showRaw ? "Hide raw JSON" : "Show raw JSON"}
                  </button>
                </div>
              )}
            </div>
          </div>

          {response ? (
            <div className="assistant-content">
              <p>{summary}</p>

              <section>
                <h4>Answer</h4>
                <ul>
                  {response.synthesis.final_answer.map((bullet, index) => (
                    <li key={index}>{bullet}</li>
                  ))}
                </ul>
              </section>

              <section>
                <h4>Evidence</h4>
                <ul>
                  {response.papers.map((paper) => (
                    <li key={paper.paper_id}>
                      <strong>{paper.title}</strong> — {paper.authors.join(", ")} {paper.year ?? ""}
                      {paper.url && (
                        <span>
                          {" "}
                          <a href={paper.url} target="_blank" rel="noreferrer">
                            Link
                          </a>
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </section>

              <Collapsible title="Evidence table">
                <div className="table">
                  <div className="table-row header">
                    <div>Paper</div>
                    <div>Study Type</div>
                    <div>Effect</div>
                    <div>Bias</div>
                    <div>Limitations</div>
                  </div>
                  {response.extractions.map((extraction) => {
                    const critique = response.critiques.find((item) => item.paper_id === extraction.paper_id);
                    return (
                      <div className="table-row" key={extraction.paper_id}>
                        <div>{extraction.paper_id.slice(0, 8)}</div>
                        <div>{extraction.study_type}</div>
                        <div>{extraction.effect_direction ?? "-"}</div>
                        <div>{critique?.risk_of_bias ?? "-"}</div>
                        <div>{extraction.limitations?.length ? extraction.limitations.join("; ") : "-"}</div>
                      </div>
                    );
                  })}
                </div>
              </Collapsible>

              <Collapsible title="Agent steps">
                <div className="agent-steps">
                  {[
                    { label: "Retriever", data: response.logs?.retrieve ?? response.papers },
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
