import { AGENT_LABELS, AGENT_NAMES, type AgentProgress, type AgentStatus } from "../types";

type Props = {
  progress: AgentProgress;
};

function StatusIcon({ status }: { status: AgentStatus }) {
  switch (status) {
    case "completed":
      return (
        <svg viewBox="0 0 24 24" className="status-icon" fill="none" stroke="currentColor" strokeWidth="3">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      );
    case "failed":
      return (
        <svg viewBox="0 0 24 24" className="status-icon" fill="none" stroke="currentColor" strokeWidth="3">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      );
    case "running":
      return <div className="status-spinner" />;
    default:
      return null;
  }
}

export default function AgentProgressWidget({ progress }: Props) {
  return (
    <div className="agent-progress-widget">
      <div className="progress-list">
        {AGENT_NAMES.map((agent, idx) => (
          <div key={agent} className="progress-item">
            <div className="progress-track">
              <div className={`progress-circle ${progress[agent]}`}>
                <StatusIcon status={progress[agent]} />
              </div>
              {idx < AGENT_NAMES.length - 1 && (
                <div className={`progress-line ${progress[AGENT_NAMES[idx + 1]] !== "pending" ? "active" : ""}`} />
              )}
            </div>
            <span className={`progress-label ${progress[agent]}`}>{AGENT_LABELS[agent]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
