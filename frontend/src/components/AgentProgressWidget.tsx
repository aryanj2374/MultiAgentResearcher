import { AGENT_LABELS, AGENT_NAMES, type AgentProgress, type AgentStatus } from "../types";

type Props = {
  progress: AgentProgress;
};

function StatusIcon({ status }: { status: AgentStatus }) {
  if (status === "completed") {
    return (
      <svg viewBox="0 0 24 24" className="status-check" fill="none" stroke="currentColor" strokeWidth="2.5">
        <polyline points="20 6 9 17 4 12" />
      </svg>
    );
  }
  if (status === "failed") {
    return (
      <svg viewBox="0 0 24 24" className="status-x" fill="none" stroke="currentColor" strokeWidth="2.5">
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
      </svg>
    );
  }
  if (status === "running") {
    return <div className="status-pulse" />;
  }
  return null;
}

export default function AgentProgressWidget({ progress }: Props) {
  const getStepNumber = (agent: string): number => {
    return AGENT_NAMES.indexOf(agent as typeof AGENT_NAMES[number]) + 1;
  };

  return (
    <div className="progress-widget">
      <div className="progress-header">
        <span className="progress-title">Processing</span>
        <span className="progress-dots">
          <span />
          <span />
          <span />
        </span>
      </div>
      <div className="progress-steps">
        {AGENT_NAMES.map((agent, idx) => {
          const status = progress[agent];
          const isLast = idx === AGENT_NAMES.length - 1;
          
          return (
            <div key={agent} className={`progress-step ${status}`}>
              <div className="step-indicator">
                <div className={`step-circle ${status}`}>
                  {status === "pending" && <span className="step-number">{getStepNumber(agent)}</span>}
                  <StatusIcon status={status} />
                </div>
                {!isLast && <div className={`step-line ${progress[AGENT_NAMES[idx + 1]] !== "pending" ? "active" : ""}`} />}
              </div>
              <div className="step-content">
                <span className={`step-label ${status}`}>{AGENT_LABELS[agent]}</span>
                {status === "running" && <span className="step-status">In progress...</span>}
                {status === "completed" && <span className="step-status done">Done</span>}
                {status === "failed" && <span className="step-status error">Failed</span>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
