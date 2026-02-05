import { AGENT_LABELS, AGENT_NAMES, type AgentProgress, type AgentStatus, type SubQuestionProgress } from "../types";

type Props = {
  progress: AgentProgress;
  subQuestions?: SubQuestionProgress[];
  isDeepResearch?: boolean;
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

function getStepNumber(agent: string): number {
  return AGENT_NAMES.indexOf(agent as typeof AGENT_NAMES[number]) + 1;
}

// Check if all sub-questions are completed
function allSubQuestionsCompleted(subQuestions?: SubQuestionProgress[]): boolean {
  if (!subQuestions || subQuestions.length === 0) return false;
  return subQuestions.every((sq) => sq.status === "completed");
}

export default function AgentProgressWidget({ progress, subQuestions, isDeepResearch }: Props) {
  const subQsCompleted = allSubQuestionsCompleted(subQuestions);
  
  return (
    <div className="progress-widget">
      <div className="progress-header">
        <span className="progress-title">
          {isDeepResearch ? "Deep Research" : "Processing"}
        </span>
        <span className="progress-dots">
          <span />
          <span />
          <span />
        </span>
      </div>
      
      <div className="progress-steps">
        {/* Deep Research Mode: Planner with nested sub-questions */}
        {isDeepResearch && (
          <>
            {/* Planner Step */}
            <div className={`progress-step has-children ${progress.planner}`}>
              <div className="step-indicator">
                <div className={`step-circle ${progress.planner}`}>
                  {progress.planner === "pending" && <span className="step-number">1</span>}
                  <StatusIcon status={progress.planner} />
                </div>
                {/* Main line continues through sub-questions */}
                <div className={`step-line extended ${subQsCompleted ? "active" : ""}`} />
              </div>
              <div className="step-content compact">
                <span className={`step-label ${progress.planner}`}>Planner</span>
                {progress.planner === "running" && <span className="step-status">Analyzing question...</span>}
                {progress.planner === "completed" && (
                  <span className="step-status done">
                    {subQuestions && subQuestions.length > 0 
                      ? `Decomposed into ${subQuestions.length} sub-questions` 
                      : "Decomposed into sub-questions"}
                  </span>
                )}
              </div>
            </div>

            {/* Nested Sub-questions as branches */}
            {subQuestions && subQuestions.length > 0 && (
              <div className="sub-questions-branch">
                {subQuestions.map((sq, idx) => {
                  const isLast = idx === subQuestions.length - 1;
                  return (
                    <div key={idx} className={`sub-question-node ${sq.status}`}>
                      <div className="sub-node-indicator">
                        {/* Vertical connector from main line */}
                        <div className="branch-connector">
                          <div className={`branch-vertical ${idx === 0 ? "first" : ""} ${isLast ? "last" : ""}`} />
                          <div className="branch-horizontal" />
                        </div>
                        {/* Sub-question circle */}
                        <div className={`sub-circle ${sq.status}`}>
                          {sq.status === "pending" && <span className="sub-number">{idx + 1}</span>}
                          <StatusIcon status={sq.status} />
                        </div>
                      </div>
                      <div className="sub-node-content">
                        <span className={`sub-node-label ${sq.status}`}>{sq.sub_question}</span>
                        {sq.status === "running" && <span className="sub-node-status">Researching...</span>}
                        {sq.status === "completed" && (
                          <span className="sub-node-status done">
                            {sq.papers_found !== undefined ? `${sq.papers_found} papers` : "Done"}
                          </span>
                        )}
                        {sq.status === "failed" && <span className="sub-node-status error">Failed</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Synthesizer Step */}
            <div className={`progress-step ${progress.synthesizer}`}>
              <div className="step-indicator">
                <div className={`step-circle ${progress.synthesizer}`}>
                  {progress.synthesizer === "pending" && <span className="step-number">2</span>}
                  <StatusIcon status={progress.synthesizer} />
                </div>
                <div className={`step-line ${progress.referee !== "pending" ? "active" : ""}`} />
              </div>
              <div className="step-content">
                <span className={`step-label ${progress.synthesizer}`}>Synthesizer</span>
                {progress.synthesizer === "running" && <span className="step-status">Combining findings...</span>}
                {progress.synthesizer === "completed" && <span className="step-status done">Done</span>}
              </div>
            </div>

            {/* Referee Step */}
            <div className={`progress-step ${progress.referee}`}>
              <div className="step-indicator">
                <div className={`step-circle ${progress.referee}`}>
                  {progress.referee === "pending" && <span className="step-number">3</span>}
                  <StatusIcon status={progress.referee} />
                </div>
              </div>
              <div className="step-content">
                <span className={`step-label ${progress.referee}`}>Referee</span>
                {progress.referee === "running" && <span className="step-status">Verifying...</span>}
                {progress.referee === "completed" && <span className="step-status done">Done</span>}
              </div>
            </div>
          </>
        )}

        {/* Standard Pipeline Mode */}
        {!isDeepResearch && AGENT_NAMES.map((agent, idx) => {
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
