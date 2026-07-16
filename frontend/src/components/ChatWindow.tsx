import { useEffect, useRef, useState } from "react";
import type { AgentProgress, Conversation, SubQuestionProgress } from "../types";
import AgentProgressWidget from "./AgentProgressWidget";
import Header from "./Header";
import Message from "./Message";

type ChatWindowProps = {
  conversation: Conversation | null;
  loading: boolean;
  agentProgress: AgentProgress | null;
  subQuestionProgress?: SubQuestionProgress[] | null;
  isDeepResearch?: boolean;
  onToggleSidebar: () => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onRetry: (messageId: string, question: string) => void;
  onSuggestionSelect: (question: string) => void;
};

const SUGGESTIONS = [
  {
    label: "Sleep & cognition",
    question: "How does sleep deprivation affect memory and cognitive performance in adults?",
    icon: "moon",
  },
  {
    label: "Compare treatments",
    question: "Is cognitive behavioral therapy more effective than medication for chronic insomnia?",
    icon: "compare",
  },
  {
    label: "Nutrition evidence",
    question: "What does the evidence say about omega-3 supplementation and cardiovascular health?",
    icon: "leaf",
  },
];

export default function ChatWindow({
  conversation,
  loading,
  agentProgress,
  subQuestionProgress,
  isDeepResearch,
  onToggleSidebar,
  theme,
  onToggleTheme,
  onRetry,
  onSuggestionSelect,
}: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    setAutoScroll(true);
  }, [conversation?.id]);

  useEffect(() => {
    if (!scrollRef.current || !autoScroll) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [conversation?.messages, loading, autoScroll]);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 80;
    setAutoScroll(atBottom);
  };

  const scrollToBottom = () => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    setAutoScroll(true);
  };

  return (
    <div className="chat-window">
      <Header
        title={conversation?.title ?? "New chat"}
        theme={theme}
        onToggleTheme={onToggleTheme}
        onToggleSidebar={onToggleSidebar}
      />

      <div className="message-list" ref={scrollRef} onScroll={handleScroll}>
        {conversation?.messages.length ? (
          <div className="message-column">
            {conversation.messages.map((message) => (
              <Message
                key={message.id}
                message={message}
                onRetry={(id, question) => onRetry(id, question)}
              />
            ))}
            {agentProgress && (
              <div className="message-row assistant progress-message">
                <div className="message-inner">
                  <div className="avatar assistant" aria-hidden="true">R</div>
                  <div className="bubble assistant">
                    <AgentProgressWidget 
                      progress={agentProgress} 
                      subQuestions={subQuestionProgress ?? undefined}
                      isDeepResearch={isDeepResearch}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-glow" aria-hidden="true" />
            <div className="empty-content">
              <div className="empty-mark" aria-hidden="true">
                <span className="empty-orbit"><i /></span>
                <span>R</span>
              </div>
              <span className="empty-eyebrow"><i aria-hidden="true" /> Evidence-first research</span>
              <h1>Turn a question into a<br /><em>clear research brief.</em></h1>
              <p>
                Search the literature, compare findings, assess bias, and verify citations—guided by six specialized agents.
              </p>

              <div className="suggestion-grid" aria-label="Example research questions">
                {SUGGESTIONS.map((suggestion, index) => (
                  <button
                    key={suggestion.label}
                    type="button"
                    className="suggestion-card"
                    style={{ "--delay": `${index * 70}ms` } as React.CSSProperties}
                    onClick={() => onSuggestionSelect(suggestion.question)}
                  >
                    <span className={`suggestion-icon ${suggestion.icon}`} aria-hidden="true">
                      {suggestion.icon === "moon" && <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M20 15.2A8.5 8.5 0 0 1 8.8 4a8.5 8.5 0 1 0 11.2 11.2Z" /></svg>}
                      {suggestion.icon === "compare" && <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M8 7h11m0 0-3-3m3 3-3 3M16 17H5m0 0 3 3m-3-3 3-3" /></svg>}
                      {suggestion.icon === "leaf" && <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M19.5 4.5C13 4.5 7 7 5 13c-1 3 1 6 4 6 6 0 9-6 10.5-14.5Z" /><path d="M5 20c2-5 5-8 10-10" /></svg>}
                    </span>
                    <span>
                      <strong>{suggestion.label}</strong>
                      <small>{suggestion.question}</small>
                    </span>
                    <svg className="suggestion-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
                  </button>
                ))}
              </div>

              <div className="capability-row" aria-label="Research capabilities">
                <span><i aria-hidden="true" /> Search papers</span>
                <span><i aria-hidden="true" /> Assess evidence</span>
                <span><i aria-hidden="true" /> Verify citations</span>
              </div>
            </div>
          </div>
        )}
        {!autoScroll && conversation?.messages.length ? (
          <button className="scroll-to-bottom" type="button" onClick={scrollToBottom} aria-label="Scroll to latest message">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg>
            Latest
          </button>
        ) : null}
      </div>
    </div>
  );
}
