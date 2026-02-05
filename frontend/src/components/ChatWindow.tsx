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
};

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
}: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);

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

  return (
    <div className="chat-window">
      <Header
        title={conversation?.title ?? "New chat"}
        subtitle="Multi-Agent Research"
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
              <div className="message-row assistant">
                <div className="message-inner">
                  <div className="avatar assistant">MA</div>
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
            <div className="empty-logo">MA</div>
            <h1>What would you like to research?</h1>
            <p>
              Ask any scientific or medical research question. I'll find relevant papers, 
              extract key evidence, assess study quality, and synthesize a grounded answer with citations.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
