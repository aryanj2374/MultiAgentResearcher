import { useEffect, useRef, useState } from "react";
import type { Conversation } from "../types";
import Header from "./Header";
import Message from "./Message";

type ChatWindowProps = {
  conversation: Conversation | null;
  loading: boolean;
  onToggleSidebar: () => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onRetry: (messageId: string, question: string) => void;
};

export default function ChatWindow({
  conversation,
  loading,
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
        subtitle="Local conversations"
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
          </div>
        ) : (
          <div className="empty-state">
            <h1>Ask a research question</h1>
            <p>Retrieve papers, extract evidence, critique quality, and synthesize a grounded answer.</p>
          </div>
        )}
      </div>
    </div>
  );
}
