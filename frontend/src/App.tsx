import { useCallback, useEffect, useMemo, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import Composer from "./components/Composer";
import Sidebar from "./components/Sidebar";
import { askQuestion } from "./lib/api";
import { loadConversations, loadTheme, saveConversations, saveTheme } from "./lib/storage";
import type { BackendResponse, Conversation, Message, Theme } from "./types";

function createId() {
  return crypto.randomUUID();
}

function now() {
  return new Date().toISOString();
}

function buildTitle(messages: Message[]): string {
  const firstUser = messages.find((msg) => msg.role === "user");
  if (!firstUser) return "New chat";
  return firstUser.content.slice(0, 56);
}

function buildSummary(content: string, response?: BackendResponse): string {
  if (response?.synthesis?.evidence_consensus) return response.synthesis.evidence_consensus;
  if (response?.synthesis?.final_answer?.length) return response.synthesis.final_answer[0];
  return content;
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [composerText, setComposerText] = useState("");
  const [theme, setTheme] = useState<Theme>("dark");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    const stored = loadConversations();
    setConversations(stored);
    if (stored[0]) setActiveId(stored[0].id);
    const storedTheme = loadTheme();
    if (storedTheme) setTheme(storedTheme);
  }, []);

  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    saveTheme(theme);
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const activeConversation = useMemo(
    () => conversations.find((conv) => conv.id === activeId) ?? null,
    [conversations, activeId]
  );

  const updateConversation = useCallback(
    (id: string, updater: (conv: Conversation) => Conversation) => {
      setConversations((prev) => prev.map((conv) => (conv.id === id ? updater(conv) : conv)));
    },
    []
  );

  const handleNewChat = useCallback(() => {
    const convo: Conversation = {
      id: createId(),
      title: "New chat",
      createdAt: now(),
      messages: [],
    };
    setConversations((prev) => [convo, ...prev]);
    setActiveId(convo.id);
  }, []);

  const handleSelectConversation = useCallback((id: string) => {
    setActiveId(id);
    setSidebarOpen(false);
  }, []);

  const appendTypingMessage = useCallback((convoId: string, question: string) => {
    const typingMessage: Message = {
      id: createId(),
      role: "assistant",
      content: "",
      createdAt: now(),
      meta: { typing: true, request: { question } },
    };

    updateConversation(convoId, (conv) => {
      const messages = [...conv.messages, typingMessage];
      return { ...conv, messages, title: buildTitle(messages) };
    });

    return typingMessage.id;
  }, [updateConversation]);

  const handleSend = useCallback(async () => {
    if (!composerText.trim()) return;

    let convoId = activeId;
    if (!convoId) {
      const convo: Conversation = {
        id: createId(),
        title: "New chat",
        createdAt: now(),
        messages: [],
      };
      setConversations((prev) => [convo, ...prev]);
      convoId = convo.id;
      setActiveId(convo.id);
    }

    const question = composerText.trim();
    const userMessage: Message = {
      id: createId(),
      role: "user",
      content: question,
      createdAt: now(),
    };

    setComposerText("");
    setLoading(true);

    updateConversation(convoId, (conv) => {
      const messages = [...conv.messages, userMessage];
      return { ...conv, messages, title: buildTitle(messages) };
    });

    const typingId = appendTypingMessage(convoId, question);

    try {
      const response = await askQuestion(question);
      updateConversation(convoId, (conv) => {
        const messages = conv.messages.map((msg) =>
          msg.id === typingId
            ? {
                ...msg,
                meta: { response, request: { question } },
                content: buildSummary(question, response),
              }
            : msg
        );
        return { ...conv, messages, title: buildTitle(messages) };
      });
    } catch (error) {
      updateConversation(convoId, (conv) => {
        const messages = conv.messages.map((msg) =>
          msg.id === typingId
            ? {
                ...msg,
                meta: { error: error instanceof Error ? error.message : "Request failed", request: { question } },
                content: "",
              }
            : msg
        );
        return { ...conv, messages, title: buildTitle(messages) };
      });
    } finally {
      setLoading(false);
    }
  }, [activeId, appendTypingMessage, composerText, updateConversation]);

  const handleRetry = useCallback(
    async (messageId: string, question: string) => {
      if (!activeId) return;
      setLoading(true);
      updateConversation(activeId, (conv) => {
        const messages = conv.messages.map((msg) =>
          msg.id === messageId
            ? {
                ...msg,
                meta: { typing: true, request: { question } },
                content: "",
              }
            : msg
        );
        return { ...conv, messages };
      });

      try {
        const response = await askQuestion(question);
        updateConversation(activeId, (conv) => {
          const messages = conv.messages.map((msg) =>
            msg.id === messageId
              ? {
                  ...msg,
                  meta: { response, request: { question } },
                  content: buildSummary(question, response),
                }
              : msg
          );
          return { ...conv, messages, title: buildTitle(messages) };
        });
      } catch (error) {
        updateConversation(activeId, (conv) => {
          const messages = conv.messages.map((msg) =>
            msg.id === messageId
              ? {
                  ...msg,
                  meta: { error: error instanceof Error ? error.message : "Request failed", request: { question } },
                  content: "",
                }
              : msg
          );
          return { ...conv, messages };
        });
      } finally {
        setLoading(false);
      }
    },
    [activeId, updateConversation]
  );

  const handleToggleTheme = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);

  return (
    <div className="app-shell">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        collapsed={!sidebarOpen}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="main">
        <ChatWindow
          conversation={activeConversation}
          loading={loading}
          onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          onRetry={handleRetry}
        />

        <div className="composer-wrapper">
          <Composer value={composerText} loading={loading} onChange={setComposerText} onSend={handleSend} />
          <p className="composer-footer">Responses may be inaccurate. Verify critical details.</p>
        </div>
      </div>

      {sidebarOpen && <div className="overlay" onClick={() => setSidebarOpen(false)} />}
    </div>
  );
}
