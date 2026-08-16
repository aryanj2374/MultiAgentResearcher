import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import Composer from "./components/Composer";
import Sidebar from "./components/Sidebar";
import { askQuestionStream } from "./lib/api";
import { loadConversations, loadTheme, saveConversations, saveTheme } from "./lib/storage";
import type { AgentName, AgentProgress, BackendResponse, Conversation, Message, SubQuestionProgress, Theme } from "./types";
import { createInitialProgress } from "./types";

function createId() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function now() {
  return new Date().toISOString();
}

function createConversation(): Conversation {
  return {
    id: createId(),
    title: "New chat",
    createdAt: now(),
    messages: [],
  };
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
  const [initialState] = useState(() => {
    const stored = loadConversations();
    const initialConversations = stored.length > 0 ? stored : [createConversation()];
    return {
      conversations: initialConversations,
      activeId: initialConversations[0].id,
      theme: loadTheme() ?? ("dark" as Theme),
    };
  });
  const [conversations, setConversations] = useState<Conversation[]>(initialState.conversations);
  const [activeId, setActiveId] = useState<string | null>(initialState.activeId);
  const [loading, setLoading] = useState(false);
  const [composerText, setComposerText] = useState("");
  const [theme, setTheme] = useState<Theme>(initialState.theme);
  const [sidebarOpen, setSidebarOpen] = useState(
    () => typeof window === "undefined" || window.innerWidth > 768
  );
  const [agentProgress, setAgentProgress] = useState<AgentProgress | null>(null);
  const [subQuestionProgress, setSubQuestionProgress] = useState<SubQuestionProgress[] | null>(null);
  const [isDeepResearch, setIsDeepResearch] = useState(false);
  const [requestConversationId, setRequestConversationId] = useState<string | null>(null);
  const requestInFlightRef = useRef(false);

  const [storageNotice, setStorageNotice] = useState<string | null>(null);

  useEffect(() => {
    const result = saveConversations(conversations);
    if (result.ok) {
      setStorageNotice(null);
      return;
    }
    if (result.reason === "unavailable") {
      setStorageNotice("Browser storage is unavailable, so this history will not be saved.");
    } else if (result.dropped >= conversations.length) {
      setStorageNotice("This conversation is too large for browser storage and will not be saved.");
    } else {
      setStorageNotice(
        result.dropped === 1
          ? "Browser storage is full, so the oldest conversation was removed to save this one."
          : `Browser storage is full, so the ${result.dropped} oldest conversations were removed to save this one.`
      );
    }
  }, [conversations]);

  useEffect(() => {
    saveTheme(theme);
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (!sidebarOpen || !window.matchMedia("(max-width: 768px)").matches) return;

    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSidebarOpen(false);
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [sidebarOpen]);

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
    const convo = createConversation();
    setConversations((prev) => [convo, ...prev]);
    setActiveId(convo.id);
    if (window.matchMedia("(max-width: 768px)").matches) setSidebarOpen(false);
  }, []);

  const handleSelectConversation = useCallback((id: string) => {
    setActiveId(id);
    if (window.matchMedia("(max-width: 768px)").matches) setSidebarOpen(false);
  }, []);

  const handleSuggestionSelect = useCallback((question: string) => {
    setComposerText(question);
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
    if (requestInFlightRef.current || loading || !composerText.trim()) return;
    requestInFlightRef.current = true;

    let convoId = activeId;
    if (!convoId) {
      const convo = createConversation();
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
    setRequestConversationId(convoId);

    updateConversation(convoId, (conv) => {
      const messages = [...conv.messages, userMessage];
      return { ...conv, messages, title: buildTitle(messages) };
    });

    const typingId = appendTypingMessage(convoId, question);

    // Initialize progress tracking
    setAgentProgress(createInitialProgress());
    setSubQuestionProgress(null);
    setIsDeepResearch(false);

    try {
      let finalResponse: BackendResponse | null = null;

      for await (const event of askQuestionStream(question)) {
        if (event.type === "progress" && event.agent && event.status) {
          setAgentProgress((prev) => 
            prev ? { ...prev, [event.agent as AgentName]: event.status! } : prev
          );
        } else if (event.type === "deep_research_start" && event.sub_questions) {
          // Planner has decomposed the question
          setIsDeepResearch(true);
          setSubQuestionProgress(
            event.sub_questions.map((sq) => ({ sub_question: sq, status: "pending" as const }))
          );
        } else if (event.type === "sub_question_progress") {
          // Update individual sub-question progress
          setSubQuestionProgress((prev) => {
            if (!prev || event.index === undefined) return prev;
            return prev.map((sq, idx) =>
              idx === event.index
                ? { ...sq, status: event.status!, papers_found: event.papers_found }
                : sq
            );
          });
        } else if (event.type === "result" && event.data) {
          finalResponse = event.data;
        } else if (event.type === "error") {
          throw new Error(event.message || "Request failed");
        }
      }

      if (finalResponse) {
        updateConversation(convoId, (conv) => {
          const messages = conv.messages.map((msg) =>
            msg.id === typingId
              ? {
                  ...msg,
                  meta: { response: finalResponse!, request: { question } },
                  content: buildSummary(question, finalResponse!),
                }
              : msg
          );
          return { ...conv, messages, title: buildTitle(messages) };
        });
      }
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
      setAgentProgress(null);
      setSubQuestionProgress(null);
      setIsDeepResearch(false);
      setRequestConversationId(null);
      requestInFlightRef.current = false;
    }
  }, [activeId, appendTypingMessage, composerText, loading, updateConversation]);

  const handleRetry = useCallback(
    async (messageId: string, question: string) => {
      if (!activeId || requestInFlightRef.current || loading) return;
      requestInFlightRef.current = true;
      const convoId = activeId;
      setLoading(true);
      setRequestConversationId(convoId);
      updateConversation(convoId, (conv) => {
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

      // Initialize progress tracking
      setAgentProgress(createInitialProgress());
      setSubQuestionProgress(null);
      setIsDeepResearch(false);

      try {
        let finalResponse: BackendResponse | null = null;

        for await (const event of askQuestionStream(question)) {
          if (event.type === "progress" && event.agent && event.status) {
            setAgentProgress((prev) => 
              prev ? { ...prev, [event.agent as AgentName]: event.status! } : prev
            );
          } else if (event.type === "deep_research_start" && event.sub_questions) {
            setIsDeepResearch(true);
            setSubQuestionProgress(
              event.sub_questions.map((sq) => ({ sub_question: sq, status: "pending" as const }))
            );
          } else if (event.type === "sub_question_progress") {
            setSubQuestionProgress((prev) => {
              if (!prev || event.index === undefined) return prev;
              return prev.map((sq, idx) =>
                idx === event.index
                  ? { ...sq, status: event.status!, papers_found: event.papers_found }
                  : sq
              );
            });
          } else if (event.type === "result" && event.data) {
            finalResponse = event.data;
          } else if (event.type === "error") {
            throw new Error(event.message || "Request failed");
          }
        }

        if (finalResponse) {
          updateConversation(convoId, (conv) => {
            const messages = conv.messages.map((msg) =>
              msg.id === messageId
                ? {
                    ...msg,
                    meta: { response: finalResponse!, request: { question } },
                    content: buildSummary(question, finalResponse!),
                  }
                : msg
            );
            return { ...conv, messages, title: buildTitle(messages) };
          });
        }
      } catch (error) {
        updateConversation(convoId, (conv) => {
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
        setAgentProgress(null);
        setSubQuestionProgress(null);
        setIsDeepResearch(false);
        setRequestConversationId(null);
        requestInFlightRef.current = false;
      }
    },
    [activeId, loading, updateConversation]
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
          agentProgress={requestConversationId === activeId ? agentProgress : null}
          subQuestionProgress={requestConversationId === activeId ? subQuestionProgress : null}
          isDeepResearch={requestConversationId === activeId && isDeepResearch}
          onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          onRetry={handleRetry}
          onSuggestionSelect={handleSuggestionSelect}
        />

        {storageNotice && (
          <div className="storage-notice" role="status">
            <span>{storageNotice}</span>
            <button type="button" onClick={() => setStorageNotice(null)} aria-label="Dismiss storage notice">
              Dismiss
            </button>
          </div>
        )}

        <div className="composer-wrapper">
          <Composer value={composerText} loading={loading} onChange={setComposerText} onSend={handleSend} />
          <p className="composer-footer">
            Research summaries can be incomplete. Review source papers before making important decisions.
          </p>
        </div>
      </div>

      {sidebarOpen && <div className="overlay" onClick={() => setSidebarOpen(false)} />}
    </div>
  );
}
