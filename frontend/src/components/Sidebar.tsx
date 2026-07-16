import type { Conversation } from "../types";

type SidebarProps = {
  conversations: Conversation[];
  activeId: string | null;
  collapsed: boolean;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onClose: () => void;
};

export default function Sidebar({
  conversations,
  activeId,
  collapsed,
  onSelect,
  onNewChat,
  onClose,
}: SidebarProps) {
  return (
    <aside
      className={`sidebar ${collapsed ? "collapsed" : ""}`}
      aria-label="Research conversations"
      aria-hidden={collapsed}
      {...(collapsed ? ({ inert: "" } as Record<string, string>) : {})}
    >
      <div className="sidebar-header">
        <div className="app-name">
          <span className="logo" aria-hidden="true">
            <span>R</span>
          </span>
          <div>
            <p>Research Atlas</p>
            <span>Evidence, mapped clearly</span>
          </div>
        </div>
        <button className="close-btn" onClick={onClose} type="button" aria-label="Close sidebar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
            <path d="m6 6 12 12M18 6 6 18" />
          </svg>
        </button>
      </div>

      <button className="new-chat" onClick={onNewChat} type="button">
        <span className="new-chat-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 5v14M5 12h14" />
          </svg>
        </span>
        <span>New research</span>
      </button>

      <div className="conversation-section-header">
        <span>Recent</span>
        <span>{conversations.length}</span>
      </div>

      <nav className="conversation-list" aria-label="Recent conversations">
        {conversations.length === 0 && <p className="muted">No conversations yet.</p>}
        {conversations.map((conv) => (
          <button
            key={conv.id}
            className={`conversation-item ${conv.id === activeId ? "active" : ""}`}
            onClick={() => onSelect(conv.id)}
            type="button"
          >
            <span className="conversation-indicator" aria-hidden="true" />
            <span className="conversation-copy">
              <span className="conversation-title">{conv.title}</span>
              <span className="conversation-meta">
                {conv.messages.filter((message) => message.role === "user").length || "No"} question{conv.messages.filter((message) => message.role === "user").length === 1 ? "" : "s"}
              </span>
            </span>
          </button>
        ))}
      </nav>

      <div className="sidebar-note">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
          <path d="m9 12 2 2 4-4" />
        </svg>
        <div>
          <strong>Saved in this browser</strong>
          <span>Your conversation history stays local.</span>
        </div>
      </div>
    </aside>
  );
}
