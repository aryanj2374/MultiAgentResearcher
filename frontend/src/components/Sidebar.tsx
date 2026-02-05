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
    <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
      <div className="sidebar-header">
        <div className="app-name">
          <span className="logo">MA</span>
          <div>
            <p>Multi-Agent Research</p>
            <span>Scientific Assistant</span>
          </div>
        </div>
        <button className="close-btn" onClick={onClose} type="button" aria-label="Close sidebar">
          ✕
        </button>
      </div>

      <button className="new-chat" onClick={onNewChat} type="button">
        + New chat
      </button>

      <div className="conversation-list">
        {conversations.length === 0 && <p className="muted">No conversations yet.</p>}
        {conversations.map((conv) => (
          <button
            key={conv.id}
            className={`conversation-item ${conv.id === activeId ? "active" : ""}`}
            onClick={() => onSelect(conv.id)}
            type="button"
          >
            {conv.title}
          </button>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="footer-item">Settings</div>
        <div className="footer-item">Profile</div>
        <div className="footer-item">Help</div>
      </div>
    </aside>
  );
}
