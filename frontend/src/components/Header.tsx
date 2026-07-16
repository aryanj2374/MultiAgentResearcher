type HeaderProps = {
  title: string;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onToggleSidebar: () => void;
};

export default function Header({
  title,
  theme,
  onToggleTheme,
  onToggleSidebar,
}: HeaderProps) {
  return (
    <header className="chat-header">
      <div className="header-left">
        <button className="hamburger" onClick={onToggleSidebar} type="button" aria-label="Toggle sidebar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
            <path d="M4 7h16M4 12h16M4 17h16" />
          </svg>
        </button>
        <div className="header-title">
          <h2>{title.length > 40 ? title.slice(0, 40) + "..." : title}</h2>
          <span className="header-status">
            <i aria-hidden="true" /> Evidence workspace
          </span>
        </div>
      </div>
      <div className="header-actions">
        <span className="agent-count" aria-label="Six specialized research agents">
          <span className="agent-stack" aria-hidden="true">
            <i className="planner" />
            <i className="retriever" />
            <i className="extractor" />
            <i className="critic" />
            <i className="synthesizer" />
            <i className="referee" />
          </span>
          6 agents
        </span>
        <button
          className="theme-toggle"
          onClick={onToggleTheme}
          type="button"
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 8.5 8.5 0 1 0 20.5 14.2Z" />
            </svg>
          )}
        </button>
      </div>
    </header>
  );
}
