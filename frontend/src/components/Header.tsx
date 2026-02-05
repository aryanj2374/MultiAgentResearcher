type HeaderProps = {
  title: string;
  subtitle: string;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onToggleSidebar: () => void;
};

export default function Header({ title, subtitle, theme, onToggleTheme, onToggleSidebar }: HeaderProps) {
  return (
    <header className="chat-header">
      <button className="hamburger" onClick={onToggleSidebar} type="button" aria-label="Toggle sidebar">
        ☰
      </button>
      <div className="header-title">
        <h2>{title}</h2>
        <span className="muted">{subtitle}</span>
      </div>
      <button className="theme-toggle" onClick={onToggleTheme} type="button">
        {theme === "dark" ? "Light" : "Dark"}
      </button>
    </header>
  );
}
