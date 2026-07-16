import { useState } from "react";

type CollapsibleProps = {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
};

export default function Collapsible({ title, children, defaultOpen = false }: CollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="collapsible">
      <button
        className="collapsible-toggle"
        onClick={() => setOpen((prev) => !prev)}
        type="button"
        aria-expanded={open}
      >
        <span>{title}</span>
        <span className={`chevron ${open ? "open" : ""}`} aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="m8 10 4 4 4-4" />
          </svg>
        </span>
      </button>
      <div className={`collapsible-panel ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="collapsible-content">{children}</div>
      </div>
    </div>
  );
}
