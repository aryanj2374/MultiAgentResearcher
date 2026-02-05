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
      <button className="collapsible-toggle" onClick={() => setOpen((prev) => !prev)} type="button">
        <span>{title}</span>
        <span className={`chevron ${open ? "open" : ""}`}>▾</span>
      </button>
      {open && <div className="collapsible-content">{children}</div>}
    </div>
  );
}
