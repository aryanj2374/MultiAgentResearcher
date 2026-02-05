import { useCallback, useRef, useEffect } from "react";

type ComposerProps = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
};

export default function Composer({ value, loading, onChange, onSend }: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        onSend();
      }
    },
    [onSend]
  );

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [value]);

  return (
    <div className="composer">
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a research question..."
        disabled={loading}
      />
      <button
        type="button"
        className="send-btn"
        onClick={onSend}
        disabled={loading || !value.trim()}
        aria-label="Send message"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
        </svg>
      </button>
    </div>
  );
}
