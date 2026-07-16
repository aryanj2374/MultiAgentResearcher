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
    <div className={`composer ${loading ? "is-loading" : ""}`}>
      <div className="composer-input-row">
        <span className="composer-spark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="m12 3 1.4 4.1L17.5 8.5l-4.1 1.4L12 14l-1.4-4.1-4.1-1.4 4.1-1.4L12 3Z" />
            <path d="m18.5 14 .8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2Z" />
          </svg>
        </span>
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a focused research question..."
          disabled={loading}
          maxLength={500}
          autoFocus
          aria-label="Research question"
          aria-describedby="composer-help"
        />
      </div>
      <div className="composer-toolbar">
        <span className="composer-help" id="composer-help">
          {loading ? (
            <><i className="loading-pulse" aria-hidden="true" /> Building your evidence brief</>
          ) : (
            <><kbd>Enter</kbd> research <span aria-hidden="true">·</span> <kbd>Shift</kbd> + <kbd>Enter</kbd> new line</>
          )}
        </span>
        <div className="composer-actions">
          {value.length > 420 && <span className="character-count">{value.length}/500</span>}
          <button
            type="button"
            className="send-btn"
            onClick={onSend}
            disabled={loading || !value.trim()}
            aria-label={loading ? "Research in progress" : "Start research"}
          >
            {loading ? (
              <span className="button-spinner" aria-hidden="true" />
            ) : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="M5 12h14M13 6l6 6-6 6" />
              </svg>
            )}
            <span>{loading ? "Researching" : "Research"}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
