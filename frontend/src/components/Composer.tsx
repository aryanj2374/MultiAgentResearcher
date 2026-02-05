import { useEffect, useRef } from "react";

type ComposerProps = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
};

export default function Composer({ value, loading, onChange, onSend }: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "auto";
    const height = Math.min(textareaRef.current.scrollHeight, 160);
    textareaRef.current.style.height = `${height}px`;
  }, [value]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!loading && value.trim()) {
        onSend();
      }
    }
  };

  return (
    <div className="composer">
      <textarea
        ref={textareaRef}
        placeholder="Ask a research question"
        rows={1}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
      />
      <button
        className="send-btn"
        onClick={onSend}
        type="button"
        disabled={loading || !value.trim()}
        aria-label="Send message"
      >
        ➤
      </button>
    </div>
  );
}
