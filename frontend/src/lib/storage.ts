import type { Conversation, Theme } from "../types";

const CONVO_KEY = "ma_conversations_v1";
const THEME_KEY = "ma_theme_v1";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isConversation(value: unknown): value is Conversation {
  if (!isRecord(value) || !Array.isArray(value.messages)) return false;
  if (
    typeof value.id !== "string" ||
    typeof value.title !== "string" ||
    typeof value.createdAt !== "string"
  ) {
    return false;
  }
  return value.messages.every(
    (message) =>
      isRecord(message) &&
      typeof message.id === "string" &&
      (message.role === "user" || message.role === "assistant") &&
      typeof message.content === "string" &&
      typeof message.createdAt === "string"
  );
}

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(CONVO_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isConversation);
  } catch {
    return [];
  }
}

/**
 * Strip the bulky fields a stored conversation does not need.
 *
 * `logs` repeats every paper once per sub-question and `sub_results` repeats
 * the papers, extractions and critiques already held at the top level, so a
 * single run serialises to roughly 230KB. A handful of those exceeds the ~5MB
 * localStorage quota, and the write then throws and loses the whole history.
 * The in-memory objects keep every field; only the persisted copy is trimmed.
 */
function trimForStorage(conversations: Conversation[]): Conversation[] {
  return conversations.map((conversation) => ({
    ...conversation,
    messages: conversation.messages.map((message) => {
      const response = message.meta?.response;
      if (!response) return message;
      const { logs: _logs, sub_results: _subResults, ...rest } = response;
      return {
        ...message,
        meta: {
          ...message.meta,
          response: { ...rest, logs_trimmed: true } as typeof response,
        },
      };
    }),
  }));
}

export type SaveResult = { ok: true } | { ok: false; reason: "quota" | "unavailable"; dropped: number };

/**
 * Persist conversations, evicting the oldest until the payload fits.
 *
 * Returns what happened so the caller can tell the user, rather than losing
 * history silently.
 */
export function saveConversations(conversations: Conversation[]): SaveResult {
  const trimmed = trimForStorage(conversations);

  for (let dropped = 0; dropped < trimmed.length; dropped += 1) {
    // Conversations are newest-first, so evict from the tail.
    const candidate = dropped === 0 ? trimmed : trimmed.slice(0, trimmed.length - dropped);
    try {
      localStorage.setItem(CONVO_KEY, JSON.stringify(candidate));
      return dropped === 0 ? { ok: true } : { ok: false, reason: "quota", dropped };
    } catch (error) {
      if (!isQuotaError(error)) {
        return { ok: false, reason: "unavailable", dropped: 0 };
      }
    }
  }

  // Even one conversation will not fit. Clear the key so a stale, larger
  // history does not keep occupying the quota.
  try {
    localStorage.removeItem(CONVO_KEY);
  } catch {
    // Nothing further we can do.
  }
  return { ok: false, reason: "quota", dropped: conversations.length };
}

function isQuotaError(error: unknown): boolean {
  if (!(error instanceof DOMException)) return false;
  return (
    error.name === "QuotaExceededError" ||
    error.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
    error.code === 22
  );
}

export function loadTheme(): Theme | null {
  try {
    const raw = localStorage.getItem(THEME_KEY);
    if (raw === "light" || raw === "dark") return raw;
    return null;
  } catch {
    return null;
  }
}

export function saveTheme(theme: Theme): void {
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Theme persistence is optional.
  }
}
