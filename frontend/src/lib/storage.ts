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

export function saveConversations(conversations: Conversation[]): void {
  try {
    localStorage.setItem(CONVO_KEY, JSON.stringify(conversations));
  } catch {
    // Storage can be unavailable or full; the in-memory conversation remains usable.
  }
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
