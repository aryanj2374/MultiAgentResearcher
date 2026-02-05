import type { Conversation, Theme } from "../types";

const CONVO_KEY = "ma_conversations_v1";
const THEME_KEY = "ma_theme_v1";

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(CONVO_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as Conversation[];
  } catch {
    return [];
  }
}

export function saveConversations(conversations: Conversation[]): void {
  localStorage.setItem(CONVO_KEY, JSON.stringify(conversations));
}

export function loadTheme(): Theme | null {
  const raw = localStorage.getItem(THEME_KEY);
  if (raw === "light" || raw === "dark") return raw;
  return null;
}

export function saveTheme(theme: Theme): void {
  localStorage.setItem(THEME_KEY, theme);
}
