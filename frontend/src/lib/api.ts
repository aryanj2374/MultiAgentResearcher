import type { BackendResponse, ProgressEvent } from "../types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

function parseProgressEvent(data: string): ProgressEvent {
  try {
    const event = JSON.parse(data) as ProgressEvent;
    if (!event || typeof event !== "object" || typeof event.type !== "string") {
      throw new Error("Invalid event shape");
    }
    return event;
  } catch {
    throw new Error("The server returned a malformed progress event.");
  }
}

function responseError(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object" || !("detail" in payload)) return fallback;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item
          ? String((item as { msg: unknown }).msg)
          : ""
      )
      .filter(Boolean);
    if (messages.length > 0) return messages.join(" ");
  }
  return fallback;
}

export async function askQuestion(question: string): Promise<BackendResponse> {
  const res = await fetch(apiUrl("/api/ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    throw new Error(responseError(payload, "Request failed"));
  }

  return (await res.json()) as BackendResponse;
}

export async function* askQuestionStream(question: string): AsyncGenerator<ProgressEvent> {
  const res = await fetch(apiUrl("/api/ask/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    throw new Error(responseError(payload, "Request failed"));
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let dataLines: string[] = [];
  let streamEnded = false;
  let receivedTerminalEvent = false;

  const consumeLine = (rawLine: string): ProgressEvent | null => {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "") {
      if (dataLines.length === 0) return null;
      const event = parseProgressEvent(dataLines.join("\n"));
      dataLines = [];
      return event;
    }
    if (line.startsWith(":")) return null;
    if (line === "data" || line.startsWith("data:")) {
      const value = line === "data" ? "" : line.slice(5).replace(/^ /, "");
      dataLines.push(value);
    }
    return null;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        streamEnded = true;
        buffer += decoder.decode();
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const event = consumeLine(line);
        if (event) {
          if (event.type === "result" || event.type === "error") receivedTerminalEvent = true;
          yield event;
        }
      }
    }
  } finally {
    if (!streamEnded) await reader.cancel().catch(() => undefined);
  }

  if (buffer) {
    const event = consumeLine(buffer);
    if (event) {
      if (event.type === "result" || event.type === "error") receivedTerminalEvent = true;
      yield event;
    }
  }
  const finalEvent = consumeLine("");
  if (finalEvent) {
    if (finalEvent.type === "result" || finalEvent.type === "error") receivedTerminalEvent = true;
    yield finalEvent;
  }

  if (!receivedTerminalEvent) {
    throw new Error("The research stream ended before a result was returned.");
  }
}
