import type { BackendResponse, ProgressEvent } from "../types";

export async function askQuestion(question: string): Promise<BackendResponse> {
  const res = await fetch("http://localhost:8000/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    const detail = payload?.detail ?? "Request failed";
    throw new Error(detail);
  }

  return (await res.json()) as BackendResponse;
}

export async function* askQuestionStream(question: string): AsyncGenerator<ProgressEvent> {
  const res = await fetch("http://localhost:8000/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    const detail = payload?.detail ?? "Request failed";
    throw new Error(detail);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6)) as ProgressEvent;
          yield event;
        } catch {
          // Ignore malformed JSON
        }
      }
    }
  }

  // Process any remaining data in buffer
  if (buffer.startsWith("data: ")) {
    try {
      const event = JSON.parse(buffer.slice(6)) as ProgressEvent;
      yield event;
    } catch {
      // Ignore malformed JSON
    }
  }
}
