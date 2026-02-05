import type { BackendResponse } from "../types";

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
