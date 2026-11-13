export type Role = "user" | "assistant";

export interface Source {
  title: string;
  url: string;
  type: string;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  sources?: Source[];
  error?: string;
}

export interface Info {
  updated_at: string | null;
  chunks: number;
  documents: number;
  courses: number;
  categories: Record<string, number>;
  llm: string;
  embeddings: string;
  ready: boolean;
}

export type StreamEvent =
  | { type: "token"; text: string }
  | { type: "status"; text: string }
  | { type: "sources"; items: Source[] }
  | { type: "error"; message: string }
  | { type: "done" };

// Empty in development (Vite proxies /api) and when the backend serves the UI itself.
// For GitHub Pages it is the Render URL, set at build time via VITE_API_URL.
export const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "");

export async function fetchInfo(): Promise<Info> {
  const res = await fetch(`${API_BASE}/api/info`);
  if (!res.ok) throw new Error(`info ${res.status}`);
  return res.json();
}

export async function streamChat(
  history: { role: Role; content: string }[],
  onEvent: (e: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: history }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(
      res.status === 0 || res.status >= 500
        ? "Can't reach the Sunbeam server. Check that the backend is running."
        : `The server answered with an error (${res.status}).`,
    );
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = raw.split("\n").find((l) => l.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)) as StreamEvent);
    }
  }
}
