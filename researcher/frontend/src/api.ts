import type { AnySSEEvent } from "./types";

const BASE = "/api";

export async function startQuery(
  query: string,
  mode: "normal" | "research",
  depth: number,
  accuracy: number
): Promise<string> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode, depth, accuracy }),
  });
  if (!res.ok) throw new Error(`Failed to start query: ${res.statusText}`);
  const data = await res.json();
  return data.query_id as string;
}

export function streamQuery(
  queryId: string,
  onEvent: (event: AnySSEEvent) => void,
  onDone: () => void,
  onError: (err: string) => void
): () => void {
  const es = new EventSource(`${BASE}/stream/${queryId}`);

  es.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data) as AnySSEEvent;
      if (event.type === "keepalive") return;
      onEvent(event);
      if (event.type === "result" || event.type === "error") {
        es.close();
        onDone();
      }
    } catch {
    }
  };

  es.onerror = () => {
    es.close();
    onError("Connection lost");
  };

  return () => es.close();
}

export async function fetchCapabilities(): Promise<unknown> {
  const res = await fetch(`${BASE}/capabilities`);
  if (!res.ok) return null;
  return res.json();
}
