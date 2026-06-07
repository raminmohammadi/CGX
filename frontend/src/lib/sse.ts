// Server-Sent Events client that uses fetch + ReadableStream so we can
// POST a JSON body (EventSource only supports GET). The FastAPI backend
// emits `event: <name>` and `data: <json>` lines per SSE spec.

export type SseHandler = (event: string, data: any) => void;

export interface SseConnection {
  abort: () => void;
  done: Promise<void>;
}

export function streamSSE(
  url: string,
  body: unknown,
  onEvent: SseHandler,
  onError?: (err: unknown) => void,
): SseConnection {
  const ctl = new AbortController();
  const done = (async () => {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "text/event-stream",
        },
        body: JSON.stringify(body),
        signal: ctl.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`SSE ${url} → ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done: rdDone } = await reader.read();
        if (rdDone) break;
        // Normalise \r\n → \n so frame splitting works regardless of whether
        // the server (e.g. sse-starlette 3.x) uses \r\n or \n line endings.
        buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        // SSE frames are separated by a blank line (\n\n).
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          handleFrame(frame, onEvent);
        }
      }
      if (buf.trim()) handleFrame(buf, onEvent);
    } catch (err) {
      if ((err as any)?.name === "AbortError") return;
      onError?.(err);
    }
  })();
  return { abort: () => ctl.abort(), done };
}

function handleFrame(frame: string, onEvent: SseHandler) {
  let event = "message";
  const dataLines: string[] = [];
  for (const raw of frame.split("\n")) {
    const line = raw.replace(/\r$/, "");
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return;
  const payload = dataLines.join("\n");
  let parsed: any = payload;
  try {
    parsed = JSON.parse(payload);
  } catch {
    /* keep as raw text */
  }
  onEvent(event, parsed);
}
