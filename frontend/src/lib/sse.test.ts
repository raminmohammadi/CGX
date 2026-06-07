import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { streamSSE } from "./sse";

// Build a mock Response whose body streams `chunks` as Uint8Array frames.
function mockResponseStream(chunks: string[], ok = true, status = 200): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
  return new Response(stream, { status, statusText: ok ? "OK" : "ERR" });
}

describe("streamSSE", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("parses named events with JSON payloads", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponseStream([
        "event: token\n",
        'data: {"text":"hello"}\n\n',
        "event: token\n",
        'data: {"text":" world"}\n\n',
      ]),
    ) as any;

    const events: Array<[string, any]> = [];
    const conn = streamSSE("/api/ask/stream", { q: "x" }, (e, d) =>
      events.push([e, d]),
    );
    await conn.done;

    expect(events).toEqual([
      ["token", { text: "hello" }],
      ["token", { text: " world" }],
    ]);
  });

  it("handles frames split across multiple network chunks", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponseStream([
        "event: tok",
        "en\ndata: {\"text\":\"ab",
        "c\"}\n\nevent: done\ndata: {}\n\n",
      ]),
    ) as any;

    const events: Array<[string, any]> = [];
    await streamSSE("/api/x", {}, (e, d) => events.push([e, d])).done;

    expect(events).toEqual([
      ["token", { text: "abc" }],
      ["done", {}],
    ]);
  });

  it("defaults event name to 'message' when omitted", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponseStream(['data: {"x":1}\n\n']),
    ) as any;

    const events: Array<[string, any]> = [];
    await streamSSE("/api/x", {}, (e, d) => events.push([e, d])).done;

    expect(events).toEqual([["message", { x: 1 }]]);
  });

  it("falls back to raw string when payload is not JSON", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponseStream(["event: log\ndata: just-text\n\n"]),
    ) as any;

    const events: Array<[string, any]> = [];
    await streamSSE("/api/x", {}, (e, d) => events.push([e, d])).done;

    expect(events).toEqual([["log", "just-text"]]);
  });

  it("ignores comment lines and trims data prefix", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      mockResponseStream([": keepalive\nevent: ping\ndata: {\"ok\":true}\n\n"]),
    ) as any;

    const events: Array<[string, any]> = [];
    await streamSSE("/api/x", {}, (e, d) => events.push([e, d])).done;

    expect(events).toEqual([["ping", { ok: true }]]);
  });

  it("invokes onError for non-2xx responses", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response("nope", { status: 500 }),
    ) as any;

    const onError = vi.fn();
    await streamSSE("/api/x", {}, () => {}, onError).done;

    expect(onError).toHaveBeenCalledTimes(1);
    const err = onError.mock.calls[0][0] as Error;
    expect(err.message).toContain("500");
  });

  it("does not invoke onError when aborted", async () => {
    // Hanging stream so abort() actually has something to cancel.
    const stream = new ReadableStream<Uint8Array>({ start() {} });
    globalThis.fetch = vi.fn().mockImplementation((_url, init) => {
      return new Promise((resolve, reject) => {
        const signal = (init as RequestInit | undefined)?.signal;
        signal?.addEventListener("abort", () =>
          reject(Object.assign(new Error("aborted"), { name: "AbortError" })),
        );
        // Never resolve unless aborted.
        void stream;
        return resolve;
      });
    }) as any;

    const onError = vi.fn();
    const conn = streamSSE("/api/x", {}, () => {}, onError);
    conn.abort();
    await conn.done;
    expect(onError).not.toHaveBeenCalled();
  });
});
