// Module-level SSE connection registry.
//
// Holding connections here (outside any React component) means they survive
// tab switches: when React Router unmounts a page component the connection
// keeps streaming and updates the Zustand tasks store, so the UI is up-to-date
// when the user navigates back to that tab.

import type { SseConnection } from "./sse";

const _connections = new Map<string, SseConnection>();

export function getConnection(key: string): SseConnection | null {
  return _connections.get(key) ?? null;
}

export function setConnection(key: string, conn: SseConnection): void {
  _connections.set(key, conn);
}

export function abortConnection(key: string): void {
  _connections.get(key)?.abort();
  _connections.delete(key);
}

export function hasConnection(key: string): boolean {
  return _connections.has(key);
}
