// Module-level singleton for tracking an active `ollama pull` operation.
// Lives outside React so navigating away from SettingsPage doesn't lose state.
// Components subscribe via usePullState() and get live updates on re-mount.

import { useEffect, useState } from "react";
import { api } from "./api";
import type { SseConnection } from "./sse";

export interface PullState {
  model: string;
  base_url: string;
  status: string;
  total: number;
  completed: number;
  done: boolean;
  error: string | null;
}

type Listener = (state: PullState | null) => void;

let _state: PullState | null = null;
let _conn: SseConnection | null = null;
const _listeners = new Set<Listener>();

function _notify() {
  _listeners.forEach((fn) => fn(_state));
}

/** Read-once snapshot — use usePullState() for reactive updates. */
export function getActivePull(): PullState | null {
  return _state;
}

/** Subscribe to pull-state changes. Returns an unsubscribe function. */
export function subscribePull(fn: Listener): () => void {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

/** Start a pull. Cancels any in-flight pull first. */
export function startPull(
  model: string,
  base_url: string,
  onRefreshInstalled?: () => void,
) {
  _conn?.abort();
  _state = {
    model,
    base_url,
    status: "Connecting…",
    total: 0,
    completed: 0,
    done: false,
    error: null,
  };
  _notify();

  _conn = api.ollamaPull(
    model,
    base_url,
    (data) => {
      if (!_state || _state.model !== model) return;
      // Ollama signals a failed pull in two NDJSON shapes:
      //   {"status":"error","error":"..."}  (normalised by our backend)
      //   {"error":"..."}                    (bare; e.g. manifest 404/412)
      // The backend's `_gen` rewrites the second into the first, but we
      // accept both here so any future drift / direct backend changes
      // still surface as an error rather than a silent close.
      const errMsg =
        data.status === "error" || data.error
          ? String(data.error || data.status || "pull failed")
          : null;
      _state = {
        ..._state,
        status: data.status || _state.status,
        total: data.total ?? _state.total,
        completed: data.completed ?? _state.completed,
        done: data.status === "success" || errMsg != null,
        error: errMsg ?? _state.error,
      };
      _notify();
    },
    () => {
      if (_state) {
        // Only declare success on close if nothing reported an error and the
        // model was actually marked done by a status="success" event.
        if (_state.error) {
          _state = { ..._state, done: true };
        } else if (_state.done) {
          _state = { ..._state, status: "Download complete" };
        } else {
          _state = {
            ..._state,
            done: true,
            error: "Pull ended without success; see Ollama logs.",
          };
        }
        _notify();
      }
      onRefreshInstalled?.();
    },
    (err) => {
      if (_state) {
        _state = { ..._state, error: String(err), done: true };
        _notify();
      }
    },
  );
}

/** Abort the active pull and clear state. */
export function cancelPull() {
  _conn?.abort();
  _conn = null;
  _state = null;
  _notify();
}

/** React hook — returns live pull state and re-renders on every update. */
export function usePullState(): PullState | null {
  const [state, setState] = useState<PullState | null>(() => _state);
  useEffect(() => subscribePull(setState), []);
  return state;
}
