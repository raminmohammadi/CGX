// Module-level singleton for tracking an active Hugging Face embedding pull.
// Mirrors pullManager.ts so an in-flight download survives tab navigation.

import { useEffect, useState } from "react";
import { api } from "./api";
import type { SseConnection } from "./sse";

export interface EmbedPullState {
  model: string;
  status: string;
  total: number;
  completed: number;
  done: boolean;
  error: string | null;
}

type Listener = (state: EmbedPullState | null) => void;

let _state: EmbedPullState | null = null;
let _conn: SseConnection | null = null;
const _listeners = new Set<Listener>();

function _notify() {
  _listeners.forEach((fn) => fn(_state));
}

export function getActiveEmbedPull(): EmbedPullState | null {
  return _state;
}

export function subscribeEmbedPull(fn: Listener): () => void {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

export function startEmbedPull(
  model: string,
  onRefresh?: () => void,
) {
  _conn?.abort();
  _state = {
    model,
    status: "Connecting…",
    total: 0,
    completed: 0,
    done: false,
    error: null,
  };
  _notify();

  _conn = api.embedPull(
    model,
    (data) => {
      if (!_state || _state.model !== model) return;
      _state = {
        ..._state,
        status: data.status || _state.status,
        total: data.total ?? _state.total,
        completed: data.completed ?? _state.completed,
        done: data.status === "success",
        error: data.error ? data.error : _state.error,
      };
      _notify();
    },
    () => {
      if (_state) {
        _state = { ..._state, done: true, status: "Download complete" };
        _notify();
      }
      onRefresh?.();
    },
    (err) => {
      if (_state) {
        _state = { ..._state, error: String(err), done: true };
        _notify();
      }
    },
  );
}

export function cancelEmbedPull() {
  _conn?.abort();
  _conn = null;
  _state = null;
  _notify();
}

export function useEmbedPullState(): EmbedPullState | null {
  const [state, setState] = useState<EmbedPullState | null>(() => _state);
  useEffect(() => subscribeEmbedPull(setState), []);
  return state;
}
