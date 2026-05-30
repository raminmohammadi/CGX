import { create } from "zustand";
import { api, type StatusResponse } from "../lib/api";

// Shared connection state: polls /api/status, exposes the last good payload
// plus an offline flag the Header and a top-level banner subscribe to. A
// single global poller is started on first subscribe and torn down when the
// last subscriber unmounts to keep dev-mode StrictMode double-mount safe.

export interface ConnectionState {
  status: StatusResponse | null;
  offline: boolean;
  error: string | null;
  lastCheck: number | null;
  consecutiveFailures: number;
  isChecking: boolean;
  // Trigger an out-of-band poll (e.g. the "Retry" button).
  refresh: () => Promise<void>;
}

const POLL_INTERVAL_MS = 15000;
const FAST_RETRY_MS = 3000;

export const useConnection = create<ConnectionState>((set, get) => ({
  status: null,
  offline: false,
  error: null,
  lastCheck: null,
  consecutiveFailures: 0,
  isChecking: false,
  refresh: async () => {
    if (get().isChecking) return;
    set({ isChecking: true });
    try {
      const s = await api.status();
      set({
        status: s,
        offline: false,
        error: null,
        lastCheck: Date.now(),
        consecutiveFailures: 0,
        isChecking: false,
      });
    } catch (e: any) {
      const failures = get().consecutiveFailures + 1;
      set({
        // One transient failure shouldn't show the banner — wait for two.
        offline: failures >= 2,
        error: String(e?.message || e),
        lastCheck: Date.now(),
        consecutiveFailures: failures,
        isChecking: false,
      });
    }
  },
}));

// Module-level poller — owned by whichever component calls startConnectionPoller().
// We deliberately keep this outside React so StrictMode re-mounts don't double-poll.
let pollerHandle: ReturnType<typeof setTimeout> | null = null;
let pollerActive = false;

function scheduleNext() {
  if (!pollerActive) return;
  const { offline } = useConnection.getState();
  const delay = offline ? FAST_RETRY_MS : POLL_INTERVAL_MS;
  pollerHandle = setTimeout(async () => {
    await useConnection.getState().refresh();
    scheduleNext();
  }, delay);
}

export function startConnectionPoller(): () => void {
  if (pollerActive) {
    // Already polling — just return a no-op disposer for the new caller.
    return () => {};
  }
  pollerActive = true;
  // Kick off an immediate check, then schedule the loop based on outcome.
  void useConnection.getState().refresh().then(scheduleNext);
  return () => {
    pollerActive = false;
    if (pollerHandle) {
      clearTimeout(pollerHandle);
      pollerHandle = null;
    }
  };
}
