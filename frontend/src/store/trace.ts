import { create } from "zustand";
import { api, type TraceSettings } from "../lib/api";

// Shared trace-toggle state. The Settings page owns the interactive control;
// the global Header subscribes so the "TRACE" pill is visible everywhere
// whenever the flag is on. Keeping it in a store (rather than polling the
// status endpoint) makes the UI reflect a toggle within the same tick the
// POST resolves.

interface TraceState {
  settings: TraceSettings | null;
  loaded: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  set: (next: TraceSettings) => void;
}

export const useTrace = create<TraceState>((set) => ({
  settings: null,
  loaded: false,
  error: null,
  refresh: async () => {
    try {
      const s = await api.getTraceSettings();
      set({ settings: s, loaded: true, error: null });
    } catch (e: any) {
      set({ loaded: true, error: String(e?.message || e) });
    }
  },
  set: (next) => set({ settings: next, loaded: true, error: null }),
}));
