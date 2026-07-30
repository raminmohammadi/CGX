import { create } from "zustand";
import { persist } from "zustand/middleware";

// Selection + layout state for the session-shaped Agent page. Persisted
// so a tab switch / reload comes back to the same active session and
// the same panel sizing; the actual session snapshot is reloaded from
// ``/api/agent-session/{sid}`` on mount instead of being cached
// client-side.

// Panel-width bounds. Resize handles clamp to these so the layout never
// collapses below something usable; collapse buttons replace the column
// with a slim rail instead.
export const SESSION_BAR_MIN = 160;
export const SESSION_BAR_MAX = 360;
export const SESSION_BAR_DEFAULT = 224;
export const TASK_TREE_MIN = 180;
export const TASK_TREE_MAX = 420;
export const TASK_TREE_DEFAULT = 256;
export const SIDE_PANEL_MIN = 220;
export const SIDE_PANEL_MAX = 480;
export const SIDE_PANEL_DEFAULT = 288;

export interface AgentSessionUIState {
  activeId: string | null;
  selectedTaskId: string | null;
  // Model each run was dispatched with, keyed by session id. The provider is
  // frozen into the backend drain at dispatch time, so a mid-run profile
  // switch can't retro-apply; this lets the UI show which model the in-flight
  // run is actually using versus the currently-selected profile.
  runModels: Record<string, string>;
  sessionBarWidth: number;
  sessionBarCollapsed: boolean;
  taskTreeWidth: number;
  sidePanelWidth: number;
  sidePanelCollapsed: boolean;
  setActiveId: (id: string | null) => void;
  setSelectedTaskId: (id: string | null) => void;
  setRunModel: (sid: string, model: string) => void;
  setSessionBarWidth: (w: number) => void;
  setSessionBarCollapsed: (v: boolean) => void;
  setTaskTreeWidth: (w: number) => void;
  setSidePanelWidth: (w: number) => void;
  setSidePanelCollapsed: (v: boolean) => void;
}

const clamp = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, Math.round(v)));

export const useAgentSession = create<AgentSessionUIState>()(
  persist(
    (set) => ({
      activeId: null,
      selectedTaskId: null,
      runModels: {},
      sessionBarWidth: SESSION_BAR_DEFAULT,
      sessionBarCollapsed: false,
      taskTreeWidth: TASK_TREE_DEFAULT,
      sidePanelWidth: SIDE_PANEL_DEFAULT,
      sidePanelCollapsed: false,
      setActiveId: (id) => set({ activeId: id }),
      setSelectedTaskId: (id) => set({ selectedTaskId: id }),
      setRunModel: (sid, model) =>
        set((s) => ({ runModels: { ...s.runModels, [sid]: model } })),
      setSessionBarWidth: (w) =>
        set({ sessionBarWidth: clamp(w, SESSION_BAR_MIN, SESSION_BAR_MAX) }),
      setSessionBarCollapsed: (v) => set({ sessionBarCollapsed: v }),
      setTaskTreeWidth: (w) =>
        set({ taskTreeWidth: clamp(w, TASK_TREE_MIN, TASK_TREE_MAX) }),
      setSidePanelWidth: (w) =>
        set({ sidePanelWidth: clamp(w, SIDE_PANEL_MIN, SIDE_PANEL_MAX) }),
      setSidePanelCollapsed: (v) => set({ sidePanelCollapsed: v }),
    }),
    { name: "cgx-agent-session" },
  ),
);
