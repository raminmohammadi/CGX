// Per-page task state that persists across React Router tab switches.
//
// Each page stores its streaming state here (in Zustand) instead of in
// component-local useState, so:
//   - Switching tabs unmounts the component but the SSE connection keeps
//     running and keeps writing to this store.
//   - Remounting the component reads the accumulated state immediately.
//
// We use sessionStorage (not localStorage) so state resets on a full
// page reload while persisting across tab navigation within one session.

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

// ─── shared types ───────────────────────────────────────────────────────────

export interface RawEvent {
  type: string;
  payload: any;
  at: number;
}

// ─── agent page ─────────────────────────────────────────────────────────────

export interface CodegenReportSummary {
  overall_ok: boolean;
  attempts: number;
  error?: string;
  counts?: {
    n_targets: number;
    n_patches_ok: number;
    n_patches_failed: number;
    n_syntax_ok: number;
    n_syntax_failed: number;
  };
  tests_ran?: boolean;
  tests_passed?: boolean;
  patch_failures?: { file: string; error: string; rejected_preview: string }[];
  syntax_errors?: { file: string; language: string; error: string; line?: number | null }[];
  tests?: {
    ran: boolean;
    returncode?: number | null;
    tests_selected?: string[];
    skipped_reason?: string | null;
    stdout_tail?: string;
    stderr_tail?: string;
  };
}

export interface TaskOutput {
  plan_md?: string;
  diffs?: { file: string; patch: string }[];
  codegen_report?: CodegenReportSummary;
  answer_md?: string;
  top_files?: { file: string; score: number }[];
  // ── apply task ──
  applied_files?: string[];
  failed_files?: { file: string; error: string }[];
  backup_dir?: string;
  project_tree?: string;
  // ── verify task ──
  ran?: boolean;
  tests_passed?: boolean;
  returncode?: number;
  tests_selected?: string[];
  skipped_reason?: string;
  stdout_tail?: string;
  stderr_tail?: string;
}

export interface TaskRow {
  id: string;
  name: string;
  description: string;
  kind: string;
  status: "pending" | "running" | "done" | "failed" | "skipped" | string;
  judge?: any;
  error?: string;
  elapsed?: number;
  summary?: string;
  output?: TaskOutput;
  retryBoundary?: boolean;
  dependencies?: string[];
  criteria?: string[];
}

export type ExecutionMode = "auto" | "review" | "plan-only";

export interface AgentState {
  busy: boolean;
  phase: "idle" | "planning" | "executing" | "done";
  goal: string;
  stopOnFail: boolean;
  executionMode: ExecutionMode;
  awaitingApproval: boolean;
  tasks: TaskRow[];
  planTitle: string | null;
  /** Planner-supplied explanation of why the plan is structured this way. */
  rationale: string;
  events: RawEvent[];
  summary: string | null;
  error: string | null;
}

// ─── ask page ────────────────────────────────────────────────────────────────

export interface ChatMsg {
  role: "user" | "assistant";
  content: string;
  sources?: any[];
  intent?: { mode?: string };
  streaming?: boolean;
  thought?: string;
  warning?: string;
}

export interface AskState {
  busy: boolean;
  messages: ChatMsg[];
  error: string | null;
}

// ─── plan page ───────────────────────────────────────────────────────────────

export interface PlanState {
  busy: boolean;
  thought: string;
  warning: string | null;
  planMd: string | null;
  diff: string | null;
  report: any | null;
  error: string | null;
}

// ─── index page ──────────────────────────────────────────────────────────────

export interface IndexProgressItem {
  stage?: string;
  message?: string;
}

export interface IndexState {
  busy: boolean;
  progress: IndexProgressItem[];
  result: any | null;
  error: string | null;
}

// ─── store ───────────────────────────────────────────────────────────────────

interface TasksStore {
  agent: AgentState;
  ask: AskState;
  plan: PlanState;
  index: IndexState;

  // Agent
  setAgent: (patch: Partial<AgentState>) => void;
  upsertAgentTask: (id: string, patch: Partial<TaskRow>) => void;
  appendAgentEvent: (ev: RawEvent) => void;
  resetAgent: () => void;

  // Ask
  setAsk: (patch: Partial<AskState>) => void;
  appendAskMessage: (msg: ChatMsg) => void;
  patchLastAskMessage: (patch: Partial<ChatMsg>) => void;
  resetAsk: () => void;

  // Plan
  setPlan: (patch: Partial<PlanState>) => void;
  resetPlan: () => void;

  // Index
  setIndex: (patch: Partial<IndexState>) => void;
  appendIndexProgress: (item: IndexProgressItem) => void;
  resetIndex: () => void;
}

const defaultAgent: AgentState = {
  busy: false, phase: "idle", goal: "", stopOnFail: true,
  executionMode: "auto", awaitingApproval: false,
  tasks: [], planTitle: null, rationale: "",
  events: [], summary: null, error: null,
};
const defaultAsk: AskState = { busy: false, messages: [], error: null };
const defaultPlan: PlanState = {
  busy: false, thought: "", warning: null,
  planMd: null, diff: null, report: null, error: null,
};
const defaultIndex: IndexState = { busy: false, progress: [], result: null, error: null };

export const useTasks = create<TasksStore>()(
  persist(
    (set) => ({
      agent: defaultAgent,
      ask: defaultAsk,
      plan: defaultPlan,
      index: defaultIndex,

      // ── Agent ──────────────────────────────────────────────────────────────
      setAgent: (patch) =>
        set((s) => ({ agent: { ...s.agent, ...patch } })),

      upsertAgentTask: (id, patch) =>
        set((s) => {
          const prev = s.agent.tasks;
          const idx = prev.findIndex((t) => t.id === id);
          if (idx >= 0) {
            const next = [...prev];
            next[idx] = { ...next[idx], ...patch };
            return { agent: { ...s.agent, tasks: next } };
          }
          return {
            agent: {
              ...s.agent,
              tasks: [
                ...prev,
                { id, name: "", description: "", kind: "ask", status: "pending", ...patch },
              ],
            },
          };
        }),

      appendAgentEvent: (ev) =>
        set((s) => {
          if (ev.type === "task_progress") {
            const tid = String(ev.payload?.task_id || ev.payload?.id || "");
            if (tid) {
              const events = s.agent.events;
              for (let i = events.length - 1; i >= 0; i--) {
                if (
                  events[i].type === "task_progress" &&
                  String(events[i].payload?.task_id || events[i].payload?.id || "") === tid
                ) {
                  const next = [...events];
                  next[i] = ev;
                  return { agent: { ...s.agent, events: next } };
                }
              }
            }
          }
          return { agent: { ...s.agent, events: [...s.agent.events, ev] } };
        }),

      resetAgent: () => set({ agent: defaultAgent }),

      // ── Ask ────────────────────────────────────────────────────────────────
      setAsk: (patch) =>
        set((s) => ({ ask: { ...s.ask, ...patch } })),

      appendAskMessage: (msg) =>
        set((s) => ({ ask: { ...s.ask, messages: [...s.ask.messages, msg] } })),

      patchLastAskMessage: (patch) =>
        set((s) => {
          const msgs = [...s.ask.messages];
          if (!msgs.length) return s;
          msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], ...patch };
          return { ask: { ...s.ask, messages: msgs } };
        }),

      resetAsk: () => set({ ask: defaultAsk }),

      // ── Plan ───────────────────────────────────────────────────────────────
      setPlan: (patch) =>
        set((s) => ({ plan: { ...s.plan, ...patch } })),

      resetPlan: () => set({ plan: defaultPlan }),

      // ── Index ──────────────────────────────────────────────────────────────
      setIndex: (patch) =>
        set((s) => ({ index: { ...s.index, ...patch } })),

      appendIndexProgress: (item) =>
        set((s) => ({ index: { ...s.index, progress: [...s.index.progress, item] } })),

      resetIndex: () => set({ index: defaultIndex }),
    }),
    {
      name: "averix-tasks",
      storage: createJSONStorage(() => sessionStorage),
      // Only persist non-streaming state to keep sessionStorage lean.
      // `busy` persists so the UI can show "Running" on remount.
      partialize: (s) => ({
        agent: {
          busy: s.agent.busy,
          phase: s.agent.phase,
          goal: s.agent.goal,
          stopOnFail: s.agent.stopOnFail,
          executionMode: s.agent.executionMode,
          awaitingApproval: s.agent.awaitingApproval,
          tasks: s.agent.tasks,
          planTitle: s.agent.planTitle,
          rationale: s.agent.rationale,
          events: s.agent.events.slice(-200), // cap stored events
          summary: s.agent.summary,
          error: s.agent.error,
        },
        ask: {
          busy: s.ask.busy,
          messages: s.ask.messages,
          error: s.ask.error,
        },
        plan: {
          busy: s.plan.busy,
          thought: s.plan.thought,
          warning: s.plan.warning,
          planMd: s.plan.planMd,
          diff: s.plan.diff,
          report: s.plan.report,
          error: s.plan.error,
        },
        index: {
          busy: s.index.busy,
          progress: s.index.progress,
          result: s.index.result,
          error: s.index.error,
        },
      }),
    },
  ),
);
