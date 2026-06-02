import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import {
  Circle,
  CircleCheck,
  CircleDashed,
  CircleX,
  ClipboardList,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Eye,
  Gavel,
  GitBranch,
  Lightbulb,
  Loader2,
  Network,
  Play,
  Undo2,
  X,
  Zap,
} from "lucide-react";
import { streamSSE } from "../lib/sse";
import { abortConnection, getConnection, setConnection } from "../lib/connections";
import {
  useTasks,
  type ExecutionMode,
  type TaskRow,
  type TaskOutput,
  type RawEvent,
  type CodegenReportSummary,
} from "../store/tasks";
import { useWorkspace } from "../store/workspace";
import { Card, CardHeader } from "../components/Card";
import { Field, TextArea, TextInput, Toggle } from "../components/Input";
import { Pill } from "../components/Pill";
import { Markdown } from "../components/Markdown";
import { DiffView } from "../components/DiffView";
import { api } from "../lib/api";

const PAGE_KEY = "agent";

// ─── kind colour palette ──────────────────────────────────────────────────────

const KIND_META: Record<string, {
  border: string; bg: string; text: string; badge: string;
}> = {
  search:            { border: "border-sky-500/50",     bg: "bg-sky-950/40",    text: "text-sky-300",    badge: "bg-sky-900/80 text-sky-300" },
  ask:               { border: "border-purple-500/50",  bg: "bg-purple-950/40", text: "text-purple-300", badge: "bg-purple-900/80 text-purple-300" },
  plan:              { border: "border-orange-500/50",  bg: "bg-orange-950/40", text: "text-orange-300", badge: "bg-orange-900/80 text-orange-300" },
  apply:             { border: "border-emerald-500/50", bg: "bg-emerald-950/40",text: "text-emerald-300",badge: "bg-emerald-900/80 text-emerald-300" },
  verify:            { border: "border-cyan-500/50",    bg: "bg-cyan-950/40",   text: "text-cyan-300",   badge: "bg-cyan-900/80 text-cyan-300" },
  scaffold:          { border: "border-amber-500/50",   bg: "bg-amber-950/40",  text: "text-amber-300",  badge: "bg-amber-900/80 text-amber-300" },
  scaffold_manifest: { border: "border-yellow-500/50",  bg: "bg-yellow-950/40", text: "text-yellow-300", badge: "bg-yellow-900/80 text-yellow-300" },
  scaffold_file:     { border: "border-amber-400/40",   bg: "bg-amber-900/30",  text: "text-amber-200",  badge: "bg-amber-800/70 text-amber-200" },
  fill_logic:        { border: "border-rose-500/50",    bg: "bg-rose-950/40",   text: "text-rose-300",   badge: "bg-rose-900/80 text-rose-300" },
  summarize:         { border: "border-slate-500/40",   bg: "bg-slate-800/40",  text: "text-slate-300",  badge: "bg-slate-700/80 text-slate-300" },
};
const defaultKindMeta = KIND_META.ask;

// ─── execution mode definitions ───────────────────────────────────────────────

interface ModeOption {
  id: ExecutionMode;
  icon: React.ReactNode;
  title: string;
  description: string;
}

const MODES: ModeOption[] = [
  {
    id: "auto",
    icon: <Zap className="h-4 w-4" />,
    title: "Auto mode",
    description: "Plan and execute the full pipeline automatically.",
  },
  {
    id: "review",
    icon: <Eye className="h-4 w-4" />,
    title: "Review plan",
    description: "Inspect the execution DAG before committing to run.",
  },
  {
    id: "plan-only",
    icon: <ClipboardList className="h-4 w-4" />,
    title: "Plan only",
    description: "Generate and visualise the plan — no code is changed.",
  },
];

// ─── main page ────────────────────────────────────────────────────────────────

export default function AgentPage() {
  const { provider, index, projectRoot, setProjectRoot } = useWorkspace();
  const {
    agent,
    setAgent,
    upsertAgentTask,
    appendAgentEvent,
    resetAgent,
  } = useTasks();

  const {
    busy, phase, goal, stopOnFail, executionMode, awaitingApproval,
    tasks, planTitle, rationale, events, summary, error,
  } = agent;

  const [liveProgress, setLiveProgress] = useState<
    Record<string, { name: string; kind: string; elapsed: number; at: number }>
  >({});

  // Re-hydrate after tab switch while stream was running.
  useEffect(() => {
    if (busy && !getConnection(PAGE_KEY)) {
      setAgent({ busy: false, phase: phase === "executing" ? "done" : phase });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── execution helpers ──────────────────────────────────────────────────────

  /** Wire up the SSE stream and update state incrementally. */
  const startExecution = () => {
    setAgent({
      busy: true,
      error: null,
      tasks: [],
      events: [],
      summary: null,
      planTitle: null,
      rationale: "",
      phase: "planning",
      awaitingApproval: false,
    });
    setLiveProgress({});
    abortConnection(PAGE_KEY);

    const conn = streamSSE(
      "/api/agent",
      { goal, project_root: projectRoot || null, stop_on_fail: stopOnFail, index, provider },
      (ev, data) => {
        if (ev === "done") return;

        if (ev === "task_progress") {
          const tid = String(data?.task_id || data?.id || "");
          const elapsed = Number(data?.elapsed);
          if (tid) {
            setLiveProgress((prev) => ({
              ...prev,
              [tid]: {
                name: String(data?.name || data?.description || tid),
                kind: String(data?.kind || ""),
                elapsed,
                at: Date.now() / 1000,
              },
            }));
            if (Number.isFinite(elapsed))
              upsertAgentTask(tid, { status: "running", elapsed });
          }
          return;
        }

        appendAgentEvent({ type: ev, payload: data, at: Date.now() / 1000 });

        if (ev === "status") {
          const ph = String(data?.phase || "");
          if (ph === "planning" || ph === "executing" || ph === "done")
            setAgent({ phase: ph as typeof phase });
        } else if (ev === "plan") {
          setAgent({ phase: "executing" });
          const planObj = data?.plan || data || {};
          const planTasks: any[] = planObj?.tasks || data?.tasks || [];
          const goalText = planObj?.goal || data?.goal || goal;
          setAgent({
            planTitle: String(goalText),
            rationale: String(planObj?.rationale || data?.rationale || ""),
            tasks: planTasks.map((t: any) => ({
              id: String(t.id || t.description || Math.random()),
              name: String(t.name || t.title || ""),
              description: String(t.description || ""),
              kind: String(t.kind || "ask"),
              status: String(t.status || "pending") as TaskRow["status"],
              dependencies: Array.isArray(t.dependencies) ? t.dependencies : [],
            })),
          });
        } else if (ev === "task_start") {
          const tid = String(data?.task_id || data?.task?.id || data?.id || "");
          if (tid)
            upsertAgentTask(tid, {
              name: data?.name || data?.task?.name || "",
              description: data?.description || data?.task?.description || "",
              kind: data?.kind || data?.task?.kind || "ask",
              status: "running",
            });
        } else if (ev === "task_done") {
          const tid = String(data?.task_id || data?.task?.id || data?.id || "");
          if (tid) {
            const elapsed = Number(data?.elapsed);
            upsertAgentTask(tid, {
              status: "done",
              elapsed: Number.isFinite(elapsed) ? elapsed : undefined,
              summary: data?.summary || undefined,
              output: data?.output || undefined,
            });
            setLiveProgress((prev) => { const n = { ...prev }; delete n[tid]; return n; });
          }
        } else if (ev === "task_failed") {
          const tid = String(data?.task_id || data?.task?.id || data?.id || "");
          if (tid) {
            upsertAgentTask(tid, {
              status: "failed",
              error: data?.error || data?.task?.error,
              summary: data?.summary || undefined,
              output: data?.output || undefined,
            });
            setLiveProgress((prev) => { const n = { ...prev }; delete n[tid]; return n; });
          }
        } else if (ev === "task_skipped") {
          const tid = String(data?.task_id || data?.task?.id || data?.id || "");
          if (tid) {
            upsertAgentTask(tid, { status: "skipped" });
            setLiveProgress((prev) => { const n = { ...prev }; delete n[tid]; return n; });
          }
        } else if (ev === "judge") {
          const tid = String(data?.task_id || data?.id || "");
          if (tid) upsertAgentTask(tid, { judge: data?.verdict || data });
        } else if (ev === "summary") {
          const finalTasks: any[] = data?.plan?.tasks || [];
          for (const t of finalTasks) {
            const tid = String(t.id || "");
            if (tid) {
              upsertAgentTask(tid, {
                status: String(t.status || "pending") as TaskRow["status"],
                error: t.error || undefined,
                judge: t.judge || undefined,
              });
            }
          }
          const c = Number(data?.completed ?? 0);
          const f = Number(data?.failed ?? 0);
          const s = Number(data?.skipped ?? 0);
          setAgent({
            summary:
              typeof data === "string"
                ? data
                : `${c} completed · ${f} failed · ${s} skipped`,
            phase: "done",
          });
        } else if (ev === "retry_start") {
          setAgent({ summary: null, phase: "executing",
                     error: `Attempt ${data?.attempt ?? 2}: ${data?.reason ?? "re-planning…"}` });
        } else if (ev === "retry_plan") {
          setAgent({ phase: "executing", error: null });
          const planObj = data?.plan || data || {};
          const planTasks: any[] = planObj?.tasks || data?.tasks || [];
          const newTasks: TaskRow[] = planTasks.map((t: any, i: number) => ({
            id: String(t.id || t.description || Math.random()),
            name: String(t.name || t.title || ""),
            description: String(t.description || ""),
            kind: String(t.kind || "ask"),
            status: String(t.status || "pending") as TaskRow["status"],
            retryBoundary: i === 0,
            dependencies: Array.isArray(t.dependencies) ? t.dependencies : [],
          }));
          useTasks.setState((s) => ({
            agent: { ...s.agent, tasks: [...s.agent.tasks, ...newTasks] },
          }));
        } else if (ev === "cancelled") {
          setAgent({ error: "Cancelled by user.", phase: "done", busy: false });
        } else if (ev === "error") {
          setAgent({ error: String(data?.message || data?.error || "agent error") });
        }
      },
      (err) => {
        setAgent({ error: String((err as any)?.message || err), busy: false });
      },
    );

    setConnection(PAGE_KEY, conn);

    conn.done.finally(() => {
      useTasks.setState((s) => ({
        agent: {
          ...s.agent,
          busy: false,
          phase: (s.agent.phase === "executing" || s.agent.phase === "planning")
            ? "done"
            : s.agent.phase,
        },
      }));
      abortConnection(PAGE_KEY);
    });
  };

  /** Main start handler — branches based on the selected execution mode. */
  const start = async () => {
    if (!goal.trim() || busy) return;

    if (executionMode === "auto") {
      startExecution();
      return;
    }

    // review or plan-only: call the plan-only endpoint first
    setAgent({
      busy: true,
      error: null,
      tasks: [],
      events: [],
      summary: null,
      planTitle: null,
      rationale: "",
      phase: "planning",
      awaitingApproval: false,
    });
    setLiveProgress({});

    try {
      const result = await api.agentPlan({
        goal, project_root: projectRoot || null, stop_on_fail: stopOnFail, index, provider,
      });

      if (result.error || !result.plan) {
        setAgent({ busy: false, error: result.error || "Planner returned no plan.", phase: "idle" });
        return;
      }

      const planTasks: TaskRow[] = (result.plan.tasks || []).map((t: any) => ({
        id: String(t.id || Math.random()),
        name: String(t.name || ""),
        description: String(t.description || ""),
        kind: String(t.kind || "ask"),
        status: "pending" as const,
        dependencies: Array.isArray(t.dependencies) ? t.dependencies : [],
        criteria: Array.isArray(t.criteria) ? t.criteria : [],
      }));

      setAgent({
        busy: false,
        phase: "idle",
        tasks: planTasks,
        planTitle: result.plan.goal || goal,
        rationale: String(result.plan.rationale || ""),
        awaitingApproval: executionMode === "review",
      });
    } catch (e: any) {
      setAgent({ busy: false, error: String(e?.message || e), phase: "idle" });
    }
  };

  /** Approve the previewed plan and begin execution. */
  const approve = () => {
    if (!awaitingApproval) return;
    startExecution();
  };

  const stop = async () => {
    abortConnection(PAGE_KEY);
    setAgent({ busy: false });
  };

  const stages = useMemo(() => computeStages(tasks, phase), [tasks, phase]);

  const showDag = tasks.length > 0;
  // Show the detailed timeline only during / after execution (not while just previewing plan).
  const showTimeline = planTitle !== null && !awaitingApproval;

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      <CardHeader
        title="Multi-Agent Orchestration Tracker"
        description="Live execution pipeline traversing Planner generation, Tracker task dispatches, and structural Judge diagnostics."
        right={
          busy ? (
            <button onClick={stop} className="av-btn-ghost">
              <X className="h-3 w-3" /> Cancel
            </button>
          ) : null
        }
      />

      <Card padded>
        <div className="grid grid-cols-3 gap-3">
          <Field className="col-span-3" label="Goal">
            <TextArea
              rows={3}
              value={goal}
              onChange={(e) => setAgent({ goal: e.target.value })}
              placeholder='e.g. "Add docstrings to every public function in cgx.parser"'
            />
            <p className="mt-1.5 text-[10px] text-slate-500 leading-relaxed">
              To generate a <span className="text-slate-300">brand-new project</span> from
              scratch, start your goal with{" "}
              <code className="text-slate-400 bg-white/5 px-1 rounded">"create a new …"</code>,{" "}
              <code className="text-slate-400 bg-white/5 px-1 rounded">"build a … from scratch"</code>,
              or similar — no existing index needed.
            </p>
          </Field>
          <Field label="Project root" className="col-span-2">
            <TextInput
              value={projectRoot}
              onChange={(e) => setProjectRoot(e.target.value)}
              placeholder="/abs/path/to/repo (optional)"
            />
          </Field>
          <div className="flex flex-col gap-3 justify-end">
            <Toggle
              checked={stopOnFail}
              onChange={(v) => setAgent({ stopOnFail: v })}
              label="Stop on fail"
            />
          </div>
        </div>
        <div className="flex justify-between items-center mt-4">
          <button onClick={resetAgent} disabled={busy} className="av-btn-ghost text-[10px]">
            Clear
          </button>
          <div className="flex items-center gap-2">
            <ExecutionModePicker
              mode={executionMode}
              disabled={busy}
              onChange={(m) => setAgent({ executionMode: m })}
            />
            <button
              onClick={start}
              disabled={busy || !goal.trim() || awaitingApproval}
              className="av-btn-primary"
            >
              <Play className="h-3 w-3" />
              {busy ? (phase === "planning" ? "Planning…" : "Running…") : "Launch Agent"}
            </button>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-3 gap-4">
        <StageCard index={1} title="Planner Layer" state={stages.planner}
          description="Decomposes the user goal into ordered, kind-typed tasks." />
        <StageCard index={2} title="Tracker Engine" state={stages.tracker}
          description="Dispatches each task and streams progress + outputs." />
        <StageCard index={3} title="Judge Verdict" state={stages.judge}
          description="Structural verification of task outputs against criteria." />
      </div>

      {/* ── Planner rationale — shown above the DAG whenever the planner returned one ── */}
      {rationale && <PlanRationaleCard rationale={rationale} />}

      {/* ── DAG panel — shown as soon as we have a plan (any mode) ── */}
      {showDag && (
        <PlanDAGPanel
          tasks={tasks}
          phase={phase}
          awaitingApproval={awaitingApproval}
          executionMode={executionMode}
          onApprove={approve}
        />
      )}

      {/* ── Detailed timeline — only visible once execution is underway ── */}
      {showTimeline && (
        <Card padded>
          <div className="flex items-start gap-3 mb-4">
            <GitBranch className="h-4 w-4 text-emerald-400 mt-0.5" />
            <div>
              <p className="text-[10px] uppercase tracking-wider font-mono text-slate-500">Goal</p>
              <p className="text-sm text-slate-200 font-medium">{planTitle}</p>
            </div>
          </div>
          <PlanTasksDropdown tasks={tasks} phase={phase} />
        </Card>
      )}

      {summary && (
        <Card padded>
          <CardHeader
            title="Summary"
            eyebrow="Tracker output"
            right={(() => {
              // Derive the badge from the effective (retry-collapsed) task
              // list so a recovered SCAFFOLD_FILE failure doesn't show red.
              const stage = computeStages(tasks, phase);
              const anySkipped = tasks.some((t) => t.status === "skipped");
              if (stage.tracker === "failed") {
                return <Pill tone="red">Failed</Pill>;
              }
              if (anySkipped) {
                return <Pill tone="amber">Complete with warnings</Pill>;
              }
              return <Pill tone="neon">Complete</Pill>;
            })()}
          />
          <pre className="text-[11px] text-slate-300 font-mono whitespace-pre-wrap leading-relaxed">
            {summary}
          </pre>
        </Card>
      )}

      <div>
        <p className="av-section-eyebrow mb-2">Live Event Stream</p>
        <div className="bg-slate-950 border border-white/5 rounded-xl p-4 font-mono text-xs text-slate-400 h-56 overflow-y-auto space-y-1">
          {events.length === 0 && Object.keys(liveProgress).length === 0 && (
            <p className={`italic ${busy ? "text-emerald-600/60" : "text-slate-600"}`}>
              {busy ? "Connecting to agent…" : "Awaiting agent events…"}
            </p>
          )}
          {events.map((e, i) => (
            <EventLine key={i} ev={e} />
          ))}
          {Object.entries(liveProgress).map(([tid, p]) => (
            <LiveProgressLine key={tid} taskId={tid} progress={p} />
          ))}
        </div>
      </div>

      {error && (
        <Card padded className="border-red-500/40">
          <p className="text-xs text-red-300 font-mono">{error}</p>
        </Card>
      )}
    </div>
  );
}

// ─── planner rationale card ───────────────────────────────────────────────────

function PlanRationaleCard({ rationale }: { rationale: string }) {
  const [open, setOpen] = useState(true);
  return (
    <Card padded>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 text-left"
      >
        <Lightbulb className="h-4 w-4 text-amber-300 shrink-0" />
        <span className="text-sm font-semibold text-slate-200">Plan Rationale</span>
        <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500">
          Planner thoughts
        </span>
        <span className="ml-auto text-slate-500">
          {open ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
        </span>
      </button>
      {open && (
        <div className="mt-3 text-[12px] text-slate-300 leading-relaxed">
          <Markdown text={rationale} />
        </div>
      )}
    </Card>
  );
}

// ─── execution mode picker ────────────────────────────────────────────────────

function ExecutionModePicker({
  mode,
  disabled,
  onChange,
}: {
  mode: ExecutionMode;
  disabled: boolean;
  onChange: (m: ExecutionMode) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const current = MODES.find((m2) => m2.id === mode) ?? MODES[0];

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-[11px] font-mono text-slate-300 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {current.icon}
        <span>{current.title}</span>
        <ChevronDown className="h-3 w-3 text-slate-500" />
      </button>

      {open && (
        <div className="absolute bottom-full right-0 mb-2 w-72 rounded-xl border border-white/10 bg-slate-900 shadow-2xl z-50 overflow-hidden">
          <div className="px-3 py-2 border-b border-white/5 flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-widest font-mono text-slate-400">
              Execution Mode
            </span>
            <span className="text-[10px] font-mono text-slate-600">⇧ + tab to switch</span>
          </div>
          {MODES.map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => { onChange(opt.id); setOpen(false); }}
              className="w-full flex items-start gap-3 px-3 py-2.5 hover:bg-white/5 text-left transition-colors"
            >
              <span className="mt-0.5 text-slate-400 shrink-0">{opt.icon}</span>
              <div className="flex-1 min-w-0">
                <p className="text-[12px] font-medium text-slate-200">{opt.title}</p>
                <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">{opt.description}</p>
              </div>
              {mode === opt.id && (
                <CircleCheck className="h-4 w-4 text-emerald-400 mt-0.5 shrink-0" />
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── DAG panel ────────────────────────────────────────────────────────────────

function PlanDAGPanel({
  tasks,
  phase,
  awaitingApproval,
  executionMode,
  onApprove,
}: {
  tasks: TaskRow[];
  phase: string;
  awaitingApproval: boolean;
  executionMode: ExecutionMode;
  onApprove?: () => void;
}) {
  const [detailsOpen, setDetailsOpen] = useState(awaitingApproval);

  if (tasks.length === 0) return null;

  const completed = tasks.filter((t) => t.status === "done" || t.status === "skipped").length;
  const failed = tasks.filter((t) => t.status === "failed").length;
  const running = tasks.filter((t) => t.status === "running").length;

  const statusLabel = awaitingApproval
    ? "awaiting approval"
    : running > 0
      ? `${running} running`
      : failed > 0
        ? `${failed} failed`
        : completed === tasks.length
          ? "complete"
          : phase === "planning"
            ? "planning…"
            : "pending";

  return (
    <Card padded>
      {/* Panel header */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <Network className="h-4 w-4 text-emerald-400" />
          <span className="text-sm font-semibold text-slate-200">Execution DAG</span>
          <span className="text-[10px] font-mono text-slate-500 bg-white/5 border border-white/8 px-2 py-0.5 rounded">
            {tasks.length} task{tasks.length === 1 ? "" : "s"} · sequential
          </span>
          <span className="text-[10px] font-mono text-slate-600">{statusLabel}</span>
        </div>

        {awaitingApproval && executionMode === "review" && onApprove && (
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-amber-400 animate-pulse">
              Plan ready — review before executing
            </span>
            <button onClick={onApprove} className="av-btn-primary text-[11px]">
              <Play className="h-3 w-3" /> Approve &amp; Execute
            </button>
          </div>
        )}

        {awaitingApproval && executionMode === "plan-only" && (
          <span className="text-[11px] font-mono text-slate-500">
            Dry run — no changes will be made
          </span>
        )}
      </div>

      {/* Horizontal scrollable graph */}
      <div className="overflow-x-auto pb-2">
        <div className="flex items-center gap-0 min-w-max">
          {tasks.map((task, i) => {
            const isRetryStart = task.retryBoundary === true;
            return (
              <Fragment key={task.id}>
                {isRetryStart && (
                  <div className="flex flex-col items-center justify-center px-2 h-[88px]">
                    <div className="h-full flex flex-col items-center justify-center gap-1">
                      <div className="h-3 w-px bg-amber-500/30" />
                      <span className="text-[9px] font-mono text-amber-500 uppercase tracking-widest whitespace-nowrap rotate-90">
                        retry
                      </span>
                      <div className="h-3 w-px bg-amber-500/30" />
                    </div>
                  </div>
                )}
                {!isRetryStart && i > 0 && <DAGArrow active={tasks[i - 1].status === "done"} />}
                {isRetryStart && <DAGArrow active={false} retry />}
                <DAGNode task={task} />
              </Fragment>
            );
          })}
        </div>
      </div>

      {/* Kind legend + details toggle */}
      <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-white/5 items-center">
        {Array.from(new Set(tasks.map((t) => t.kind))).map((kind) => {
          const meta = KIND_META[kind] ?? defaultKindMeta;
          return (
            <span key={kind} className={`text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded ${meta.badge}`}>
              {kind}
            </span>
          );
        })}
        <span className="text-[10px] font-mono text-slate-600">
          {completed}/{tasks.length} complete
        </span>
        <button
          type="button"
          onClick={() => setDetailsOpen((v) => !v)}
          className="ml-auto inline-flex items-center gap-1 text-[10px] font-mono text-slate-400 hover:text-slate-200 transition-colors"
        >
          {detailsOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          {detailsOpen ? "hide details" : "show details"}
        </button>
      </div>

      {/* Expandable plan details */}
      {detailsOpen && (
        <div className="mt-4 pt-4 border-t border-white/5 space-y-3">
          <p className="text-[10px] uppercase tracking-widest font-mono text-slate-500">Plan details</p>
          {tasks.map((task, i) => {
            const meta = KIND_META[task.kind] ?? defaultKindMeta;
            return (
              <div key={task.id} className="flex gap-3">
                <span className="text-[10px] font-mono text-slate-600 mt-0.5 shrink-0 w-4 text-right">{i + 1}.</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded shrink-0 ${meta.badge}`}>
                      {task.kind}
                    </span>
                    <span className="text-[12px] font-medium text-slate-200">
                      {task.name || task.description}
                    </span>
                  </div>
                  {task.description && task.description !== task.name && (
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      {task.description}
                    </p>
                  )}
                  {task.criteria && task.criteria.length > 0 && (
                    <ul className="mt-1 space-y-0.5">
                      {task.criteria.map((c, ci) => (
                        <li key={ci} className="text-[10px] text-slate-500 flex gap-1.5">
                          <span className="text-slate-600 shrink-0">·</span>
                          <span>{c}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}

function DAGArrow({ active, retry }: { active: boolean; retry?: boolean }) {
  return (
    <div className="flex items-center justify-center w-8 h-[88px] shrink-0">
      <svg width="28" height="14" viewBox="0 0 28 14" fill="none" className="shrink-0">
        <line
          x1="0" y1="7" x2="20" y2="7"
          stroke={retry ? "#f59e0b" : active ? "#34d399" : "#334155"}
          strokeWidth="1.5"
          strokeDasharray={active ? undefined : "3 2"}
        />
        <polygon
          points="20,3 28,7 20,11"
          fill={retry ? "#f59e0b" : active ? "#34d399" : "#334155"}
        />
      </svg>
    </div>
  );
}

function DAGNode({ task }: { task: TaskRow }) {
  const meta = KIND_META[task.kind] ?? defaultKindMeta;

  const statusBorderOverride =
    task.status === "running"
      ? "border-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.25)]"
      : task.status === "done"
        ? "border-emerald-500/60"
        : task.status === "failed"
          ? "border-red-500/60"
          : null;

  const borderClass = statusBorderOverride ?? meta.border;

  return (
    <div
      className={`
        w-[152px] rounded-xl border p-3 flex flex-col gap-1.5 shrink-0
        transition-all duration-300
        ${meta.bg} ${borderClass}
        ${task.status === "pending" ? "opacity-60" : "opacity-100"}
      `}
    >
      {/* Kind badge + status icon */}
      <div className="flex items-center justify-between">
        <span className={`text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded ${meta.badge}`}>
          {task.kind}
        </span>
        <DAGStatusIcon status={task.status} />
      </div>

      {/* Task name */}
      <p className={`text-[11px] font-medium leading-tight line-clamp-2 ${meta.text}`}>
        {task.name || task.description || "(task)"}
      </p>

      {/* Task description (shown when name and description differ) */}
      {task.description && task.description !== task.name && (
        <p className="text-[10px] text-slate-400 leading-tight line-clamp-2">
          {task.description}
        </p>
      )}

      {/* Elapsed time */}
      {typeof task.elapsed === "number" && task.status !== "pending" && (
        <span className="text-[10px] font-mono text-slate-500 mt-auto">
          {formatElapsed(task.elapsed)}
        </span>
      )}
    </div>
  );
}

function DAGStatusIcon({ status }: { status: string }) {
  const base = "h-3.5 w-3.5 shrink-0";
  if (status === "done")    return <CircleCheck className={`${base} text-emerald-400`} />;
  if (status === "failed")  return <CircleX className={`${base} text-red-400`} />;
  if (status === "running") return <Loader2 className={`${base} text-amber-400 animate-spin`} />;
  if (status === "skipped") return <CircleDashed className={`${base} text-slate-500`} />;
  return <Circle className={`${base} text-slate-600`} />;
}

// ─── stage cards ──────────────────────────────────────────────────────────────

function StageCard({
  index,
  title,
  description,
  state,
}: {
  index: number;
  title: string;
  description: string;
  state: "pending" | "running" | "done" | "failed";
}) {
  const tone =
    state === "done"
      ? "border-emerald-500/30"
      : state === "running"
        ? "border-emerald-500 shadow-neon-md"
        : state === "failed"
          ? "border-red-500/40"
          : "border-white/5 opacity-40";
  const status =
    state === "done" ? (
      <span className="text-emerald-400 text-[10px] flex items-center gap-1">
        <CircleCheck className="h-3 w-3" /> Pass
      </span>
    ) : state === "running" ? (
      <span className="text-emerald-400 text-[10px] flex items-center gap-1">
        <Loader2 className="h-3 w-3 animate-spin" /> Processing
      </span>
    ) : state === "failed" ? (
      <span className="text-red-400 text-[10px] flex items-center gap-1">
        <CircleX className="h-3 w-3" /> Failed
      </span>
    ) : (
      <span className="text-slate-500 text-[10px]">Pending</span>
    );
  return (
    <div className={`bg-surface p-4 rounded-xl border space-y-1.5 ${tone}`}>
      <div className="flex justify-between items-center text-xs font-bold font-mono">
        <span className={state === "running" ? "text-emerald-400" : "text-white"}>
          {index}. {title.toUpperCase()}
        </span>
        {status}
      </div>
      <p className="text-xs text-slate-400">{description}</p>
    </div>
  );
}

// ─── plan task dropdown (detailed timeline) ───────────────────────────────────

function PlanTasksDropdown({
  tasks,
  phase,
}: {
  tasks: TaskRow[];
  phase: string;
}) {
  const total = tasks.length;
  const running = tasks.some((t) => t.status === "running");
  const allDone = total > 0 && tasks.every((t) => t.status === "done" || t.status === "skipped");
  const anyFailed = tasks.some((t) => t.status === "failed");
  const [open, setOpen] = useState(true);
  const completed = tasks.filter((t) => t.status === "done" || t.status === "skipped").length;
  const dotColor = anyFailed
    ? "bg-red-400"
    : allDone
      ? "bg-emerald-400"
      : running || phase === "executing"
        ? "bg-emerald-400 animate-pulse"
        : "bg-slate-500";
  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between border-b border-white/5 pb-2 text-left"
      >
        <div className="flex items-center gap-2">
          {open ? (
            <ChevronDown className="h-4 w-4 text-slate-400" />
          ) : (
            <ChevronRight className="h-4 w-4 text-slate-400" />
          )}
          <ClipboardList className="h-4 w-4 text-slate-300" />
          <span className="text-sm font-semibold text-slate-100">Scheduled Tasks</span>
          <span className="text-[11px] font-mono text-slate-500">
            {completed}/{total}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-md bg-white/5 border border-white/10 px-2 py-0.5 text-[11px] font-mono text-slate-300">
            <span className="text-emerald-400">+</span>
            {total}
          </span>
          <span className={`h-2 w-2 rounded-full ${dotColor}`} />
        </div>
      </button>
      {open && (
        <div className="mt-3">
          {tasks.map((t, i) => (
            <div key={t.id}>
              {t.retryBoundary && (
                <div className="flex items-center gap-2 my-3 ml-7">
                  <div className="flex-1 h-px bg-amber-500/20" />
                  <span className="text-[10px] font-mono text-amber-400 uppercase tracking-widest">
                    Re-planning
                  </span>
                  <div className="flex-1 h-px bg-amber-500/20" />
                </div>
              )}
              <TaskTimelineRow task={t} isLast={i === tasks.length - 1} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TaskStatusIcon({ status }: { status: TaskRow["status"] }) {
  const base = "h-4 w-4 shrink-0";
  if (status === "done") return <CircleCheck className={`${base} text-emerald-400`} />;
  if (status === "failed") return <CircleX className={`${base} text-red-400`} />;
  if (status === "running") return <Loader2 className={`${base} text-amber-400 animate-spin`} />;
  if (status === "skipped") return <CircleDashed className={`${base} text-slate-500`} />;
  return <Circle className={`${base} text-slate-500`} />;
}

function TaskOutputPanel({ output, kind }: { output: TaskOutput; kind: string }) {
  if (kind === "plan" || kind === "scaffold") {
    const emptyLabel = kind === "scaffold"
      ? "No files were generated."
      : "No plan content produced.";
    return (
      <div className="mt-3 space-y-3">
        {output.codegen_report && <CodegenReportPanel report={output.codegen_report} />}
        {output.plan_md && (
          <div className="rounded-xl border border-white/5 bg-slate-950 p-4">
            <Markdown text={output.plan_md} />
          </div>
        )}
        {output.diffs?.length ? (
          <DiffView diff={output.diffs.map((d) => d.patch).join("\n")} />
        ) : null}
        {!output.plan_md && !output.diffs?.length && !output.codegen_report && (
          <p className="text-[11px] text-slate-500 font-mono italic">{emptyLabel}</p>
        )}
      </div>
    );
  }
  if (kind === "ask") {
    return output.answer_md ? (
      <div className="mt-3 rounded-xl border border-white/5 bg-slate-950 p-4">
        <Markdown text={output.answer_md} />
      </div>
    ) : null;
  }
  if (kind === "search" && output.top_files?.length) {
    return (
      <div className="mt-3 rounded-xl border border-white/5 bg-slate-950 p-3">
        <p className="text-[10px] uppercase tracking-wider font-mono text-slate-500 mb-2">Top files</p>
        <div className="space-y-1">
          {output.top_files.map((f, i) => (
            <div key={i} className="flex items-center justify-between font-mono text-[11px]">
              <span className="text-slate-300 truncate">{f.file}</span>
              {f.score > 0 && (
                <span className="text-slate-500 ml-2 shrink-0">{(f.score * 100).toFixed(0)}%</span>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  }
  if (kind === "apply") {
    return <ApplyPanel output={output} />;
  }
  if (kind === "verify") {
    const ok = !!output.tests_passed;
    const tone = ok ? "border-emerald-500/30" : "border-red-500/30";
    const label = ok
      ? "Tests passed"
      : output.ran === false
        ? `Skipped: ${output.skipped_reason || "no tests"}`
        : `Tests FAILED (rc=${output.returncode ?? "?"})`;
    return (
      <div className={`mt-3 rounded-xl border ${tone} bg-slate-950 p-3 space-y-2`}>
        <p className={`text-[10px] uppercase tracking-wider font-mono ${ok ? "text-emerald-400" : "text-red-400"}`}>
          {label}
        </p>
        {output.tests_selected?.length ? (
          <div className="font-mono text-[11px] text-slate-400 space-y-0.5">
            {output.tests_selected.map((t, i) => (
              <div key={i} className="truncate">{t}</div>
            ))}
          </div>
        ) : null}
        {output.stdout_tail && (
          <pre className="font-mono text-[10px] text-slate-300 whitespace-pre-wrap overflow-x-auto max-h-64">
            {output.stdout_tail}
          </pre>
        )}
        {output.stderr_tail && (
          <pre className="font-mono text-[10px] text-red-300/80 whitespace-pre-wrap overflow-x-auto max-h-32">
            {output.stderr_tail}
          </pre>
        )}
      </div>
    );
  }
  return null;
}

function ApplyPanel({ output }: { output: TaskOutput }) {
  const applied = output.applied_files || [];
  const failed = output.failed_files || [];
  const { projectRoot } = useWorkspace();
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{
    restored: string[];
    deleted: string[];
    failed: { file: string; error: string }[];
    error?: string | null;
  } | null>(null);
  const [errMsg, setErrMsg] = useState<string | null>(null);

  const canRollback =
    !!output.backup_dir && !!projectRoot.trim() && applied.length > 0 && !result;

  const onUndo = async () => {
    if (!output.backup_dir || !projectRoot.trim()) return;
    const ok = window.confirm(
      `Restore ${applied.length} file${applied.length === 1 ? "" : "s"} from\n` +
        `${output.backup_dir}\n\nNew files created by this apply will be deleted. Continue?`,
    );
    if (!ok) return;
    setBusy(true);
    setErrMsg(null);
    try {
      const res = await api.rollback(projectRoot, output.backup_dir);
      setResult({
        restored: res.restored_files,
        deleted: res.deleted_files,
        failed: res.failed_files,
        error: res.error || null,
      });
    } catch (e: any) {
      setErrMsg(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mt-3 space-y-3">
      {applied.length > 0 && (
        <div className="rounded-xl border border-emerald-500/20 bg-slate-950 p-3">
          <p className="text-[10px] uppercase tracking-wider font-mono text-emerald-400 mb-2">
            Wrote {applied.length} file{applied.length === 1 ? "" : "s"} to disk
          </p>
          <div className="space-y-1 font-mono text-[11px]">
            {applied.map((f, i) => (
              <div key={i} className="flex items-center gap-2">
                <CircleCheck className="h-3 w-3 text-emerald-400 shrink-0" />
                <span className="text-slate-300 truncate">{f}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {failed.length > 0 && (
        <div className="rounded-xl border border-red-500/30 bg-slate-950 p-3">
          <p className="text-[10px] uppercase tracking-wider font-mono text-red-400 mb-2">
            Failed ({failed.length})
          </p>
          <div className="space-y-1 font-mono text-[11px]">
            {failed.map((f, i) => (
              <div key={i} className="flex items-start gap-2">
                <CircleX className="h-3 w-3 text-red-400 shrink-0 mt-0.5" />
                <div className="min-w-0">
                  <p className="text-slate-300 truncate">{f.file}</p>
                  <p className="text-red-300/80 break-words">{f.error}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {output.project_tree && (
        <div className="rounded-xl border border-white/5 bg-slate-950 p-3">
          <p className="text-[10px] uppercase tracking-wider font-mono text-slate-500 mb-2">
            Project structure
          </p>
          <pre className="font-mono text-[11px] text-slate-300 whitespace-pre overflow-x-auto">
            {output.project_tree}
          </pre>
        </div>
      )}
      {output.backup_dir && (
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <p className="text-[10px] font-mono text-slate-500 min-w-0 break-all">
            backups → <span className="text-slate-300">{output.backup_dir}</span>
          </p>
          <button
            type="button"
            onClick={onUndo}
            disabled={!canRollback || busy}
            className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] font-mono text-slate-200 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
            title={
              !projectRoot.trim()
                ? "Set the project root in the workspace settings first."
                : applied.length === 0
                  ? "Nothing was applied — nothing to undo."
                  : result
                    ? "Already rolled back."
                    : "Restore originals from the backup directory."
            }
          >
            {busy ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Undo2 className="h-3 w-3" />
            )}
            {busy ? "Rolling back…" : result ? "Rolled back" : "Undo this change"}
          </button>
        </div>
      )}
      {errMsg && (
        <p className="text-[11px] font-mono text-red-400 break-words">{errMsg}</p>
      )}
      {result && (
        <div className="rounded-xl border border-sky-500/20 bg-slate-950 p-3 space-y-1 font-mono text-[11px]">
          <p className="text-[10px] uppercase tracking-wider text-sky-400">
            Rollback complete · restored {result.restored.length} · deleted{" "}
            {result.deleted.length}
            {result.failed.length > 0 ? ` · ${result.failed.length} failed` : ""}
          </p>
          {result.error && (
            <p className="text-red-300/80 break-words">{result.error}</p>
          )}
          {result.restored.map((f, i) => (
            <div key={`r${i}`} className="flex items-center gap-2">
              <Undo2 className="h-3 w-3 text-sky-400 shrink-0" />
              <span className="text-slate-300 truncate">{f}</span>
            </div>
          ))}
          {result.deleted.map((f, i) => (
            <div key={`d${i}`} className="flex items-center gap-2">
              <X className="h-3 w-3 text-slate-400 shrink-0" />
              <span className="text-slate-400 truncate line-through">{f}</span>
            </div>
          ))}
          {result.failed.map((f, i) => (
            <div key={`f${i}`} className="flex items-start gap-2">
              <CircleX className="h-3 w-3 text-red-400 shrink-0 mt-0.5" />
              <div className="min-w-0">
                <p className="text-slate-300 truncate">{f.file}</p>
                <p className="text-red-300/80 break-words">{f.error}</p>
              </div>
            </div>
          ))}
        </div>
      )}
      {output.diffs?.length ? (
        <DiffView diff={output.diffs.map((d) => d.patch).join("\n")} />
      ) : null}
    </div>
  );
}


function CodegenReportPanel({ report }: { report: CodegenReportSummary }) {
  const ok = report.overall_ok;
  const tone = ok ? "border-emerald-500/30" : "border-amber-500/40";
  const heading = ok ? "Self-test passed" : "Self-test failed";
  const counts = report.counts;
  return (
    <div className={`rounded-xl border ${tone} bg-slate-950 p-3 space-y-2`}>
      <div className="flex items-center gap-2 flex-wrap">
        <p
          className={`text-[10px] uppercase tracking-wider font-mono ${
            ok ? "text-emerald-400" : "text-amber-400"
          }`}
        >
          {heading}
        </p>
        <span className="text-[10px] font-mono text-slate-500">
          attempt{report.attempts === 1 ? "" : "s"} {report.attempts}
        </span>
        {counts && (
          <span className="text-[10px] font-mono text-slate-500">
            patches {counts.n_patches_ok}/{counts.n_targets} · syntax{" "}
            {counts.n_syntax_ok}/{counts.n_targets}
          </span>
        )}
        {report.tests_ran && (
          <span
            className={`text-[10px] font-mono ${
              report.tests_passed ? "text-emerald-400" : "text-red-400"
            }`}
          >
            tests {report.tests_passed ? "passed" : "failed"}
          </span>
        )}
      </div>
      {report.error && (
        <p className="text-[11px] text-red-300 font-mono break-words">{report.error}</p>
      )}
      {report.patch_failures?.length ? (
        <div className="space-y-1">
          <p className="text-[10px] uppercase tracking-wider font-mono text-slate-500">
            Patch failures
          </p>
          {report.patch_failures.map((p, i) => (
            <div key={i} className="rounded border border-red-500/20 bg-slate-900/60 p-2">
              <p className="text-[11px] font-mono text-red-300 truncate">{p.file}</p>
              <p className="text-[10px] font-mono text-slate-400 break-words">{p.error}</p>
              {p.rejected_preview && (
                <pre className="mt-1 font-mono text-[10px] text-slate-400 whitespace-pre-wrap overflow-x-auto max-h-32">
                  {p.rejected_preview}
                </pre>
              )}
            </div>
          ))}
        </div>
      ) : null}
      {report.syntax_errors?.length ? (
        <div className="space-y-1">
          <p className="text-[10px] uppercase tracking-wider font-mono text-slate-500">
            Syntax errors
          </p>
          {report.syntax_errors.map((s, i) => (
            <div key={i} className="rounded border border-amber-500/20 bg-slate-900/60 p-2">
              <p className="text-[11px] font-mono text-amber-300 truncate">
                {s.file}
                {s.line != null ? `:${s.line}` : ""}
                {s.language ? ` · ${s.language}` : ""}
              </p>
              <p className="text-[10px] font-mono text-slate-400 break-words">{s.error}</p>
            </div>
          ))}
        </div>
      ) : null}
      {report.tests?.ran && !report.tests_passed && (
        <div className="space-y-1">
          <p className="text-[10px] uppercase tracking-wider font-mono text-slate-500">
            Test output (rc={report.tests?.returncode ?? "?"})
          </p>
          {report.tests.tests_selected?.length ? (
            <div className="font-mono text-[10px] text-slate-400 space-y-0.5">
              {report.tests.tests_selected.map((t, i) => (
                <div key={i} className="truncate">{t}</div>
              ))}
            </div>
          ) : null}
          {report.tests.stdout_tail && (
            <pre className="font-mono text-[10px] text-slate-300 whitespace-pre-wrap overflow-x-auto max-h-48">
              {report.tests.stdout_tail}
            </pre>
          )}
          {report.tests.stderr_tail && (
            <pre className="font-mono text-[10px] text-red-300/80 whitespace-pre-wrap overflow-x-auto max-h-32">
              {report.tests.stderr_tail}
            </pre>
          )}
        </div>
      )}
      {report.tests?.ran === false && report.tests?.skipped_reason && (
        <p className="text-[10px] font-mono text-slate-500 italic">
          Tests skipped: {report.tests.skipped_reason}
        </p>
      )}
    </div>
  );
}

function TaskTimelineRow({ task, isLast }: { task: TaskRow; isLast: boolean }) {
  const [expanded, setExpanded] = useState(
    task.status === "done" || task.status === "failed",
  );

  useEffect(() => {
    if (task.status === "done" || task.status === "failed") {
      setExpanded(true);
    }
  }, [task.status]);
  const hasDetail =
    !!(task.name && task.description && task.name.trim() !== task.description.trim());
  const hasOutput =
    (task.status === "done" || task.status === "failed") &&
    !!task.output &&
    Object.keys(task.output).length > 0;

  return (
    <div className="flex gap-3 items-start">
      <div className="flex flex-col items-center self-stretch pt-1">
        <TaskStatusIcon status={task.status} />
        {!isLast && <div className="w-px flex-1 bg-white/10 mt-1" />}
      </div>
      <div className="flex-1 min-w-0 pb-4">
        <div className="flex items-center gap-2">
          {hasOutput ? (
            <button
              onClick={() => setExpanded((v) => !v)}
              className="flex items-center gap-1 text-left min-w-0"
            >
              {expanded
                ? <ChevronDown className="h-3 w-3 text-slate-500 shrink-0" />
                : <ChevronRight className="h-3 w-3 text-slate-500 shrink-0" />}
              <p className={`text-sm font-medium truncate ${task.status === "pending" ? "text-slate-400" : "text-slate-100"}`}>
                {task.name || task.description || "(untitled task)"}
              </p>
            </button>
          ) : (
            <p className={`text-sm font-medium truncate ${task.status === "pending" ? "text-slate-400" : "text-slate-100"}`}>
              {task.name || task.description || "(untitled task)"}
            </p>
          )}
          <span className="text-[10px] uppercase tracking-wider font-mono text-slate-500 shrink-0">
            {task.kind}
          </span>
          {typeof task.elapsed === "number" &&
            (task.status === "running" || task.status === "done") && (
              <span className={`text-[10px] font-mono shrink-0 ${task.status === "running" ? "text-amber-400" : "text-slate-500"}`}>
                {formatElapsed(task.elapsed)}
              </span>
            )}
        </div>
        {hasDetail && (
          <p className="mt-1 text-xs text-slate-400 leading-relaxed whitespace-pre-wrap break-words">
            {task.description}
          </p>
        )}
        {task.summary && (task.status === "done" || task.status === "failed") && (
          <p
            className={`mt-0.5 text-[11px] font-mono break-words ${
              task.status === "failed" ? "text-amber-300/80" : "text-emerald-300/80"
            }`}
          >
            {task.summary}
          </p>
        )}
        {task.error && (
          <p className="mt-1 text-[11px] text-red-400 font-mono break-words">{task.error}</p>
        )}
        {task.judge && (
          <p className="mt-0.5 text-[11px] text-purple-300/80 font-mono flex items-start gap-1">
            <Gavel className="h-3 w-3 mt-0.5 shrink-0" />
            <span className="break-words">
              {typeof task.judge === "object"
                ? [
                    task.judge.verdict,
                    typeof task.judge.confidence === "number"
                      ? `(${(task.judge.confidence * 100).toFixed(0)}%)`
                      : null,
                    task.judge.rationale && task.judge.rationale !== "No criteria specified."
                      ? `— ${task.judge.rationale}`
                      : null,
                  ]
                    .filter(Boolean)
                    .join(" ")
                : String(task.judge)}
            </span>
          </p>
        )}
        {hasOutput && expanded && task.output && (
          <TaskOutputPanel output={task.output} kind={task.kind} />
        )}
      </div>
    </div>
  );
}

// ─── event stream ─────────────────────────────────────────────────────────────

function formatElapsed(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}m ${secs.toString().padStart(2, "0")}s`;
}

function LiveProgressLine({
  taskId: _tid,
  progress,
}: {
  taskId: string;
  progress: { name: string; kind: string; elapsed: number; at: number };
}) {
  const time = new Date(progress.at * 1000).toLocaleTimeString();
  const label = progress.kind ? `${progress.kind}: ` : "";
  const name = progress.name.length > 80 ? progress.name.slice(0, 80) + "…" : progress.name;
  return (
    <p className="leading-relaxed">
      <span className="text-slate-600">[{time}] </span>
      <span className="text-amber-400">[task progress]</span>{" "}
      <span className="text-slate-400">{label}{name}</span>
      <span className="text-amber-400/70"> · {formatElapsed(progress.elapsed)}</span>
    </p>
  );
}

function EventLine({ ev }: { ev: RawEvent }) {
  const label = ev.type.replace(/_/g, " ");
  const time = new Date(ev.at * 1000).toLocaleTimeString();
  const isRetry = ev.type === "retry_start" || ev.type === "retry_plan";
  return (
    <p className={`leading-relaxed ${isRetry ? "border-t border-amber-500/20 pt-1 mt-1" : ""}`}>
      <span className="text-slate-600">[{time}] </span>
      <span className={isRetry ? "text-amber-400" : "text-emerald-400"}>[{label}]</span>{" "}
      <span className={isRetry ? "text-amber-300/80" : "text-slate-400"}>
        {summarisePayload(ev.payload, ev.type)}
      </span>
    </p>
  );
}

function summarisePayload(p: any, ev: string): string {
  if (!p) return "";
  if (typeof p === "string") return p;
  if (typeof p !== "object") return String(p);
  if (ev === "plan") {
    const plan = p.plan || p;
    const goal = plan?.goal ? `goal=${plan.goal}` : "";
    const n = Array.isArray(plan?.tasks) ? plan.tasks.length : 0;
    return [goal, n ? `${n} task(s)` : ""].filter(Boolean).join(" · ");
  }
  if (ev === "task_start" || ev === "task_done" || ev === "task_skipped" || ev === "task_failed") {
    const kind = p.kind || p.task?.kind;
    const title = p.name || p.task?.name || p.description || p.task?.description;
    const extra = p.summary || p.error || p.reason;
    const head = kind && title ? `${kind}: ${title}` : title || `task ${p.task_id || ""}`;
    return extra ? `${head} — ${extra}` : head;
  }
  if (ev === "task_progress") {
    const kind = p.kind || "";
    const title = p.name || p.description || `task ${p.task_id || ""}`;
    const t = typeof p.elapsed === "number" ? ` · ${formatElapsed(p.elapsed)}` : "";
    return `${kind ? kind + ": " : ""}${title}${t}`;
  }
  if (ev === "judge") {
    const v = p.verdict || "?";
    const c = typeof p.confidence === "number" ? ` (${p.confidence.toFixed(2)})` : "";
    const r = p.rationale ? `: ${p.rationale}` : "";
    return `${v}${c}${r}`;
  }
  if (ev === "summary") {
    return `${p.completed ?? 0} completed · ${p.failed ?? 0} failed · ${p.skipped ?? 0} skipped`;
  }
  if (ev === "error") return String(p.message || p.error || "error");
  if (ev === "status") return `[${p.phase ?? ""}] ${p.message ?? ""}`.trim();
  if (ev === "retry_start") return String(p.reason || `attempt ${p.attempt ?? 2}`);
  if (ev === "retry_plan") {
    const plan = p.plan || p;
    const n = Array.isArray(plan?.tasks) ? plan.tasks.length : 0;
    return `fix plan ready · ${n} task(s)`;
  }
  const desc = p.description || p.message || p.summary || p.goal;
  if (desc) return String(desc);
  try { return JSON.stringify(p).slice(0, 240); } catch { return String(p); }
}

// ─── stage helpers ────────────────────────────────────────────────────────────

function computeStages(tasks: TaskRow[], phase: string) {
  if (tasks.length === 0) {
    return {
      planner: phase === "planning" ? "running" : "pending",
      tracker: "pending",
      judge: "pending",
    } as const;
  }
  // Collapse retried tasks: when a SCAFFOLD_FILE failed on the first
  // attempt and a retry-plan regenerated the same file, the latter row
  // supersedes the former. Without this, the stage cards would keep
  // showing red long after the failure was recovered.
  const latestByKey = new Map<string, TaskRow>();
  for (const t of tasks) {
    const key = `${t.kind}::${t.name || t.description}`;
    latestByKey.set(key, t);
  }
  const effective = Array.from(latestByKey.values());

  const anyRunning = tasks.some((t) => t.status === "running");
  const allDone = effective.every(
    (t) => t.status === "done" || t.status === "skipped",
  );
  const anyFailed = effective.some((t) => t.status === "failed");
  const anyJudge = tasks.some((t) => t.judge);
  // Judge is "running" only in the narrow window where a task just finished
  // (status === "done") but its judge verdict hasn't arrived yet.
  // It is idle — "pending" — while tasks are still executing.
  const anyDoneWithoutJudge = tasks.some((t) => t.status === "done" && !t.judge);
  const judgeState: "pending" | "running" | "done" | "failed" =
    anyFailed ? "failed" :
    allDone && anyJudge ? "done" :
    anyDoneWithoutJudge ? "running" :
    "pending";
  return {
    planner: "done",
    tracker: anyFailed ? "failed" : anyRunning ? "running" : allDone ? "done" : "running",
    judge: judgeState,
  } as const;
}
