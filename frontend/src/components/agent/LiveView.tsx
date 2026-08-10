import {
  History, PanelLeftClose, PanelRightClose,
  Plus, RefreshCw, Send, Square, Trash2,
} from "lucide-react";
import type {
  AgentSessionState, AgentSessionSummary, SessionModeValue, TaskNodeDTO,
  TaskProgress,
} from "../../lib/api";
import { TaskTree } from "./TaskTree";
import { ActiveTaskPanel } from "./ActiveTask";
import { SidePanel } from "./SidePanel";
import { ResizeHandle } from "./ResizeHandle";
import { CollapsedRail } from "../CollapsedRail";
import { TextArea } from "../Input";
import { ErrorBoundary } from "../ErrorBoundary";
import { cn, formatRelative } from "../../lib/utils";
import { useAgentSession } from "../../store/agentSession";

// Pick the "most interesting" task for the centre pane: an open ASK
// outranks a running executor, which outranks the most recent terminal
// node. Used both to default-select on first load and to route blind
// decision posts when no task is explicitly selected.
export function activeTask(state: AgentSessionState): TaskNodeDTO | null {
  const tasks = state.tasks;
  const askOpen = tasks.find(
    (t) => t.kind === "ask_user" && t.status === "in_progress");
  if (askOpen) return askOpen;
  const running = tasks.find((t) => t.status === "in_progress");
  if (running) return running;
  const ready = tasks.find((t) => t.status === "ready");
  if (ready) return ready;
  const sorted = [...tasks].sort(
    (a, b) => (b.completed_at || 0) - (a.completed_at || 0));
  return sorted[0] ?? null;
}

export function LiveView({
  state, sessions, pending, error, progress, running,
  runModel, selectedModel,
  reply, setReply,
  selectedTaskId, setSelectedTaskId,
  onDecide, onSend, onCancel, onSwitch, onNew, onRefresh, onDelete,
}: {
  state: AgentSessionState;
  sessions: AgentSessionSummary[];
  pending: boolean;
  error: string | null;
  progress?: Record<string, TaskProgress>;
  running?: boolean;
  // Model the in-flight run was dispatched with, and the currently-selected
  // profile model. When a run is active and these differ, the header surfaces
  // a note that the selection applies to the next run (the provider is frozen
  // into the running drain and can't be hot-swapped).
  runModel?: string | null;
  selectedModel?: string;
  reply: string;
  setReply: (s: string) => void;
  selectedTaskId: string | null;
  setSelectedTaskId: (id: string | null) => void;
  onDecide: (payload: { chosen: Record<string, any>; rationale?: string }) => any;
  onSend: () => void;
  onCancel?: () => void | Promise<void>;
  onSwitch: (sid: string) => void;
  onNew: () => void;
  onRefresh: () => void;
  onDelete?: (sid: string) => void | Promise<void>;
}) {
  const auto = activeTask(state);
  const focused = selectedTaskId
    ? state.tasks.find((t) => t.task_id === selectedTaskId) || auto
    : auto;
  const {
    sessionBarWidth, sessionBarCollapsed, setSessionBarWidth, setSessionBarCollapsed,
    taskTreeWidth, setTaskTreeWidth,
    sidePanelWidth, sidePanelCollapsed, setSidePanelWidth, setSidePanelCollapsed,
  } = useAgentSession();
  // Only meaningful while a run is in flight: the provider is frozen into the
  // drain at dispatch, so a profile switch mid-run applies only to the next.
  const providerMismatch = !!running && !!runModel && !!selectedModel
    && runModel !== selectedModel;
  return (
    <div className="flex h-full w-full overflow-hidden">
      {sessionBarCollapsed ? (
        <CollapsedRail
          side="left"
          label="Sessions"
          onExpand={() => setSessionBarCollapsed(false)}
        />
      ) : (
        <>
          <SessionBar
            state={state}
            sessions={sessions}
            width={sessionBarWidth}
            onCollapse={() => setSessionBarCollapsed(true)}
            onSwitch={onSwitch}
            onNew={onNew}
            onRefresh={onRefresh}
            onDelete={onDelete}
          />
          <ResizeHandle
            direction="left"
            ariaLabel="Resize sessions panel"
            getCurrent={() => sessionBarWidth}
            onResize={setSessionBarWidth}
          />
        </>
      )}
      <div className="flex-1 flex flex-col bg-surface border-l border-r border-muted min-w-0">
        <header className="px-4 py-2.5 border-b border-muted bg-slate-950/40">
          <div className="flex items-center gap-2">
            <p className="text-[10px] uppercase tracking-[0.18em] font-mono text-slate-500">
              Objective
            </p>
            <ModeBadge mode={state.session.mode} />
          </div>
          <p className="text-[13px] text-slate-100 truncate">
            {state.session.original_objective}
          </p>
          {providerMismatch && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[10px] font-mono text-amber-300">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-400" />
              <span>
                Run uses <span className="text-amber-200">{runModel}</span>
                {" · selected "}
                <span className="text-amber-200">{selectedModel}</span>
                {" — applies to the next run"}
              </span>
            </div>
          )}
        </header>
        <div className="flex-1 flex overflow-hidden min-h-0">
          <div
            style={{ width: taskTreeWidth }}
            className="shrink-0 border-r border-muted overflow-y-auto px-2 py-3"
          >
            <TaskTree
              tasks={state.tasks}
              rootTaskId={state.session.root_task_id}
              selectedId={focused?.task_id ?? null}
              onSelect={setSelectedTaskId}
            />
          </div>
          <ResizeHandle
            direction="left"
            ariaLabel="Resize task tree"
            getCurrent={() => taskTreeWidth}
            onResize={setTaskTreeWidth}
          />
          <div className="flex-1 overflow-y-auto px-5 py-4 min-w-0">
            {error && (
              <div className="mb-3 rounded-lg border border-red-500/30 bg-red-950/30 p-3 text-[11px] font-mono text-red-300 whitespace-pre-wrap">
                {error}
              </div>
            )}
            {focused && progress?.[focused.task_id]
              && focused.status === "in_progress" && (
              <LiveProgress p={progress[focused.task_id]} taskKind={focused.kind} />
            )}
            <ErrorBoundary label="active-task">
              <ActiveTaskPanel
                task={focused ?? null}
                artifacts={state.artifacts}
                decisions={state.decisions}
                facts={state.facts}
                onDecide={onDecide}
                pending={pending}
              />
            </ErrorBoundary>
          </div>
        </div>
        <FollowUpBar
          reply={reply} setReply={setReply}
          onSend={onSend} pending={pending}
          running={!!running} onCancel={onCancel}
        />
      </div>
      {sidePanelCollapsed ? (
        <CollapsedRail
          side="right"
          label="Artifacts"
          onExpand={() => setSidePanelCollapsed(false)}
        />
      ) : (
        <>
          <ResizeHandle
            direction="right"
            ariaLabel="Resize artifacts panel"
            getCurrent={() => sidePanelWidth}
            onResize={setSidePanelWidth}
          />
          <ErrorBoundary label="side-panel">
            <SidePanel
              facts={state.facts}
              artifacts={state.artifacts}
              decisions={state.decisions}
              width={sidePanelWidth}
              onCollapse={() => setSidePanelCollapsed(true)}
              onSelectArtifact={(taskId) => setSelectedTaskId(taskId)}
            />
          </ErrorBoundary>
        </>
      )}
    </div>
  );
}

function SessionBar({
  state, sessions, width, onSwitch, onNew, onRefresh, onDelete, onCollapse,
}: {
  state: AgentSessionState;
  sessions: AgentSessionSummary[];
  width: number;
  onSwitch: (sid: string) => void;
  onNew: () => void;
  onRefresh: () => void;
  onDelete?: (sid: string) => void | Promise<void>;
  onCollapse: () => void;
}) {
  const current = state.session.session_id;
  return (
    <aside
      style={{ width }}
      className="shrink-0 bg-slate-950/40 flex flex-col min-w-0"
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-muted">
        <span className="av-section-eyebrow">Sessions</span>
        <div className="flex items-center gap-1">
          <button type="button" onClick={onRefresh} title="Refresh"
                  className="av-btn-icon h-5 w-5"><RefreshCw className="h-3 w-3" /></button>
          <button type="button" onClick={onNew} title="New session"
                  className="av-btn-icon h-5 w-5"><Plus className="h-3 w-3" /></button>
          <button type="button" onClick={onCollapse} title="Hide sessions panel"
                  className="av-btn-icon h-5 w-5"><PanelLeftClose className="h-3 w-3" /></button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {sessions.length === 0 && (
          <p className="text-[10px] text-slate-500 italic font-mono px-1 py-2">
            <History className="h-3 w-3 inline mr-1" /> No prior sessions.
          </p>
        )}
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={cn(
              "group relative rounded border transition",
              s.session_id === current
                ? "bg-emerald-500/10 border-emerald-500/30"
                : "bg-slate-950/40 border-white/5 hover:border-white/10",
            )}
          >
            <button
              type="button" onClick={() => onSwitch(s.session_id)}
              className="w-full text-left px-2 py-1.5 pr-7"
            >
              <p className="text-[11px] text-slate-100 truncate">{s.title || s.session_id.slice(0, 8)}</p>
              <p className="text-[9px] font-mono text-slate-500">
                {s.status} · {formatRelative(s.updated_at)}
              </p>
            </button>
            {onDelete && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  const label = s.title || s.session_id.slice(0, 8);
                  if (window.confirm(`Delete session "${label}"? This cannot be undone.`)) {
                    void onDelete(s.session_id);
                  }
                }}
                title="Delete session"
                className="absolute top-1.5 right-1.5 av-btn-icon h-5 w-5 opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-400"
              ><Trash2 className="h-3 w-3" /></button>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
}

function FollowUpBar({
  reply, setReply, onSend, pending, running, onCancel,
}: {
  reply: string; setReply: (s: string) => void; onSend: () => void;
  pending: boolean; running?: boolean;
  onCancel?: () => void | Promise<void>;
}) {
  return (
    <div className="px-4 pt-2.5 pb-3 border-t border-muted bg-slate-950/60 flex items-end gap-2">
      <TextArea
        rows={2}
        value={reply}
        onChange={(e) => setReply(e.target.value)}
        placeholder="Post a follow-up objective…"
        disabled={pending}
        className="flex-1"
      />
      {running && onCancel && (
        <button
          type="button" onClick={() => { void onCancel(); }}
          title="Stop after the current step"
          className="av-btn-ghost text-red-300 hover:text-red-200"
        ><Square className="h-3 w-3" /> Stop</button>
      )}
      <button
        type="button" onClick={onSend}
        disabled={pending || !reply.trim()}
        className="av-btn-primary"
      ><Send className="h-3 w-3" /> Send</button>
    </div>
  );
}

export function PriorSessions({
  sessions, onPick, onDelete,
}: {
  sessions: AgentSessionSummary[];
  onPick: (sid: string) => void;
  onDelete?: (sid: string) => void | Promise<void>;
}) {
  if (sessions.length === 0) return null;
  return (
    <div className="max-w-2xl mx-auto w-full px-6 py-6 space-y-2">
      <p className="av-section-eyebrow">Resume a prior session</p>
      <ul className="space-y-1.5">
        {sessions.map((s) => (
          <li key={s.session_id}>
            <div className="group relative rounded-md border border-white/10 bg-slate-950/60 hover:border-emerald-500/30">
              <button
                type="button" onClick={() => onPick(s.session_id)}
                className="w-full text-left px-3 py-2 pr-9"
              >
                <p className="text-[12px] text-slate-100 truncate">{s.title || s.session_id.slice(0, 8)}</p>
                <p className="text-[10px] font-mono text-slate-500">
                  {s.status} · {formatRelative(s.updated_at)}
                </p>
              </button>
              {onDelete && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    const label = s.title || s.session_id.slice(0, 8);
                    if (window.confirm(`Delete session "${label}"? This cannot be undone.`)) {
                      void onDelete(s.session_id);
                    }
                  }}
                  title="Delete session"
                  className="absolute top-2 right-2 av-btn-icon h-6 w-6 opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-400"
                ><Trash2 className="h-3.5 w-3.5" /></button>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}


// Live scaffold progress banner. Fed by ``task.output_partial`` SSE
// frames so a long, otherwise-silent generation shows an advancing
// count, the file in flight, and a coarse ETA instead of a spinner.
// A ``failed`` beat is rendered distinctly (amber) and the running
// failed tally is always shown so a failed file is never mistaken for a
// silent counter reset (on failure ``index`` does not advance).
function LiveProgress({ p, taskKind }: { p: TaskProgress; taskKind?: string }) {
  const pct = p.total > 0
    ? Math.min(100, Math.round((p.index / p.total) * 100)) : 0;
  const failedCount = p.failed_count ?? 0;
  const isFailed = p.status === "failed";
  
  let prefix = "Generating";
  if (taskKind === "SWARM_TECH_LEAD") prefix = "Planning";
  if (taskKind === "SWARM_VERIFY") prefix = "Verifying";
  
  return (
    <div className={`mb-3 rounded-lg border p-3 ${isFailed
      ? "border-amber-500/40 bg-amber-950/20"
      : "border-emerald-500/30 bg-emerald-950/20"}`}>
      <div className={`flex items-center justify-between text-[11px] font-mono ${isFailed ? "text-amber-300" : "text-emerald-300"}`}>
        <span>
          {prefix} {p.index}/{p.total}
          {failedCount > 0 && (
            <span className="ml-2 text-amber-400">
              · {failedCount} failed
            </span>
          )}
        </span>
        {p.eta_seconds != null
          ? <span>~{p.eta_seconds}s left</span>
          : (p.status === "stream" && p.bytes != null
              && <span>{p.bytes} chars…</span>)}
      </div>
      {isFailed
        ? (
          <p className="mt-1 text-[10px] font-mono text-amber-400 truncate">
            ⚠ Failed: {p.path} — skipping to next file
          </p>
        )
        : (
          <p className="mt-1 text-[10px] font-mono text-slate-400 truncate">{p.path}</p>
        )}
      <div className="mt-2 h-1.5 rounded bg-slate-800 overflow-hidden">
        <div
          className={`h-full transition-all ${isFailed ? "bg-amber-500" : "bg-emerald-500"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}


function ModeBadge({ mode }: { mode: SessionModeValue | undefined }) {
  const tone = mode === "greenfield"
    ? "border-emerald-500/30 bg-emerald-950/40 text-emerald-300"
    : mode === "swarm"
    ? "border-amber-500/30 bg-amber-950/40 text-amber-300"
    : "border-indigo-500/30 bg-indigo-950/40 text-indigo-300";
  const label = mode === "greenfield"
    ? "greenfield"
    : mode === "swarm" ? "swarm" : "explore";
  return (
    <span className={cn(
      "text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border",
      tone,
    )}>{label}</span>
  );
}
