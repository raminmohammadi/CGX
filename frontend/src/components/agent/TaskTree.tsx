import { Fragment, useEffect, useState } from "react";
import {
  Circle, CircleCheck, CircleDashed, CircleX, Loader2, MinusCircle,
} from "lucide-react";
import type { TaskKind, TaskNodeDTO, TaskNodeStatus } from "../../lib/api";
import { cn, formatRelative } from "../../lib/utils";
import { formatDuration } from "./sessionTelemetry";

// Tailwind palette per task kind. Mirrors the legacy AgentPage palette
// so the two views feel related without being identical.
const KIND_META: Record<TaskKind, { badge: string; label: string }> = {
  explore:               { badge: "bg-sky-900/80 text-sky-300",         label: "explore" },
  investigate:           { badge: "bg-indigo-900/80 text-indigo-300",   label: "investigate" },
  recommend:             { badge: "bg-purple-900/80 text-purple-300",   label: "recommend" },
  plan_change:           { badge: "bg-orange-900/80 text-orange-300",   label: "plan_change" },
  apply:                 { badge: "bg-emerald-900/80 text-emerald-300", label: "apply" },
  verify:                { badge: "bg-cyan-900/80 text-cyan-300",       label: "verify" },
  ask_user:              { badge: "bg-fuchsia-900/80 text-fuchsia-300", label: "ask_user" },
  search:                { badge: "bg-slate-700 text-slate-300",        label: "search" },
  summarize:             { badge: "bg-slate-700 text-slate-300",        label: "summarize" },
  clarify_requirements:  { badge: "bg-emerald-900/80 text-emerald-300", label: "clarify" },
  decompose:             { badge: "bg-teal-900/80 text-teal-300",       label: "decompose" },
  scaffold:              { badge: "bg-lime-900/80 text-lime-300",       label: "scaffold" },
  bootstrap_env:         { badge: "bg-amber-900/80 text-amber-300",     label: "bootstrap" },
  repair:                { badge: "bg-rose-900/80 text-rose-300",       label: "repair" },
  swarm_assess:          { badge: "bg-amber-900/80 text-amber-300",     label: "assess" },
  swarm_tech_lead:       { badge: "bg-fuchsia-900/80 text-fuchsia-300", label: "tech lead" },
  swarm_developer:       { badge: "bg-emerald-900/80 text-emerald-300", label: "developer" },
  swarm_verify:          { badge: "bg-cyan-900/80 text-cyan-300",       label: "swarm verify" },
};

// Cap how deep the tree indents. Swarm developer tasks are siblings under the
// Tech Lead, so real depth stays shallow; this only guards against a
// pathological chain (the runaway-indentation bug fixed in the router).
const MAX_INDENT_DEPTH = 6;

// A left "status rail" accent per node so the tree reads as a live board at a
// glance rather than a flat outline.
const STATUS_ACCENT: Record<TaskNodeStatus, string> = {
  done: "border-l-emerald-500/50",
  failed: "border-l-red-500/60",
  in_progress: "border-l-amber-400/70",
  ready: "border-l-emerald-500/30",
  blocked: "border-l-slate-600/40",
  pending: "border-l-slate-700/40",
  abandoned: "border-l-slate-700/30",
};

// Tick once a second while any task is in flight so the running node's elapsed
// timer advances live; idle otherwise so a settled tree doesn't re-render.
// Returns "now" in epoch seconds to match the task timestamps.
function useLiveClock(tasks: TaskNodeDTO[]): number {
  const anyRunning = tasks.some((t) => t.status === "in_progress");
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!anyRunning) return;
    setNow(Date.now() / 1000);
    const id = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(id);
  }, [anyRunning]);
  return now;
}

// Wall-clock a node has spent (ms), or null when it hasn't started. Uses the
// live clock while running so the number ticks up in place.
function taskElapsedMs(task: TaskNodeDTO, nowSec: number): number | null {
  if (task.started_at == null) return null;
  const end = task.completed_at ?? (task.status === "in_progress" ? nowSec : null);
  if (end == null) return null;
  return Math.max(0, (end - task.started_at) * 1000);
}

function StatusIcon({ status }: { status: TaskNodeStatus }) {
  const base = "h-3.5 w-3.5 shrink-0";
  if (status === "done")        return <CircleCheck className={`${base} text-emerald-400`} />;
  if (status === "failed")      return <CircleX     className={`${base} text-red-400`} />;
  if (status === "in_progress") return <Loader2     className={`${base} text-amber-400 animate-spin`} />;
  if (status === "ready")       return <Circle      className={`${base} text-emerald-500`} />;
  if (status === "blocked")     return <MinusCircle className={`${base} text-slate-500`} />;
  if (status === "abandoned")   return <CircleDashed className={`${base} text-slate-500`} />;
  return <Circle className={`${base} text-slate-600`} />;
}

interface TaskTreeProps {
  tasks: TaskNodeDTO[];
  rootTaskId: string | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

// Build adjacency by parent_task_id; tasks without an explicit parent
// (or whose parent is missing) are surfaced as additional roots so the
// tree never silently drops nodes the router spawned out-of-band.
function groupChildren(tasks: TaskNodeDTO[]): Map<string | null, TaskNodeDTO[]> {
  const idx = new Map<string | null, TaskNodeDTO[]>();
  const ids = new Set(tasks.map((t) => t.task_id));
  for (const t of tasks) {
    const key = t.parent_task_id && ids.has(t.parent_task_id)
      ? t.parent_task_id : null;
    const bucket = idx.get(key) ?? [];
    bucket.push(t);
    idx.set(key, bucket);
  }
  for (const bucket of idx.values()) {
    bucket.sort((a, b) => a.created_at - b.created_at);
  }
  return idx;
}

export function TaskTree({ tasks, rootTaskId, selectedId, onSelect }: TaskTreeProps) {
  const now = useLiveClock(tasks);
  if (tasks.length === 0) {
    return (
      <p className="text-[11px] text-slate-500 font-mono italic px-2 py-4">
        No tasks yet -- start a session below.
      </p>
    );
  }
  const children = groupChildren(tasks);
  const topLevel = children.get(null) ?? [];
  // Pin the declared root first when present so the visual order matches
  // the session creation order even if more roots get spawned later.
  const sortedTop = rootTaskId
    ? [
        ...topLevel.filter((t) => t.task_id === rootTaskId),
        ...topLevel.filter((t) => t.task_id !== rootTaskId),
      ]
    : topLevel;
  return (
    <div className="space-y-0.5">
      {sortedTop.map((t) => (
        <TreeNode
          key={t.task_id}
          task={t}
          depth={0}
          children={children}
          selectedId={selectedId}
          onSelect={onSelect}
          now={now}
        />
      ))}
    </div>
  );
}

function TreeNode({
  task, depth, children, selectedId, onSelect, now,
}: {
  task: TaskNodeDTO;
  depth: number;
  children: Map<string | null, TaskNodeDTO[]>;
  selectedId: string | null;
  onSelect: (id: string) => void;
  now: number;
}) {
  const meta = KIND_META[task.kind] ?? KIND_META.search;
  const kids = children.get(task.task_id) ?? [];
  const isSelected = selectedId === task.task_id;
  const isPendingAsk = task.kind === "ask_user" && task.status === "in_progress";
  const isRunning = task.status === "in_progress";
  const elapsed = taskElapsedMs(task, now);
  return (
    <Fragment>
      <button
        type="button"
        onClick={() => onSelect(task.task_id)}
        // Clamp indentation so a deep (or accidentally chained) task graph can
        // never march off the right edge; beyond MAX_INDENT_DEPTH levels the
        // padding stops growing.
        style={{ paddingLeft: 8 + Math.min(depth, MAX_INDENT_DEPTH) * 14 }}
        className={cn(
          "w-full flex items-center gap-2 py-1.5 pr-2 rounded text-left transition-colors border-l-2",
          STATUS_ACCENT[task.status] ?? "border-l-transparent",
          "hover:bg-white/5",
          isRunning && "bg-amber-500/5",
          isSelected && "bg-emerald-500/10 ring-1 ring-emerald-500/30",
          isPendingAsk && !isSelected && "bg-fuchsia-500/5",
        )}
        title={task.description || task.name}
      >
        <StatusIcon status={task.status} />
        <span
          className={cn(
            "text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded shrink-0",
            meta.badge,
          )}
        >
          {meta.label}
        </span>
        <span className="text-[12px] text-slate-200 truncate flex-1">
          {task.name || task.description || task.kind}
        </span>
        {elapsed != null ? (
          <span
            className={cn(
              "text-[10px] font-mono shrink-0 tabular-nums",
              isRunning ? "text-amber-300" : "text-slate-500",
            )}
            title={task.completed_at ? formatRelative(task.completed_at) : "running"}
          >
            {formatDuration(elapsed)}
          </span>
        ) : task.completed_at ? (
          <span className="text-[10px] font-mono text-slate-500 shrink-0">
            {formatRelative(task.completed_at)}
          </span>
        ) : null}
      </button>
      {kids.map((c) => (
        <TreeNode
          key={c.task_id}
          task={c}
          depth={depth + 1}
          children={children}
          selectedId={selectedId}
          onSelect={onSelect}
          now={now}
        />
      ))}
    </Fragment>
  );
}
