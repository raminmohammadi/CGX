import type { AgentSessionState, FactDTO, TaskNodeDTO } from "../../lib/api";

// Client-side rollups for the live agent dashboard. All derived from the
// session snapshot + streamed facts we already hold, so no backend change is
// needed for a first cut: LLM cost/tokens/latency live inside ``llm_call``
// facts (attributed to a task via ``surfaced_in_task_id``) and swarm
// sub-agent progress lives inside the ``swarm_beat`` facts the fixed
// ``swarm_log`` now persists.

export interface PerTaskTelemetry {
  tokens: number;
  cost: number;
  calls: number;
  latencyMs: number;
}

export interface SwarmTelemetry {
  filesDone: number;
  filesTotal: number;
  filesFailed: number;
  lastRole: string | null;
  lastPhase: string | null;
}

export interface SessionTelemetry {
  total: number;        // work tasks (excludes ask_user gates)
  done: number;
  failed: number;
  inProgress: number;
  ready: number;
  blocked: number;
  completion: number;   // 0..1 over work tasks
  statusCounts: Record<string, number>;
  tokensTotal: number;
  costUsd: number;
  llmCalls: number;
  avgLatencyMs: number | null;
  perTask: Record<string, PerTaskTelemetry>;
  swarm: SwarmTelemetry | null;
}

const WORK_EXCLUDED = new Set(["ask_user"]);

function num(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

export function deriveSessionTelemetry(state: AgentSessionState): SessionTelemetry {
  const tasks = state.tasks || [];
  const facts = state.facts || [];

  const statusCounts: Record<string, number> = {};
  let total = 0, done = 0, failed = 0, inProgress = 0, ready = 0, blocked = 0;
  for (const t of tasks) {
    statusCounts[t.status] = (statusCounts[t.status] || 0) + 1;
    if (WORK_EXCLUDED.has(t.kind)) continue;
    total += 1;
    if (t.status === "done") done += 1;
    else if (t.status === "failed") failed += 1;
    else if (t.status === "in_progress") inProgress += 1;
    else if (t.status === "ready") ready += 1;
    else if (t.status === "blocked") blocked += 1;
  }

  // LLM telemetry from llm_call facts, rolled up globally and per task.
  const perTask: Record<string, PerTaskTelemetry> = {};
  let tokensTotal = 0, costUsd = 0, llmCalls = 0, latencySum = 0, latencyN = 0;
  for (const f of facts) {
    if (f.kind !== "llm_call") continue;
    const c = f.content || {};
    const tokens = num(c.tokens_total) || num(c.tokens_in) + num(c.tokens_out);
    const cost = num(c.cost_usd);
    const latency = num(c.latency_ms);
    tokensTotal += tokens;
    costUsd += cost;
    llmCalls += 1;
    if (latency > 0) { latencySum += latency; latencyN += 1; }
    const tid = f.surfaced_in_task_id;
    if (tid) {
      const p = perTask[tid] || { tokens: 0, cost: 0, calls: 0, latencyMs: 0 };
      p.tokens += tokens; p.cost += cost; p.calls += 1; p.latencyMs += latency;
      perTask[tid] = p;
    }
  }

  const swarm = deriveSwarm(facts, tasks);

  return {
    total, done, failed, inProgress, ready, blocked,
    completion: total > 0 ? done / total : 0,
    statusCounts,
    tokensTotal,
    costUsd,
    llmCalls,
    avgLatencyMs: latencyN > 0 ? latencySum / latencyN : null,
    perTask,
    swarm,
  };
}

// Swarm developer progress from ``swarm_beat`` facts. Developer beats carry
// ``index``/``total`` (one file per turn) and a ``write`` beat carries ``ok``;
// we take the largest declared total and count successful/failed writes.
function deriveSwarm(facts: FactDTO[], tasks: TaskNodeDTO[]): SwarmTelemetry | null {
  const isSwarmSession = tasks.some((t) => t.kind.startsWith("swarm_"));
  const beats = facts
    .filter((f) => f.kind === "swarm_beat")
    .sort((a, b) => a.created_at - b.created_at);
  if (!isSwarmSession && beats.length === 0) return null;

  let filesTotal = 0, filesDone = 0, filesFailed = 0;
  let lastRole: string | null = null, lastPhase: string | null = null;
  for (const b of beats) {
    const c = b.content || {};
    filesTotal = Math.max(filesTotal, num(c.total));
    if (c.phase === "write") {
      if (c.ok === true) filesDone += 1;
      else if (c.ok === false) filesFailed += 1;
    }
    if (c.role) lastRole = String(c.role);
    if (c.phase) lastPhase = String(c.phase);
  }
  // Fall back to task-derived file totals if beats didn't declare one.
  if (filesTotal === 0) {
    filesTotal = tasks.filter((t) => t.kind === "swarm_developer").length;
    filesDone = tasks.filter((t) => t.kind === "swarm_developer" && t.status === "done").length;
  }
  return { filesDone, filesTotal, filesFailed, lastRole, lastPhase };
}

// Compact human formatting shared by the dashboard tiles.
export function formatTokens(n: number): string {
  if (n <= 0) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function formatCost(n: number): string {
  if (n <= 0) return "$0";
  if (n < 0.01) return "<$0.01";
  return `$${n.toFixed(2)}`;
}

// ``123456`` ms → ``2m 3s``; ``4200`` → ``4.2s``; ``850`` → ``850ms``.
export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || ms <= 0) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return `${m}m ${rem}s`;
}
