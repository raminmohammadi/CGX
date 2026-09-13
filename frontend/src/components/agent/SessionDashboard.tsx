import { useMemo } from "react";
import { Coins, Cpu, Gauge as GaugeIcon, Timer } from "lucide-react";
import type { AgentSessionState } from "../../lib/api";
import { Donut, Gauge, type DonutSlice, type ChartTone } from "../charts";
import {
  deriveSessionTelemetry, formatCost, formatDuration, formatTokens,
} from "./sessionTelemetry";

// A compact, updates-in-place telemetry strip for the running session: an
// overall-completion gauge, a task-status donut, live LLM cost/token/latency
// tiles, and (for swarm runs) a sub-agent file-progress bar. Replaces the
// "read a growing log to guess how it's going" experience with a board that
// mutates as SSE events land.

const STATUS_TONE: Record<string, ChartTone> = {
  done: "emerald",
  in_progress: "amber",
  failed: "red",
  ready: "blue",
  blocked: "slate",
  pending: "slate",
  abandoned: "slate",
};
const STATUS_ORDER = ["in_progress", "done", "ready", "failed", "blocked", "pending", "abandoned"];

export function SessionDashboard({ state }: { state: AgentSessionState }) {
  const t = useMemo(() => deriveSessionTelemetry(state), [state]);

  const slices: DonutSlice[] = STATUS_ORDER
    .filter((s) => (t.statusCounts[s] || 0) > 0)
    .map((s) => ({ label: s.replace("_", " "), value: t.statusCounts[s], tone: STATUS_TONE[s] }));

  const swarm = t.swarm;
  const swarmPct = swarm && swarm.filesTotal > 0
    ? Math.min(100, Math.round((swarm.filesDone / swarm.filesTotal) * 100))
    : 0;

  return (
    <div className="px-4 py-3 border-b border-muted bg-slate-950/30">
      <div className="flex items-stretch gap-3 flex-wrap">
        {/* Overall completion */}
        <Panel>
          <Gauge
            value={t.completion}
            size={84}
            thickness={9}
            label="complete"
            display={`${t.done}/${t.total || 0}`}
            tone={t.failed > 0 && t.done + t.failed === t.total ? "amber" : "emerald"}
          />
        </Panel>

        {/* Task-status mix */}
        <Panel>
          <Donut data={slices} size={92} thickness={11} centerValue={t.total} centerLabel="tasks" />
        </Panel>

        {/* Live LLM telemetry tiles */}
        <Panel grow>
          <div className="grid grid-cols-2 gap-2 h-full content-center">
            <Tile icon={<Cpu className="h-3 w-3" />} label="tokens" value={formatTokens(t.tokensTotal)} />
            <Tile icon={<Coins className="h-3 w-3" />} label="cost" value={formatCost(t.costUsd)} />
            <Tile icon={<GaugeIcon className="h-3 w-3" />} label="llm calls" value={String(t.llmCalls)} />
            <Tile icon={<Timer className="h-3 w-3" />} label="avg latency" value={formatDuration(t.avgLatencyMs)} />
          </div>
        </Panel>

        {/* Swarm sub-agent progress (only for swarm sessions) */}
        {swarm && (
          <Panel grow>
            <div className="flex flex-col justify-center h-full gap-1.5 min-w-[160px]">
              <div className="flex items-baseline justify-between text-[10px] font-mono">
                <span className="uppercase tracking-widest text-slate-500">Developer</span>
                <span className="text-slate-300">
                  {swarm.filesDone}/{swarm.filesTotal || "?"} files
                  {swarm.filesFailed > 0 && <span className="text-red-400"> · {swarm.filesFailed} failed</span>}
                </span>
              </div>
              <div className="h-2 rounded bg-slate-900 border border-white/5 overflow-hidden">
                <div className="h-full bg-emerald-500 transition-all duration-500" style={{ width: `${swarmPct}%` }} />
              </div>
              {(swarm.lastRole || swarm.lastPhase) && (
                <p className="text-[10px] font-mono text-slate-500 truncate">
                  {swarm.lastRole || "agent"} · {swarm.lastPhase || "…"}
                </p>
              )}
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}

function Panel({ children, grow }: { children: React.ReactNode; grow?: boolean }) {
  return (
    <div
      className={
        "rounded-lg border border-white/5 bg-slate-950/50 px-3 py-2 flex items-center justify-center " +
        (grow ? "flex-1 min-w-[220px]" : "")
      }
    >
      {children}
    </div>
  );
}

function Tile({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-slate-500 shrink-0">{icon}</span>
      <div className="min-w-0">
        <p className="text-[9px] uppercase tracking-widest text-slate-500 font-mono">{label}</p>
        <p className="text-sm font-mono font-bold text-slate-100 truncate">{value}</p>
      </div>
    </div>
  );
}
