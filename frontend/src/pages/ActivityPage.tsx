import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Card, CardHeader } from "../components/Card";
import { Pill } from "../components/Pill";
import { StatCard } from "../components/StatCard";
import { api, type ActivitySummary, type RunDetail, type RunRecord } from "../lib/api";
import { cn, formatRelative } from "../lib/utils";

const KINDS = ["all", "ask", "plan", "agent"] as const;

function fmtCost(v?: number | null) {
  return v == null ? "--" : `$${v.toFixed(4)}`;
}
function fmtMs(v?: number | null) {
  return v == null ? "--" : `${Math.round(v)}ms`;
}

export default function ActivityPage() {
  const [summary, setSummary] = useState<ActivitySummary | null>(null);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [kind, setKind] = useState<(typeof KINDS)[number]>("all");
  const [selected, setSelected] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, r] = await Promise.all([
        api.activitySummary(),
        api.activityRuns({ kind: kind === "all" ? undefined : kind, limit: 200 }),
      ]);
      setSummary(s);
      setRuns(r.runs);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }, [kind]);

  useEffect(() => {
    void load();
  }, [load]);

  const openDetail = async (runId: string) => {
    try {
      setSelected(await api.activityRunDetail(runId));
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  };

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-6xl">
      <CardHeader
        title="User Activity"
        description="Per-run observability — grounding, cost and feedback, from /api/activity."
        right={
          <button onClick={() => void load()} disabled={loading} className="av-btn-ghost">
            <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} /> Refresh
          </button>
        }
      />

      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Runs" value={summary?.total ?? "--"} tone="neon" />
        <StatCard label="Cost" value={fmtCost(summary?.cost_usd)} caption="all recorded runs" />
        <StatCard label="Tokens" value={summary?.tokens_total?.toLocaleString() ?? "--"} />
        <StatCard
          label="Errors"
          value={summary?.errors ?? "--"}
          tone={summary && summary.errors > 0 ? "red" : "slate"}
        />
      </div>

      <div className="flex items-center gap-2">
        {KINDS.map((k) => (
          <button
            key={k}
            onClick={() => setKind(k)}
            className={cn("av-btn-ghost capitalize", kind === k && "text-emerald-400")}
          >
            {k}
          </button>
        ))}
      </div>

      {error && <p className="text-xs text-red-400 font-mono">{error}</p>}

      <div className="grid grid-cols-3 gap-4">
        <Card padded className="col-span-2">
          <CardHeader eyebrow="Runs" title="Recent" description="Most recent first." />
          <div className="space-y-1 max-h-[28rem] overflow-y-auto">
            {runs.length === 0 && (
              <p className="text-[11px] text-slate-500 font-mono italic">No runs recorded yet.</p>
            )}
            {runs.map((r) => (
              <button
                key={r.run_id}
                onClick={() => void openDetail(r.run_id)}
                className={cn(
                  "w-full text-left p-2.5 rounded border text-xs font-mono flex items-center justify-between gap-3 transition",
                  selected?.run.run_id === r.run_id
                    ? "bg-slate-950 border-emerald-500/30"
                    : "bg-slate-950/40 border-white/5 hover:border-white/10",
                )}
              >
                <span className="flex items-center gap-2 truncate">
                  <Pill tone={r.kind === "ask" ? "neon" : r.kind === "plan" ? "purple" : "slate"}>{r.kind}</Pill>
                  <span className="text-slate-300 truncate">{r.question || r.run_id}</span>
                </span>
                <span className="flex items-center gap-2 shrink-0 text-slate-500">
                  {r.status !== "ok" && <Pill tone="red">{r.status}</Pill>}
                  {r.grounded === false && <Pill tone="amber">ungrounded</Pill>}
                  <span>{fmtCost(r.cost_usd)}</span>
                  <span>{formatRelative(r.created_at)}</span>
                </span>
              </button>
            ))}
          </div>
        </Card>

        <Card padded>
          <CardHeader eyebrow="Detail" title="Run" description={selected ? undefined : "Select a run."} />
          {selected && (
            <div className="space-y-2 text-[11px] font-mono">
              {(["run_id", "model", "prompt_version", "owner"] as const).map((k) => (
                <Row key={k} label={k} value={String(selected.run[k] ?? "--")} />
              ))}
              <Row label="latency" value={fmtMs(selected.run.latency_ms)} />
              <Row label="tokens" value={String(selected.run.tokens_total ?? "--")} />
              <Row label="sources/cites" value={`${selected.run.n_sources}/${selected.run.n_citations}`} />
              <Row label="feedback" value={`${selected.feedback.length} rating(s)`} />
              <Row label="alerts" value={`${selected.alerts.length} alert(s)`} />
              {selected.alerts.map((a, i) => (
                <p key={i} className="text-amber-400 truncate">• {a.code}: {a.message}</p>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-950 p-2 rounded border border-white/5 flex justify-between gap-3">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-200 truncate">{value}</span>
    </div>
  );
}
