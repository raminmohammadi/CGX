import { useEffect, useState } from "react";
import { Card, CardHeader } from "../../components/Card";
import { Pill } from "../../components/Pill";
import { StatCard } from "../../components/StatCard";
import { Donut, type DonutSlice } from "../../components/charts";
import { api, type RunDetail, type RunRecord } from "../../lib/api";
import { cn, formatRelative } from "../../lib/utils";
import { ErrorLine, fmtCost, fmtMs, fmtNum, useAsync, type SectionProps } from "./common";

const KINDS = ["all", "ask", "plan", "agent"] as const;
const KIND_TONE: Record<string, DonutSlice["tone"]> = { ask: "emerald", plan: "blue", agent: "purple" };

export default function ActivitySection({ refreshKey }: SectionProps) {
  const [kind, setKind] = useState<(typeof KINDS)[number]>("all");
  const [selected, setSelected] = useState<RunDetail | null>(null);
  const { data, error } = useAsync(
    () =>
      Promise.all([
        api.activitySummary(),
        api.activityRuns({ kind: kind === "all" ? undefined : kind, limit: 200 }),
      ]),
    [refreshKey, kind],
  );
  const [summary, runsResp] = data ?? [];
  const runs = runsResp?.runs ?? [];
  useEffect(() => setSelected(null), [kind, refreshKey]);

  const kindSlices: DonutSlice[] = Object.entries(summary?.by_kind ?? {}).map(([k, v]) => ({
    label: k,
    value: v.runs,
    tone: KIND_TONE[k] ?? "slate",
  }));

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Runs" value={fmtNum(summary?.total)} tone="neon" />
        <StatCard label="Cost" value={fmtCost(summary?.cost_usd)} />
        <StatCard label="Tokens" value={fmtNum(summary?.tokens_total)} />
        <StatCard
          label="Errors"
          value={fmtNum(summary?.errors)}
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

      <div className="grid grid-cols-3 gap-4">
        <Card padded className="col-span-2">
          <CardHeader eyebrow="User activity (C)" title="Recent runs" description="Most recent first." />
          <div className="space-y-1 max-h-[26rem] overflow-y-auto">
            {runs.length === 0 && <p className="text-[11px] text-slate-500 font-mono italic">No runs recorded.</p>}
            {runs.map((r: RunRecord) => (
              <button
                key={r.run_id}
                onClick={() => api.activityRunDetail(r.run_id).then(setSelected).catch(() => {})}
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

        <div className="space-y-4">
          <Card padded>
            <CardHeader eyebrow="Mix" title="By kind" />
            <Donut data={kindSlices} centerValue={summary?.total ?? 0} centerLabel="runs" size={124} />
          </Card>
          <Card padded>
            <CardHeader eyebrow="Detail" title="Run" description={selected ? undefined : "Select a run."} />
            {selected && (
              <div className="space-y-1.5 text-[11px] font-mono">
                {(["run_id", "model", "prompt_version", "owner"] as const).map((k) => (
                  <Row key={k} label={k} value={String(selected.run[k] ?? "--")} />
                ))}
                <Row label="latency" value={fmtMs(selected.run.latency_ms)} />
                <Row label="tokens" value={String(selected.run.tokens_total ?? "--")} />
                <Row label="sources/cites" value={`${selected.run.n_sources}/${selected.run.n_citations}`} />
                <Row label="feedback" value={`${selected.feedback.length} rating(s)`} />
                <Row label="alerts" value={`${selected.alerts.length} alert(s)`} />
              </div>
            )}
          </Card>
        </div>
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
