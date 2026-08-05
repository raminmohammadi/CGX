import { ThumbsDown, ThumbsUp } from "lucide-react";
import { Card, CardHeader } from "../../components/Card";
import { StatCard } from "../../components/StatCard";
import { BarList, Donut, Gauge, type BarRow } from "../../components/charts";
import { api } from "../../lib/api";
import { formatRelative } from "../../lib/utils";
import { ErrorLine, fmtPct, useAsync, type SectionProps } from "./common";

export default function FeedbackSection({ refreshKey }: SectionProps) {
  const { data, error } = useAsync(
    () => Promise.all([api.feedbackStats(), api.feedbackList({ limit: 100 })]),
    [refreshKey],
  );
  const [stats, list] = data ?? [];
  const rows = list?.feedback ?? [];

  const kindBars: BarRow[] = Object.entries(stats?.by_kind ?? {}).flatMap(([k, v]) => [
    { label: `${k} ▲`, value: v.up, tone: "emerald" as const },
    { label: `${k} ▼`, value: v.down, tone: "red" as const },
  ]);

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Ratings" value={stats?.total ?? "--"} tone="neon" />
        <StatCard label="Up" value={stats?.up ?? "--"} tone="neon" />
        <StatCard label="Down" value={stats?.down ?? "--"} tone={stats && stats.down > 0 ? "red" : "slate"} />
        <StatCard label="Satisfaction" value={fmtPct(stats?.satisfaction)} />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card padded className="flex items-center justify-center">
          <Gauge value={stats?.satisfaction ?? null} label="Satisfaction (H)" />
        </Card>
        <Card padded className="flex items-center justify-center">
          <Donut
            data={[
              { label: "up", value: stats?.up ?? 0, tone: "emerald" },
              { label: "down", value: stats?.down ?? 0, tone: "red" },
            ]}
            centerValue={stats?.total ?? 0}
            centerLabel="ratings"
          />
        </Card>
        <Card padded>
          <CardHeader eyebrow="By kind" title="Up vs down" />
          <BarList data={kindBars} />
        </Card>
      </div>

      <Card padded>
        <CardHeader
          eyebrow="Feedback loop (H)"
          title="Recent ratings"
          description="Down-votes drain into the eval-candidate flywheel (Subsystem E)."
        />
        <div className="space-y-1 max-h-[24rem] overflow-y-auto text-[11px] font-mono">
          {rows.length === 0 && <p className="text-slate-500 italic">No feedback recorded.</p>}
          {rows.map((r) => (
            <div key={r.feedback_id} className="flex items-start gap-2 bg-slate-950 p-2 rounded border border-white/5">
              {r.rating === "up" ? (
                <ThumbsUp className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
              ) : (
                <ThumbsDown className="h-3.5 w-3.5 text-red-400 shrink-0 mt-0.5" />
              )}
              <span className="text-slate-500 shrink-0">{r.kind}</span>
              <span className="text-slate-300 truncate flex-1">
                {r.comment || r.question || r.run_id || "(no comment)"}
              </span>
              <span className="text-slate-600 shrink-0">{formatRelative(r.created_at)}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
