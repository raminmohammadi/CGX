import { Card, CardHeader } from "../../components/Card";
import { BarList, Histogram, type BarRow, type HistBar } from "../../components/charts";
import { api, type HistogramSeries, type MetricSeries } from "../../lib/api";
import { ErrorLine, fmtNum, useAsync, type SectionProps } from "./common";

function sumBy(series: MetricSeries[], name: string): number {
  return series.filter((s) => s.name === name).reduce((t, s) => t + s.value, 0);
}

// Cumulative "le" buckets → per-bucket counts for a readable distribution.
function toBars(h: HistogramSeries): HistBar[] {
  let prev = 0;
  return h.buckets.map(([le, cum]) => {
    const v = Math.max(0, cum - prev);
    prev = cum;
    return { label: le === "+Inf" || le === Infinity ? "∞" : String(le), value: v };
  });
}

export default function MetricsSection({ refreshKey }: SectionProps) {
  const { data, error } = useAsync(() => api.adminMetrics(), [refreshKey]);
  const counters = data?.counters ?? [];
  const gauges = data?.gauges ?? [];
  const hists = data?.histograms ?? [];

  const latency = hists.find((h) => h.name.includes("latency"));
  const counterBars: BarRow[] = [...counters]
    .sort((a, b) => b.value - a.value)
    .slice(0, 10)
    .map((c) => ({
      label:
        c.name.replace(/^cgx_/, "") +
        (Object.keys(c.labels).length
          ? ` {${Object.entries(c.labels).map(([k, v]) => `${k}=${v}`).join(",")}}`
          : ""),
      value: c.value,
      tone: "blue" as const,
    }));

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <div className="grid grid-cols-4 gap-4">
        <MiniStat label="HTTP requests" value={sumBy(counters, "cgx_http_requests_total")} />
        <MiniStat label="LLM calls" value={sumBy(counters, "cgx_llm_calls_total")} />
        <MiniStat label="LLM tokens" value={sumBy(counters, "cgx_llm_tokens_total")} />
        <MiniStat label="Guardrail events" value={sumBy(counters, "cgx_guardrail_events_total")} />
      </div>

      <Card padded>
        <CardHeader
          eyebrow="Observability (B)"
          title="LLM call latency"
          description={
            latency
              ? `${fmtNum(latency.count)} calls · avg ${latency.count ? Math.round(latency.sum / latency.count) : 0}ms`
              : "No latency observations yet."
          }
        />
        <Histogram data={latency ? toBars(latency) : []} tone="emerald" height={140} />
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card padded>
          <CardHeader eyebrow="Registry" title="Top counters" description="RED + LLM + monitor series." />
          <BarList data={counterBars} format={(v) => fmtNum(v)} />
        </Card>
        <Card padded>
          <CardHeader eyebrow="Registry" title="Gauges" description="Point-in-time values." />
          <div className="space-y-1 max-h-[16rem] overflow-y-auto text-[11px] font-mono">
            {gauges.length === 0 && <p className="text-slate-500 italic">No gauges.</p>}
            {gauges.map((g, i) => (
              <div key={i} className="flex justify-between gap-3 bg-slate-950 p-2 rounded border border-white/5">
                <span className="text-slate-300 truncate">
                  {g.name.replace(/^cgx_/, "")}
                  <span className="text-slate-600">
                    {Object.keys(g.labels).length
                      ? ` {${Object.entries(g.labels).map(([k, v]) => `${k}=${v}`).join(",")}}`
                      : ""}
                  </span>
                </span>
                <span className="text-emerald-400 shrink-0">{g.value}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <p className="text-[10px] text-slate-500 font-mono">
        Prometheus scrape: <span className="text-slate-400">GET /api/metrics</span> · full time-series
        dashboards ship with the deploy stack (Grafana).
      </p>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: number }) {
  return (
    <Card padded>
      <p className="av-section-eyebrow mb-1">{label}</p>
      <p className="text-xl font-bold font-mono text-emerald-400">{fmtNum(value)}</p>
    </Card>
  );
}
