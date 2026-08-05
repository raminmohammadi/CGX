import type { ReactNode } from "react";
import { Card, CardHeader } from "../../components/Card";
import { BarList, Histogram, type BarRow, type HistBar } from "../../components/charts";
import type { HistogramSeries, MetricSeries, MetricsSnapshot } from "../../lib/api";
import { fmtCost, fmtMs, fmtNum } from "./common";

type MiniTone = "emerald" | "red" | "amber" | "slate";

// --- metric-snapshot helpers (shared by the live pipeline cards) ---
export function sumCounter(list: MetricSeries[], name: string): number {
  return list.filter((s) => s.name === name).reduce((t, s) => t + s.value, 0);
}
export function byLabel(list: MetricSeries[], name: string, key: string): Record<string, number> {
  const m: Record<string, number> = {};
  for (const s of list)
    if (s.name === name) {
      const k = String(s.labels[key] ?? "?");
      m[k] = (m[k] ?? 0) + s.value;
    }
  return m;
}
export function gaugeVal(list: MetricSeries[], name: string): number | null {
  const s = list.find((g) => g.name === name);
  return s ? s.value : null;
}
export function histStat(list: HistogramSeries[], name: string) {
  const hs = list.filter((h) => h.name === name);
  const count = hs.reduce((t, h) => t + h.count, 0);
  const sum = hs.reduce((t, h) => t + h.sum, 0);
  return { count, sum, avg: count ? sum / count : 0, series: hs[0] as HistogramSeries | undefined };
}
// Cumulative "le" buckets -> per-bucket counts for a readable distribution.
function toBars(h?: HistogramSeries): HistBar[] {
  if (!h) return [];
  let prev = 0;
  return h.buckets.map(([le, cum]) => {
    const v = Math.max(0, cum - prev);
    prev = cum;
    return { label: le === "+Inf" || le === Infinity ? "\u221e" : String(le), value: v };
  });
}

function Mini({ label, value, tone = "emerald" }: { label: string; value: ReactNode; tone?: MiniTone }) {
  const c =
    tone === "red" ? "text-red-400" : tone === "amber" ? "text-amber-400" : tone === "slate" ? "text-slate-200" : "text-emerald-400";
  return (
    <div className="bg-slate-950 p-2 rounded border border-white/5">
      <p className="av-section-eyebrow mb-0.5">{label}</p>
      <p className={`text-sm font-bold font-mono ${c}`}>{value}</p>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return <p className="text-[11px] text-slate-500 font-mono italic py-4 text-center">{text}</p>;
}

export function IndexingCard({ snap }: { snap: MetricsSnapshot }) {
  const builds = sumCounter(snap.counters, "cgx_index_builds_total");
  const dur = histStat(snap.histograms, "cgx_index_build_duration_ms");
  const records = gaugeVal(snap.gauges, "cgx_index_records");
  const files = gaugeVal(snap.gauges, "cgx_index_files");
  const byStatus = byLabel(snap.counters, "cgx_index_builds_total", "status");
  const bars: BarRow[] = Object.entries(byStatus).map(([label, value]) => ({
    label,
    value,
    tone: label === "ok" ? "emerald" : label === "error" ? "red" : "amber",
  }));
  return (
    <Card padded>
      <CardHeader eyebrow="Incremental indexing" title="Index builds" description="run_index_auto: parse → graph → records → FAISS." />
      <div className="grid grid-cols-4 gap-2 mb-3">
        <Mini label="builds" value={fmtNum(builds)} />
        <Mini label="records" value={records == null ? "--" : fmtNum(records)} tone="slate" />
        <Mini label="files" value={files == null ? "--" : fmtNum(files)} tone="slate" />
        <Mini label="avg build" value={dur.count ? fmtMs(dur.avg) : "--"} tone="amber" />
      </div>
      {dur.count ? <Histogram data={toBars(dur.series)} tone="blue" height={100} /> : <Empty text="No builds observed yet." />}
      {bars.length > 0 && (
        <div className="mt-2">
          <BarList data={bars} />
        </div>
      )}
    </Card>
  );
}

export function RetrievalCard({ snap }: { snap: MetricsSnapshot }) {
  const q = sumCounter(snap.counters, "cgx_retrieval_queries_total");
  const errs = byLabel(snap.counters, "cgx_retrieval_queries_total", "status")["error"] ?? 0;
  const lat = histStat(snap.histograms, "cgx_retrieval_latency_ms");
  const cand = histStat(snap.histograms, "cgx_retrieval_candidates");
  return (
    <Card padded>
      <CardHeader eyebrow="Hybrid retrieval" title="Retrieval queries" description="run_query_auto: semantic + lexical + graph, fused by RRF." />
      <div className="grid grid-cols-4 gap-2 mb-3">
        <Mini label="queries" value={fmtNum(q)} />
        <Mini label="avg latency" value={lat.count ? fmtMs(lat.avg) : "--"} tone="amber" />
        <Mini label="avg cands" value={cand.count ? Math.round(cand.avg) : "--"} tone="slate" />
        <Mini label="errors" value={fmtNum(errs)} tone={errs ? "red" : "slate"} />
      </div>
      {lat.count ? <Histogram data={toBars(lat.series)} tone="emerald" height={100} /> : <Empty text="No queries observed yet." />}
    </Card>
  );
}

export function ThroughputCard({
  eyebrow,
  title,
  description,
  entry,
}: {
  eyebrow: string;
  title: string;
  description: string;
  entry?: { runs: number; cost_usd: number; tokens_total: number; errors: number };
}) {
  return (
    <Card padded>
      <CardHeader eyebrow={eyebrow} title={title} description={description} />
      <div className="grid grid-cols-4 gap-2">
        <Mini label="runs" value={fmtNum(entry?.runs)} />
        <Mini label="cost" value={fmtCost(entry?.cost_usd)} tone="slate" />
        <Mini label="tokens" value={fmtNum(entry?.tokens_total)} tone="slate" />
        <Mini label="errors" value={fmtNum(entry?.errors)} tone={entry?.errors ? "red" : "slate"} />
      </div>
    </Card>
  );
}
