import { CHART_COLORS, toneAt, type ChartTone } from "./palette";

export type BarRow = {
  label: string;
  value: number;
  tone?: ChartTone;
  sub?: string;
  // Optional per-row ceiling (e.g. a budget limit) drawn as a faint marker.
  limit?: number | null;
};

// Horizontal bar list — the workhorse for "top N by X" tables (cost by owner,
// alerts by code, tokens by kind). Bars are proportional to the max value.
export function BarList({
  data,
  format = (v) => String(v),
  max: maxProp,
}: {
  data: BarRow[];
  format?: (v: number) => string;
  max?: number;
}) {
  const max = maxProp ?? Math.max(1, ...data.map((d) => Math.max(d.value, d.limit ?? 0)));
  return (
    <div className="space-y-2">
      {data.length === 0 && <p className="text-[11px] text-slate-500 font-mono italic">no data</p>}
      {data.map((d, i) => {
        const color = CHART_COLORS[d.tone ?? toneAt(i)];
        const pct = Math.max(0, Math.min(100, (d.value / max) * 100));
        const limitPct =
          d.limit && d.limit > 0 ? Math.max(0, Math.min(100, (d.limit / max) * 100)) : null;
        return (
          <div key={i}>
            <div className="flex items-baseline justify-between gap-2 text-[11px] font-mono mb-0.5">
              <span className="text-slate-300 truncate">{d.label}</span>
              <span className="text-slate-200 shrink-0">
                {format(d.value)}
                {d.sub && <span className="text-slate-500"> · {d.sub}</span>}
              </span>
            </div>
            <div className="relative h-2 rounded bg-slate-950 border border-white/5 overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 rounded"
                style={{ width: `${pct}%`, background: color }}
              />
              {limitPct != null && (
                <div
                  className="absolute inset-y-0 w-px bg-white/40"
                  style={{ left: `${limitPct}%` }}
                  title="limit"
                />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
