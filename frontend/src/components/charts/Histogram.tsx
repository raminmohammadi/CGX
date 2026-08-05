import { CHART_COLORS, type ChartTone } from "./palette";

export type HistBar = { label: string; value: number };

// Vertical bar chart for bucketed distributions (e.g. LLM latency histogram
// buckets). Bars scale to the tallest value; labels sit under each column.
export function Histogram({
  data,
  tone = "blue",
  height = 120,
}: {
  data: HistBar[];
  tone?: ChartTone;
  height?: number;
}) {
  const color = CHART_COLORS[tone];
  const max = Math.max(1, ...data.map((d) => d.value));
  return (
    <div>
      <div className="flex items-end gap-1" style={{ height }}>
        {data.length === 0 && (
          <p className="text-[11px] text-slate-500 font-mono italic self-center">no data</p>
        )}
        {data.map((d, i) => {
          const h = (d.value / max) * 100;
          return (
            <div key={i} className="flex-1 flex flex-col items-center justify-end h-full group">
              <span className="text-[9px] font-mono text-slate-500 mb-0.5 opacity-0 group-hover:opacity-100 transition">
                {d.value}
              </span>
              <div
                className="w-full rounded-t"
                style={{
                  height: `${Math.max(d.value > 0 ? 3 : 0, h)}%`,
                  background: color,
                  opacity: 0.85,
                }}
                title={`${d.label}: ${d.value}`}
              />
            </div>
          );
        })}
      </div>
      <div className="flex gap-1 mt-1">
        {data.map((d, i) => (
          <span
            key={i}
            className="flex-1 text-center text-[8px] font-mono text-slate-600 truncate"
          >
            {d.label}
          </span>
        ))}
      </div>
    </div>
  );
}
