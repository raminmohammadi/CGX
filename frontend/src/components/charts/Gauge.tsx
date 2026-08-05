import { CHART_COLORS, TRACK, type ChartTone } from "./palette";

// A radial ratio gauge (0..1). The arc fills clockwise from the top and the
// centre prints the percentage. Used for satisfaction, readiness, budget use.
export function Gauge({
  value,
  label,
  tone = "emerald",
  size = 132,
  thickness = 12,
  display,
}: {
  value: number | null | undefined;
  label?: string;
  tone?: ChartTone;
  size?: number;
  thickness?: number;
  display?: string;
}) {
  const v = value == null || Number.isNaN(value) ? null : Math.max(0, Math.min(1, value));
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  const len = (v ?? 0) * c;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={TRACK} strokeWidth={thickness} />
          {v != null && (
            <circle
              cx={size / 2}
              cy={size / 2}
              r={r}
              fill="none"
              stroke={CHART_COLORS[tone]}
              strokeWidth={thickness}
              strokeDasharray={`${len} ${c - len}`}
              strokeLinecap="round"
            />
          )}
        </g>
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-white font-mono font-bold"
          style={{ fontSize: size * 0.22 }}
        >
          {display ?? (v == null ? "--" : `${Math.round(v * 100)}%`)}
        </text>
      </svg>
      {label && (
        <span className="text-[10px] uppercase tracking-widest text-slate-500 font-mono">{label}</span>
      )}
    </div>
  );
}
