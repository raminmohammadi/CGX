import { CHART_COLORS, TRACK, toneAt, type ChartTone } from "./palette";

export type DonutSlice = { label: string; value: number; tone?: ChartTone };

// A ring chart built from stacked stroke-dasharray arcs on one circle. No
// external dependency — just SVG. The centre shows a headline value/label.
export function Donut({
  data,
  size = 148,
  thickness = 14,
  centerValue,
  centerLabel,
}: {
  data: DonutSlice[];
  size?: number;
  thickness?: number;
  centerValue?: string | number;
  centerLabel?: string;
}) {
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  const total = data.reduce((s, d) => s + Math.max(0, d.value), 0);
  let offset = 0;

  return (
    <div className="flex items-center gap-4">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={TRACK}
            strokeWidth={thickness}
          />
          {total > 0 &&
            data.map((d, i) => {
              const frac = Math.max(0, d.value) / total;
              const len = frac * c;
              const seg = (
                <circle
                  key={i}
                  cx={size / 2}
                  cy={size / 2}
                  r={r}
                  fill="none"
                  stroke={CHART_COLORS[d.tone ?? toneAt(i)]}
                  strokeWidth={thickness}
                  strokeDasharray={`${len} ${c - len}`}
                  strokeDashoffset={-offset}
                  strokeLinecap="butt"
                />
              );
              offset += len;
              return seg;
            })}
        </g>
        {(centerValue != null || centerLabel) && (
          <>
            <text
              x="50%"
              y="47%"
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-white font-mono font-bold"
              style={{ fontSize: size * 0.2 }}
            >
              {centerValue ?? ""}
            </text>
            {centerLabel && (
              <text
                x="50%"
                y="63%"
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-slate-500 font-mono uppercase"
                style={{ fontSize: 9, letterSpacing: 1 }}
              >
                {centerLabel}
              </text>
            )}
          </>
        )}
      </svg>
      <ul className="space-y-1 text-[11px] font-mono min-w-0">
        {data.length === 0 && <li className="text-slate-500 italic">no data</li>}
        {data.map((d, i) => (
          <li key={i} className="flex items-center gap-2">
            <span
              className="h-2 w-2 rounded-sm shrink-0"
              style={{ background: CHART_COLORS[d.tone ?? toneAt(i)] }}
            />
            <span className="text-slate-400 truncate">{d.label}</span>
            <span className="text-slate-200 ml-auto pl-2">{d.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
