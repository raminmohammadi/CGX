import { CHART_COLORS, type ChartTone } from "./palette";

// A compact area+line sparkline. Points are evenly spaced across the width;
// the y-axis auto-scales to the series range. Renders nothing meaningful for
// <2 points (shows a flat baseline) so callers don't have to guard.
export function Sparkline({
  data,
  tone = "emerald",
  width = 240,
  height = 48,
  strokeWidth = 1.5,
}: {
  data: number[];
  tone?: ChartTone;
  width?: number;
  height?: number;
  strokeWidth?: number;
}) {
  const color = CHART_COLORS[tone];
  const pad = 2;
  const n = data.length;
  const min = Math.min(...data, 0);
  const max = Math.max(...data, 1);
  const span = max - min || 1;
  const x = (i: number) => (n <= 1 ? pad : pad + (i * (width - 2 * pad)) / (n - 1));
  const y = (v: number) => height - pad - ((v - min) / span) * (height - 2 * pad);

  const pts = data.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  const line = n === 0 ? "" : `M ${pts.join(" L ")}`;
  const area =
    n === 0
      ? ""
      : `M ${x(0).toFixed(1)},${height - pad} L ${pts.join(" L ")} L ${x(n - 1).toFixed(1)},${height - pad} Z`;
  const gid = `spark-${tone}`;

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {area && <path d={area} fill={`url(#${gid})`} />}
      {line && <path d={line} fill="none" stroke={color} strokeWidth={strokeWidth} />}
    </svg>
  );
}
