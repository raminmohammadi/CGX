// Shared colour palette for the dependency-free SVG charts. Values mirror
// the Tailwind tokens used across the design system (index.css) so charts
// sit flush with cards, pills and buttons on the dark base.

export type ChartTone =
  | "emerald"
  | "amber"
  | "red"
  | "purple"
  | "blue"
  | "slate";

export const CHART_COLORS: Record<ChartTone, string> = {
  emerald: "#10b981",
  amber: "#f59e0b",
  red: "#ef4444",
  purple: "#a855f7",
  blue: "#3b82f6",
  slate: "#64748b",
};

// A stable rotation so categorical series (run kinds, alert codes, owners)
// get distinct, repeatable colours without a caller having to pick them.
export const TONE_CYCLE: ChartTone[] = [
  "emerald",
  "blue",
  "purple",
  "amber",
  "red",
  "slate",
];

export function toneAt(i: number): ChartTone {
  return TONE_CYCLE[i % TONE_CYCLE.length];
}

// Neutral track colour drawn under every filled arc / bar.
export const TRACK = "rgba(255,255,255,0.06)";
