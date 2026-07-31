import type { ReactNode } from "react";
import { Card } from "./Card";
import { cn } from "../lib/utils";

export type StatTone = "neon" | "amber" | "red" | "slate";

const valueClasses: Record<StatTone, string> = {
  neon: "text-emerald-400",
  amber: "text-amber-400",
  red: "text-red-400",
  slate: "text-slate-200",
};

export function StatCard({
  label,
  value,
  caption,
  tone = "slate",
  className,
}: {
  label: string;
  value: ReactNode;
  caption?: ReactNode;
  tone?: StatTone;
  className?: string;
}) {
  return (
    <Card padded className={cn(tone === "neon" && "border-bright", className)}>
      <p className="av-section-eyebrow mb-1">{label}</p>
      <p className={cn("text-xl font-bold font-mono", valueClasses[tone])}>{value}</p>
      {caption && <p className="text-[10px] text-slate-500 mt-1 italic">{caption}</p>}
    </Card>
  );
}
