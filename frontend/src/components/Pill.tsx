import type { ReactNode } from "react";
import { cn } from "../lib/utils";

export type PillTone = "neon" | "amber" | "red" | "slate" | "purple";

const toneClasses: Record<PillTone, string> = {
  neon: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  amber: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  red: "bg-red-500/10 text-red-400 border-red-500/20",
  slate: "bg-slate-800 text-slate-400 border-white/5",
  purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
};

export function Pill({
  tone = "neon",
  children,
  className,
}: {
  tone?: PillTone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-1.5 py-0.5 rounded font-mono text-[10px] font-bold border uppercase tracking-wider",
        toneClasses[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function StatusDot({ tone = "neon" }: { tone?: "neon" | "amber" | "red" | "slate" }) {
  const bg =
    tone === "neon"
      ? "bg-emerald-500"
      : tone === "amber"
        ? "bg-amber-400"
        : tone === "red"
          ? "bg-red-400"
          : "bg-slate-500";
  return <span className={cn("h-1.5 w-1.5 rounded-full animate-pulse", bg)} />;
}
