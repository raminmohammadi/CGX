import { PanelLeftOpen, PanelRightOpen } from "lucide-react";
import { cn } from "../lib/utils";

// Slim 28px rail rendered in place of a collapsed side panel. A single
// icon button restores the panel; the rotated label hints at what lives
// behind it. Shared across any page with a collapsible side/nav panel
// (the Agent Run view's Sessions/Artifacts panels, the Agent page's own
// category nav, ...).
export function CollapsedRail({
  side,
  label,
  onExpand,
}: {
  side: "left" | "right";
  label: string;
  onExpand: () => void;
}) {
  const Icon = side === "left" ? PanelLeftOpen : PanelRightOpen;
  return (
    <aside className={cn(
      "w-7 shrink-0 bg-slate-950/40 flex flex-col items-center py-2 gap-2",
      side === "left" ? "border-r" : "border-l",
      "border-muted",
    )}>
      <button
        type="button"
        onClick={onExpand}
        title={`Show ${label.toLowerCase()} panel`}
        className="av-btn-icon h-5 w-5"
      ><Icon className="h-3 w-3" /></button>
      <span
        className="text-[10px] font-mono uppercase tracking-widest text-slate-500 select-none"
        style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
      >{label}</span>
    </aside>
  );
}
