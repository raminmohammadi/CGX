import { cn } from "../lib/utils";

export function TabBar<T extends string>({
  tabs,
  active,
  onChange,
}: {
  tabs: { key: T; label: string }[];
  active: T;
  onChange: (key: T) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {tabs.map((t) => (
        <button
          key={t.key}
          onClick={() => onChange(t.key)}
          className={cn(
            "px-2.5 py-1 rounded-full text-[10px] font-mono uppercase tracking-wider border transition",
            active === t.key
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
              : "bg-slate-950 text-slate-500 border-white/5 hover:text-slate-300",
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
