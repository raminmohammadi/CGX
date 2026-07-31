import { useState, type ReactNode } from "react";
import { ChevronDown, ChevronRight, Check, ShieldCheck, Trash2 } from "lucide-react";
import type { ProfileSummary } from "../../lib/api";
import { Card, CardHeader } from "../Card";
import { Pill } from "../Pill";
import { cn } from "../../lib/utils";
import { KIND_LABELS, type ProviderKind } from "./providerKinds";

const FILTERS: { key: ProviderKind | "all"; label: string }[] = [
  { key: "all", label: "All" },
  { key: "ollama", label: "Ollama" },
  { key: "openai-compat", label: "OpenAI" },
  { key: "gemini", label: "Gemini" },
  { key: "custom", label: "Custom" },
];

export function ProfilesTable({
  profiles,
  activeProfileName,
  onUse,
  onEdit,
  onDelete,
}: {
  profiles: ProfileSummary[];
  activeProfileName: string | null;
  onUse: (p: ProfileSummary) => void;
  onEdit: (p: ProfileSummary) => void;
  onDelete: (name: string) => void;
}) {
  const [filter, setFilter] = useState<ProviderKind | "all">("all");
  const [expanded, setExpanded] = useState<string | null>(null);

  const filtered = filter === "all" ? profiles : profiles.filter((p) => p.kind === filter);

  return (
    <Card padded>
      <CardHeader
        eyebrow="Saved profiles"
        title="Profiles"
        description="Select a profile to expand its details, or use the action buttons to switch, edit, or delete."
      />

      <div className="flex flex-wrap gap-1.5 mb-3">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className={cn(
              "px-2.5 py-1 rounded-full text-[10px] font-mono uppercase tracking-wider border transition",
              filter === f.key
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                : "bg-slate-950 text-slate-500 border-white/5 hover:text-slate-300",
            )}
          >
            {f.label}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <p className="text-xs text-slate-500 text-center py-4">
          {profiles.length === 0
            ? "No saved profiles yet -- click New profile above to save your current setup."
            : "No profiles match this filter."}
        </p>
      ) : (
        <div className="space-y-2">
          {filtered.map((p) => {
            const active = activeProfileName === p.name;
            const isOpen = expanded === p.name;
            return (
              <div
                key={p.name}
                className={cn(
                  "rounded-lg border transition-colors",
                  active ? "bg-emerald-500/5 border-emerald-500/20" : "bg-slate-950/60 border-white/5",
                )}
              >
                <button
                  onClick={() => setExpanded(isOpen ? null : p.name)}
                  className="w-full flex items-center gap-3 px-3 py-2.5 text-left"
                >
                  {isOpen ? (
                    <ChevronDown className="h-3 w-3 text-slate-500 shrink-0" />
                  ) : (
                    <ChevronRight className="h-3 w-3 text-slate-500 shrink-0" />
                  )}
                  <span className="text-xs font-semibold text-white truncate">{p.name}</span>
                  <span className="text-[10px] text-slate-500 font-mono truncate flex-1">
                    {KIND_LABELS[p.kind as ProviderKind]?.split(" ")[0] || p.kind} · {p.model}
                  </span>
                  <Pill tone={active ? "neon" : "slate"} className="text-[9px] shrink-0">
                    {active ? "ACTIVE" : "STANDBY"}
                  </Pill>
                </button>

                {isOpen && (
                  <div className="px-3 pb-3 space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                      <KV label="base_url" value={p.base_url || "--"} />
                      <KV label="temperature" value={p.temperature.toFixed(2)} />
                      <KV label="num_predict" value={String(p.num_predict)} />
                      <KV label="num_ctx" value={p.num_ctx != null ? String(p.num_ctx) : "auto"} />
                      {p.endpoint_path && <KV label="endpoint_path" value={p.endpoint_path} />}
                      <KV
                        label="api_key"
                        value={p.has_api_key ? "stored" : "--"}
                        icon={p.has_api_key ? <ShieldCheck className="h-3 w-3 text-emerald-400" /> : undefined}
                      />
                    </div>
                    <div className="flex items-center justify-end gap-1.5">
                      <button className="av-btn-primary py-1 px-2 text-[10px]" onClick={() => onUse(p)}>
                        <Check className="h-3 w-3" /> Use
                      </button>
                      <button className="av-btn-ghost py-1 px-2 text-[10px]" onClick={() => onEdit(p)}>
                        Edit
                      </button>
                      <button
                        className="av-btn py-1 px-2 text-[10px] bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20"
                        onClick={() => onDelete(p.name)}
                      >
                        <Trash2 className="h-3 w-3" /> Delete
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}

function KV({ label, value, icon }: { label: string; value: string; icon?: ReactNode }) {
  return (
    <div className="bg-slate-950 px-2.5 py-1.5 rounded border border-white/5 flex justify-between items-center gap-2">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-200 truncate flex items-center gap-1">
        {icon}
        {value}
      </span>
    </div>
  );
}
