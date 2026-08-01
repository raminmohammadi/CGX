import { useState } from "react";
import { ChevronDown, ChevronRight, Pencil, Trash2 } from "lucide-react";
import { Card, CardHeader } from "../Card";
import { Pill } from "../Pill";
import { api, type SkillSummary } from "../../lib/api";

// Expandable card: clicking the header reveals the skill's actual source
// (lazily fetched and cached) so a user can read what it does before
// assigning it, not just its one-line description.
export function SkillCard({
  skill,
  onEdit,
  onDelete,
}: {
  skill: SkillSummary;
  onEdit?: () => void;
  onDelete?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [source, setSource] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (next && source === null && !loading) {
      setLoading(true);
      api
        .getSkillSource(skill.name)
        .then((r) => setSource(r.source))
        .catch((e) => setError(String(e?.message || e)))
        .finally(() => setLoading(false));
    }
  };

  return (
    <Card padded={false} className="overflow-hidden">
      <button onClick={toggle} className="w-full text-left p-5">
        <div className="flex items-start gap-2">
          {open ? (
            <ChevronDown className="h-3.5 w-3.5 text-slate-500 shrink-0 mt-1" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-slate-500 shrink-0 mt-1" />
          )}
          <div className="min-w-0 flex-1">
            <CardHeader
              eyebrow={skill.role}
              title={skill.name}
              right={
                <Pill tone={skill.is_custom ? "purple" : "slate"}>
                  {skill.is_custom ? "custom" : "built-in"}
                </Pill>
              }
            />
            <p className="text-xs text-slate-400">
              {skill.description || "No description provided."}
            </p>
            {skill.aliases.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {skill.aliases.map((a) => (
                  <span
                    key={a}
                    className="text-[9px] font-mono text-slate-500 bg-slate-950 border border-white/5 rounded px-1.5 py-0.5"
                  >
                    {a}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </button>

      {open && (
        <div className="px-5 pb-5 space-y-3">
          {loading && <p className="text-[11px] text-slate-500 font-mono">Loading source…</p>}
          {error && <p className="text-[11px] text-red-300 font-mono">{error}</p>}
          {source && (
            <pre className="bg-slate-950 border border-white/5 rounded-lg p-3 text-[10px] font-mono text-slate-300 whitespace-pre-wrap break-words max-h-96 overflow-y-auto">
              {source}
            </pre>
          )}
          {skill.is_custom && (onEdit || onDelete) && (
            <div className="flex items-center justify-end gap-1.5">
              {onEdit && (
                <button className="av-btn-ghost py-1 px-2 text-[10px]" onClick={onEdit}>
                  <Pencil className="h-3 w-3" /> Edit
                </button>
              )}
              {onDelete && (
                <button
                  className="av-btn py-1 px-2 text-[10px] bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20"
                  onClick={onDelete}
                >
                  <Trash2 className="h-3 w-3" /> Delete
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
