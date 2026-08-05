// Duck-typed rather than importing lib/pullManager's PullState directly:
// the edit-modal's local pull-tracking state has the same shape minus
// `base_url`, which this component never needs.
export type PullProgressState = {
  model: string;
  status: string;
  total: number;
  completed: number;
  done: boolean;
  error: string | null;
  // Optional clean local name the model was re-aliased to after download.
  renamedTo?: string | null;
};

export function PullProgress({ pull, model }: { pull: PullProgressState | null; model: string }) {
  if (!pull || pull.model !== model) return null;
  const pct =
    pull.total > 0 ? Math.min(100, Math.round((pull.completed / pull.total) * 100)) : null;

  return (
    <div className="mt-1.5 space-y-1">
      <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
        {pull.error ? (
          <div className="h-full w-full bg-red-500/60" />
        ) : pull.done ? (
          <div className="h-full w-full bg-emerald-500" />
        ) : pct !== null ? (
          <div
            className="h-full bg-emerald-500 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="h-full w-1/3 bg-emerald-500/70 animate-pulse" />
        )}
      </div>
      <p className={`text-[10px] font-mono ${pull.error ? "text-red-400" : "text-slate-400"}`}>
        {pull.error
          ? pull.error.slice(0, 80)
          : pull.done
          ? pull.renamedTo
            ? `Download complete -- saved as ${pull.renamedTo}`
            : "Download complete"
          : pct !== null
          ? `${pull.status} -- ${pct}%`
          : pull.status}
      </p>
    </div>
  );
}
