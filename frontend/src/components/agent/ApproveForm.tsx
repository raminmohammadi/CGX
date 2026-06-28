import { Check, X } from "lucide-react";
import type { ArtifactDTO } from "../../lib/api";
import { Markdown } from "../Markdown";
import { DiffView } from "../DiffView";

type DecidePayload = { chosen: Record<string, any>; rationale?: string };

// Approve-checkpoint UI: surfaces the CODE_CHANGE_PLAN artifact (plan
// markdown + unified diffs) and gates the loop on an explicit user
// approval. ``approved: false`` halts the write loop; ``true`` lets the
// router spawn the APPLY task.
export function ApproveForm({
  linked, onDecide, pending,
}: {
  linked: ArtifactDTO | null;
  onDecide: (p: DecidePayload) => any;
  pending: boolean;
}) {
  const planMd = String(linked?.content?.plan_md || "");
  const diffs: { file: string; patch: string }[] = linked?.content?.diffs || [];
  const confidence = linked?.content?.confidence;
  const citations: any[] = linked?.content?.citations || [];
  return (
    <div className="rounded-lg border border-orange-500/30 bg-orange-950/10 p-4 space-y-3">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <p className="text-[10px] uppercase tracking-wider font-mono text-orange-300">
          Approve change plan
        </p>
        <div className="flex items-center gap-3 text-[10px] font-mono text-slate-400">
          {diffs.length > 0 && (
            <span>{diffs.length} file{diffs.length === 1 ? "" : "s"}</span>
          )}
          {confidence !== undefined && confidence !== null && (
            <span>
              confidence: {typeof confidence === "number"
                ? `${(confidence * 100).toFixed(0)}%` : String(confidence)}
            </span>
          )}
        </div>
      </div>
      {planMd && (
        <div className="rounded-lg border border-white/5 bg-slate-950 p-3 max-h-72 overflow-y-auto">
          <Markdown text={planMd} />
        </div>
      )}
      {diffs.length > 0 && (
        <DiffView diff={diffs.map((d) => d.patch).join("\n")} />
      )}
      {diffs.length === 0 && !planMd && (
        <p className="text-[11px] text-slate-400 italic">Plan artifact is empty.</p>
      )}
      {citations.length > 0 && (
        <div className="text-[10px] font-mono text-slate-500">
          <span className="uppercase tracking-wider mr-1">cites</span>
          {citations.slice(0, 8).map((c, i) => (
            <span key={i} className="text-slate-400 mr-1.5">
              [[{typeof c === "string" ? c : c?.chunk_id || ""}]]
            </span>
          ))}
        </div>
      )}
      <div className="flex items-center gap-2 justify-end pt-1">
        <button
          type="button"
          disabled={pending}
          onClick={() => onDecide({ chosen: { approved: false } })}
          className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/5 px-3 py-1.5 text-[11px] font-mono text-slate-200 hover:bg-white/10 disabled:opacity-40"
        >
          <X className="h-3 w-3" /> Reject
        </button>
        <button
          type="button"
          disabled={pending}
          onClick={() => onDecide({ chosen: { approved: true } })}
          className="av-btn-primary"
        >
          <Check className="h-3 w-3" /> Approve &amp; Apply
        </button>
      </div>
    </div>
  );
}
