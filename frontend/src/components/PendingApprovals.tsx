import { ShieldAlert } from "lucide-react";
import type { ApprovalRequest, ApprovalRisk } from "../lib/api";
import { Pill, type PillTone } from "./Pill";

// Presentational list of pending human-in-the-loop approval requests with
// Approve / Deny actions. Shared by the standalone Approvals page and the live
// agent run view, so both surfaces render identically. Stateless: the caller
// owns fetching and the resolve call.

const RISK_TONE: Record<ApprovalRisk, PillTone> = {
  low: "slate",
  medium: "amber",
  high: "red",
};

export function PendingApprovals({
  items,
  onResolve,
  busyId,
  compact,
}: {
  items: ApprovalRequest[];
  onResolve: (req: ApprovalRequest, approved: boolean) => void;
  busyId?: string | null;
  compact?: boolean;
}) {
  if (items.length === 0) return null;
  return (
    <div className="space-y-2">
      {items.map((r) => {
        const busy = busyId === r.request_id;
        return (
          <div
            key={r.request_id}
            className="rounded-lg border border-amber-500/40 bg-amber-950/20 px-3 py-2.5"
          >
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-amber-400 shrink-0" />
              <span className="text-sm font-medium text-slate-100 truncate">
                {r.tool}
              </span>
              <Pill tone={RISK_TONE[r.risk] ?? "slate"}>{r.risk}</Pill>
              <div className="ml-auto flex items-center gap-1.5">
                <button
                  className="av-btn-ghost"
                  disabled={busy}
                  onClick={() => onResolve(r, false)}
                >
                  Deny
                </button>
                <button
                  className="av-btn-primary"
                  disabled={busy}
                  onClick={() => onResolve(r, true)}
                >
                  Approve
                </button>
              </div>
            </div>
            {!compact && (
              <pre className="mt-1.5 text-[11px] font-mono text-slate-400 whitespace-pre-wrap break-all">
                {JSON.stringify(r.args)}
              </pre>
            )}
          </div>
        );
      })}
    </div>
  );
}
