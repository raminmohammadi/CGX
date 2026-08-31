import { useCallback, useEffect, useState } from "react";
import { RefreshCw, ShieldCheck } from "lucide-react";
import { api, type ApprovalRequest } from "../lib/api";
import { Card, CardHeader } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { PendingApprovals } from "../components/PendingApprovals";

// Standalone view of every risky tool call awaiting a human decision across
// active sessions. Polls every few seconds so a request raised by a running
// swarm appears without a manual refresh; resolving one unblocks its worker.
const POLL_MS = 3000;

export default function ApprovalsPage() {
  const [items, setItems] = useState<ApprovalRequest[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await api.approvalsPending();
      setItems(res.pending);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
    const t = setInterval(() => void load(), POLL_MS);
    return () => clearInterval(t);
  }, [load]);

  const resolve = async (req: ApprovalRequest, approved: boolean) => {
    setBusyId(req.request_id);
    try {
      await api.approvalsResolve({
        session_id: req.session_id,
        request_id: req.request_id,
        approved,
      });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-4xl">
      <CardHeader
        eyebrow="Control"
        title="Tool approvals"
        description="Risky tool calls awaiting a human decision (code execution, file writes, MCP calls)."
        right={
          <button className="av-btn-ghost" onClick={() => void load()}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        }
      />
      {error && <p className="text-xs text-red-400 font-mono">{error}</p>}
      <Card padded>
        {items.length === 0 ? (
          <EmptyState
            icon={<ShieldCheck className="h-5 w-5" />}
            title="Nothing awaiting approval"
            description="When a session runs with approval enabled, risky tool calls appear here."
          />
        ) : (
          <PendingApprovals items={items} onResolve={resolve} busyId={busyId} />
        )}
      </Card>
    </div>
  );
}
