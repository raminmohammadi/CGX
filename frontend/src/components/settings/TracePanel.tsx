import { Card, CardHeader } from "../Card";
import { Pill } from "../Pill";
import type { TraceSettings } from "../../lib/api";

export function TracePanel({
  traceSettings,
  traceBusy,
  traceError,
  onToggle,
}: {
  traceSettings: TraceSettings | null;
  traceBusy: boolean;
  traceError: string | null;
  onToggle: (next: boolean) => void;
}) {
  return (
    <Card padded>
      <CardHeader
        eyebrow="Observability"
        title="Function-call tracing"
        description="When on, the agent emits enter/exit records for every curated entry point (router, executors, LLM calls, retrieval, codegen, HTTP) into <project>/.cgx/agent.log."
        right={
          <div className="flex items-center gap-2">
            {traceSettings?.enabled && <Pill tone="amber">TRACE</Pill>}
            <button
              className="av-btn"
              disabled={traceBusy || traceSettings === null || traceSettings.source === "env"}
              onClick={() => traceSettings && onToggle(!traceSettings.enabled)}
            >
              {traceSettings === null ? "Loading…" : traceSettings.enabled ? "Turn off" : "Turn on"}
            </button>
          </div>
        }
      />
      {traceSettings?.source === "env" && (
        <p className="text-[11px] text-amber-300 font-mono mt-2">
          Pinned by <code>CGX_TRACE</code> environment variable; unset it to control from the UI.
        </p>
      )}
      {traceError && <p className="text-[11px] text-red-300 font-mono mt-2">{traceError}</p>}
    </Card>
  );
}
