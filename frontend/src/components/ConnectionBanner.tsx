import { AlertTriangle, RefreshCw } from "lucide-react";
import { useConnection } from "../store/connection";

// Persistent banner shown when the FastAPI backend is unreachable. Auto-hides
// the moment a poll succeeds again. The Retry button forces an immediate
// status fetch instead of waiting for the next scheduled tick.

export default function ConnectionBanner() {
  const offline = useConnection((s) => s.offline);
  const error = useConnection((s) => s.error);
  const isChecking = useConnection((s) => s.isChecking);
  const failures = useConnection((s) => s.consecutiveFailures);
  const refresh = useConnection((s) => s.refresh);

  if (!offline) return null;

  return (
    <div
      role="alert"
      className="flex items-center justify-between gap-3 px-6 py-2 border-b text-xs font-mono bg-red-500/10 border-red-500/30 text-red-200 flex-shrink-0"
    >
      <div className="flex items-center gap-2 min-w-0">
        <AlertTriangle className="h-3.5 w-3.5 text-red-300 flex-shrink-0" />
        <span className="text-red-100 font-semibold">API Offline</span>
        <span className="text-red-300/80 truncate">
          {error
            ? `cannot reach backend (${failures} failed checks): ${error}`
            : `cannot reach backend (${failures} failed checks)`}
        </span>
      </div>
      <button
        type="button"
        onClick={() => void refresh()}
        disabled={isChecking}
        className="flex items-center gap-1.5 px-2 py-1 rounded border border-red-400/30 hover:bg-red-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-red-100 flex-shrink-0"
      >
        <RefreshCw
          className={"h-3 w-3 " + (isChecking ? "animate-spin" : "")}
        />
        {isChecking ? "checking…" : "Retry"}
      </button>
    </div>
  );
}
