import { Wifi, WifiOff } from "lucide-react";
import type { PingResult } from "../../lib/api";

export function PingBanner({
  result,
  model,
  className,
}: {
  result: PingResult | null;
  model: string;
  className?: string;
}) {
  if (!result) return null;

  if (result.ok) {
    return (
      <div
        className={`mt-3 flex items-start gap-3 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-xs font-mono ${className ?? ""}`}
      >
        <Wifi className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
        <div>
          <p className="text-emerald-400 font-semibold">Ping successful</p>
          <p className="text-slate-400 mt-0.5">
            Connected to <span className="text-white">{model}</span> in{" "}
            <span className="text-white">{result.latency_ms?.toFixed(0)}ms</span>. Provider is reachable and responding.
          </p>
        </div>
      </div>
    );
  }

  const errMsg = result.error || "Unknown error";
  const isNotInstalled =
    errMsg.toLowerCase().includes("not found") ||
    errMsg.toLowerCase().includes("model") ||
    errMsg.toLowerCase().includes("pull");

  return (
    <div
      className={`mt-3 flex items-start gap-3 rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-3 text-xs font-mono ${className ?? ""}`}
    >
      <WifiOff className="h-4 w-4 text-red-400 shrink-0 mt-0.5" />
      <div>
        <p className="text-red-400 font-semibold">
          {isNotInstalled ? "Model not available" : "Ping failed"}
        </p>
        <p className="text-slate-400 mt-0.5 break-all">{errMsg}</p>
        {isNotInstalled && (
          <p className="text-slate-500 mt-1">
            Use the <strong className="text-slate-300">Pull</strong> button to download this model first.
          </p>
        )}
      </div>
    </div>
  );
}
