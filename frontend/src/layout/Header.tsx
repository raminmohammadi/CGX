import { Cpu } from "lucide-react";
import { useConnection } from "../store/connection";
import { useWorkspace } from "../store/workspace";
import { StatusDot } from "../components/Pill";

// Top platform header: brand + provider health pulse + mode badge.
// Reads from the shared connection store and active workspace provider.

const PROVIDER_LABELS: Record<string, string> = {
  ollama: "Ollama",
  gemini: "Gemini",
  "openai-compat": "OpenAI",
  custom: "Custom",
};

export default function Header() {
  const status = useConnection((s) => s.status);
  const offline = useConnection((s) => s.offline);
  const provider = useWorkspace((s) => s.provider);

  const isLocal = provider.kind === "ollama";
  const ollamaOK = !offline && !!status?.ollama?.ok;

  // For local: use live Ollama health; for cloud: status is always "cloud"
  const statusTone = isLocal
    ? (ollamaOK ? "neon" : offline || status?.ollama?.error ? "red" : "amber")
    : "slate";

  const connectionLabel = isLocal
    ? (ollamaOK ? "Connected" : offline ? "Backend offline" : "Disconnected")
    : "Cloud";

  const providerLabel = PROVIDER_LABELS[provider.kind] || "Provider";
  const modelLabel = provider.model
    ? provider.model.length > 28
      ? provider.model.slice(0, 26) + "…"
      : provider.model
    : "—";

  const modeLabel = isLocal ? "Local / Air-Gapped" : "Cloud";
  const modeClass = isLocal ? "text-emerald-400" : "text-sky-400";

  return (
    <header
      className="h-14 border-b flex items-center justify-between px-6 bg-header flex-shrink-0"
      style={{ borderColor: "rgba(255,255,255,0.06)" }}
    >
      <div className="flex items-center space-x-3">
        <div className="h-7 w-7 bg-emerald-500 rounded flex items-center justify-center text-slate-950 font-bold text-sm shadow-md shadow-emerald-500/10">
          A
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="font-bold tracking-tight text-white text-base font-mono">
            AVERIX
          </span>
          <span className="text-[10px] text-slate-500 font-mono">
            cgx.webui v{status?.version || "0.2.0"}
          </span>
        </div>
      </div>

      <div className="flex items-center space-x-6 text-xs font-mono">
        <div className="flex items-center gap-2 bg-slate-950 px-2.5 py-1 rounded border border-white/5">
          <StatusDot tone={statusTone as any} />
          <span className="text-slate-400">
            {providerLabel}:{" "}
            <span className={statusTone === "neon" ? "text-white" : statusTone === "red" ? "text-red-300" : "text-amber-300"}>
              {connectionLabel}
            </span>
          </span>
          {provider.model && (
            <>
              <span className="text-slate-700">·</span>
              <span className="text-slate-300">{modelLabel}</span>
            </>
          )}
        </div>
        <div className="text-slate-400 flex items-center gap-1.5">
          <Cpu className="h-3.5 w-3.5 text-slate-600" /> Mode:{" "}
          <span className={modeClass}>{modeLabel}</span>
        </div>
      </div>
    </header>
  );
}
