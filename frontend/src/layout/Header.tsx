import { Cpu } from "lucide-react";
import { useConnection } from "../store/connection";
import { StatusDot } from "../components/Pill";

// Top platform header: brand + Ollama health pulse + air-gapped badge.
// Reads from the shared connection store; polling is owned by AppShell.

export default function Header() {
  const status = useConnection((s) => s.status);
  const offline = useConnection((s) => s.offline);
  const error = useConnection((s) => s.error);

  const ollamaOK = !offline && !!status?.ollama?.ok;
  const ollamaTone = ollamaOK
    ? "neon"
    : offline || error || status?.ollama?.error
      ? "red"
      : "amber";

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
          <StatusDot tone={ollamaTone as any} />
          <span className="text-slate-400">
            Ollama status:{" "}
            <span className={ollamaOK ? "text-white" : "text-amber-300"}>
              {ollamaOK
                ? "Connected"
                : offline
                  ? "Backend offline"
                  : "Disconnected"}
            </span>
          </span>
        </div>
        <div className="text-slate-400 flex items-center gap-1.5">
          <Cpu className="h-3.5 w-3.5 text-slate-600" /> Mode:{" "}
          <span className="text-emerald-400">Air-Gapped / Local-First</span>
        </div>
      </div>
    </header>
  );
}
