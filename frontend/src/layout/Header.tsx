import { Cpu } from "lucide-react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { useConnection } from "../store/connection";
import { useWorkspace } from "../store/workspace";
import { useTrace } from "../store/trace";
import { StatusDot } from "../components/Pill";
import { Breadcrumb } from "../components/Breadcrumb";
import { cn } from "../lib/utils";
import { formatCtx, placementLabel, embedPillState, findActiveRunningModel } from "../lib/hardware";
import { api, type RunningModel, type SessionSummary } from "../lib/api";

// Top platform header: brand + breadcrumb + provider health pulse + mode badge.
// Reads from the shared connection store and active workspace provider.

const PROVIDER_LABELS: Record<string, string> = {
  ollama: "Ollama",
  gemini: "Gemini",
  "openai-compat": "OpenAI",
  custom: "Custom",
};

const PAGE_TITLES: Record<string, string> = {
  "/": "Overview",
  "/ask": "Ask",
  "/plan": "Plan",
  "/agent": "Agent",
  "/agent-legacy": "Agent (legacy)",
  "/index": "Index",
  "/settings": "Settings",
};

export default function Header() {
  const status = useConnection((s) => s.status);
  const offline = useConnection((s) => s.offline);
  const provider = useWorkspace((s) => s.provider);
  const traceSettings = useTrace((s) => s.settings);
  const setTrace = useTrace((s) => s.set);
  const [traceBusy, setTraceBusy] = useState(false);

  // Header trace toggle. The pill is always visible so tracing can be flipped
  // from anywhere without navigating to Settings (which owns the same control
  // via the shared store). Env-pinned (CGX_TRACE) state is non-interactive.
  const traceOn = !!traceSettings?.enabled;
  const traceEnvPinned = traceSettings?.source === "env";
  const traceLocked = traceSettings === null || traceEnvPinned || traceBusy;

  const toggleTrace = async () => {
    if (!traceSettings || traceEnvPinned || traceBusy) return;
    setTraceBusy(true);
    try {
      const updated = await api.setTraceSettings(!traceSettings.enabled);
      setTrace(updated);
    } catch {
      // Detailed errors are surfaced on the Settings page; keep the header quiet.
    } finally {
      setTraceBusy(false);
    }
  };

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
    : "--";

  // Match the active model against Ollama's currently-loaded set so the user
  // can see at a glance whether their picked model is actually resident, with
  // what effective context window and GPU/CPU placement.
  const running = (status?.ollama?.running_models || []) as RunningModel[];
  const activeRunning =
    isLocal && provider.model ? findActiveRunningModel(running, provider.model) : undefined;
  const placement = activeRunning ? placementLabel(activeRunning) : null;

  const modeLabel = isLocal ? "Local / Air-Gapped" : "Cloud";
  const modeClass = isLocal ? "text-emerald-400" : "text-sky-400";

  const embedPill = embedPillState(status?.hardware);
  const embedTextClass =
    embedPill?.tone === "neon"
      ? "text-emerald-300"
      : embedPill?.tone === "red"
        ? "text-red-300"
        : embedPill?.tone === "amber"
          ? "text-amber-300"
          : "text-slate-400";

  const location = useLocation();
  const pageTitle = PAGE_TITLES[location.pathname] || null;

  return (
    <header
      className="h-14 border-b flex items-center justify-between px-6 bg-header flex-shrink-0 gap-4"
      style={{ borderColor: "rgba(255,255,255,0.06)" }}
    >
      <div className="flex items-center gap-4 min-w-0">
        <div
          className="h-7 w-7 bg-emerald-500 rounded flex items-center justify-center text-slate-950 font-bold text-sm shadow-md shadow-emerald-500/10 shrink-0"
          title={`cgx.webui v${status?.version || "0.2.0"}`}
        >
          C
        </div>
        <Breadcrumb items={pageTitle ? ["CGX", pageTitle] : ["CGX"]} />
      </div>

      {location.pathname === "/ask" && <AskSessionDropdown />}

      <div className="flex items-center space-x-4 text-xs font-mono">
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
        {isLocal && ollamaOK && (
          <div
            className="flex items-center gap-2 bg-slate-950 px-2.5 py-1 rounded border border-white/5"
            title={
              activeRunning
                ? `Loaded in Ollama · ctx ${activeRunning.context_length ?? "?"} · ${placement?.label}`
                : "Active model is not currently resident in Ollama"
            }
          >
            <StatusDot tone={activeRunning ? "neon" : "slate" as any} />
            <span className="text-slate-400">
              Loaded:{" "}
              {activeRunning ? (
                <>
                  <span className="text-white">ctx {formatCtx(activeRunning.context_length)}</span>
                  <span className="text-slate-700"> · </span>
                  <span className={placement?.tone || "text-slate-300"}>
                    {placement?.label}
                  </span>
                </>
              ) : (
                <span className="text-slate-500">idle</span>
              )}
            </span>
          </div>
        )}
        {embedPill && (
          <div
            className="flex items-center gap-2 bg-slate-950 px-2.5 py-1 rounded border border-white/5"
            title={embedPill.title}
          >
            <StatusDot tone={embedPill.tone as any} />
            <span className={embedTextClass}>{embedPill.label}</span>
          </div>
        )}
        <div className="text-slate-400 flex items-center gap-1.5">
          <Cpu className="h-3.5 w-3.5 text-slate-600" /> Mode:{" "}
          <span className={modeClass}>{modeLabel}</span>
        </div>
        <button
          type="button"
          onClick={toggleTrace}
          disabled={traceLocked}
          title={
            traceSettings === null
              ? "Trace state loading…"
              : traceEnvPinned
                ? "Curated function-call tracing is pinned ON by CGX_TRACE; unset it to toggle from the UI"
                : traceOn
                  ? "Function-call tracing is ON — click to turn off"
                  : "Function-call tracing is OFF — click to turn on"
          }
          className={cn(
            "inline-flex items-center gap-1.5 px-2 py-0.5 rounded font-mono text-[10px] font-bold border uppercase tracking-wider transition-colors",
            traceOn
              ? "bg-amber-500/10 text-amber-400 border-amber-500/20 hover:bg-amber-500/20"
              : "bg-slate-800 text-slate-400 border-white/5 hover:bg-slate-700",
            traceLocked && "opacity-60 cursor-not-allowed",
          )}
        >
          <StatusDot tone={traceOn ? "amber" : "slate"} />
          TRACE {traceOn ? "ON" : "OFF"}
          {traceEnvPinned && " ·"}
        </button>
      </div>
    </header>
  );
}

// Thin "jump to session" dropdown shown only on /ask. The Sidebar already
// owns full session CRUD (create/delete/list); this is just a fast switcher
// so a session can be picked without leaving the header.
function AskSessionDropdown() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const selected = useWorkspace((s) => s.selectedSessionId);
  const setSelected = useWorkspace((s) => s.setSelectedSession);

  useEffect(() => {
    api.listSessions().then(setSessions).catch(() => setSessions([]));
  }, []);

  return (
    <select
      value={selected || ""}
      onChange={(e) => setSelected(e.target.value || null)}
      title="Jump to session"
      className="av-input appearance-none text-xs py-1 max-w-[220px] shrink-0"
    >
      <option value="">New session</option>
      {sessions.map((s) => (
        <option key={s.id} value={s.id}>
          {s.title || s.id.slice(0, 8)}
        </option>
      ))}
    </select>
  );
}
