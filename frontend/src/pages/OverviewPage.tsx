import { useState, type ReactNode } from "react";
import { Link } from "react-router";
import { RefreshCw, Wifi, WifiOff } from "lucide-react";
import { useConnection } from "../store/connection";
import { useWorkspace } from "../store/workspace";
import { Card, CardHeader } from "../components/Card";
import { Pill, StatusDot } from "../components/Pill";
import { StatCard } from "../components/StatCard";
import { api, type PingResult } from "../lib/api";
import { embedPillState, findActiveRunningModel, formatCtx, placementLabel } from "../lib/hardware";

const PROVIDER_LABELS: Record<string, string> = {
  ollama: "Ollama",
  gemini: "Gemini",
  "openai-compat": "OpenAI",
  custom: "Custom",
};

export default function OverviewPage() {
  const status = useConnection((s) => s.status);
  const offline = useConnection((s) => s.offline);
  const refresh = useConnection((s) => s.refresh);
  const provider = useWorkspace((s) => s.provider);

  const [pinging, setPinging] = useState(false);
  const [pingResult, setPingResult] = useState<PingResult | null>(null);

  const ping = async () => {
    setPinging(true);
    setPingResult(null);
    try {
      const r = await api.pingProvider({
        kind: provider.kind,
        base_url: provider.base_url,
        model: provider.model,
        api_key: provider.api_key,
        endpoint_path: provider.endpoint_path,
        allow_no_auth: provider.allow_no_auth,
      });
      setPingResult(r);
    } catch (e: any) {
      setPingResult({ ok: false, error: String(e?.message || e) });
    } finally {
      setPinging(false);
    }
  };

  const isLocal = provider.kind === "ollama";
  const ollamaOK = !offline && !!status?.ollama?.ok;
  const running = status?.ollama?.running_models || [];
  const activeRunning = isLocal && provider.model ? findActiveRunningModel(running, provider.model) : undefined;
  const placement = activeRunning ? placementLabel(activeRunning) : null;
  const embedPill = embedPillState(status?.hardware);

  const hardwareValue = status?.hardware?.ram_gb != null ? `${status.hardware.ram_gb.toFixed(1)} GB RAM` : "--";
  const hardwareCaption =
    status?.hardware?.gpu_vram_gb != null
      ? `${status.hardware.gpu_vram_gb.toFixed(1)} GB VRAM detected`
      : "No GPU detected";

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-6xl">
      <CardHeader
        title="Overview"
        description="At-a-glance system health, pulled from /api/status."
        right={
          <button onClick={refresh} className="av-btn-ghost">
            <RefreshCw className="h-3 w-3" /> Refresh
          </button>
        }
      />

      <div className="grid grid-cols-2 gap-4">
        <Card padded>
          <CardHeader
            eyebrow="Provider Connection"
            title={PROVIDER_LABELS[provider.kind] || provider.kind}
            description="Active model provider and reachability."
            right={
              <button onClick={ping} disabled={pinging} className="av-btn-ghost">
                {pinging ? <RefreshCw className="h-3 w-3 animate-spin" /> : <Wifi className="h-3 w-3" />}
                {pinging ? "Pinging…" : "Ping"}
              </button>
            }
          />
          <div className="space-y-2 text-xs font-mono">
            <div className="bg-slate-950 p-2.5 rounded border border-white/5 flex justify-between items-center gap-3">
              <span className="text-slate-500">model</span>
              <span className="text-slate-200 truncate">{provider.model || "--"}</span>
            </div>
            <div className="bg-slate-950 p-2.5 rounded border border-white/5 flex justify-between items-center gap-3">
              <span className="text-slate-500">base_url</span>
              <span className="text-slate-200 truncate">{provider.base_url || "--"}</span>
            </div>
            {activeRunning && (
              <div className="bg-slate-950 p-2.5 rounded border border-white/5 flex justify-between items-center gap-3">
                <span className="text-slate-500">loaded</span>
                <span className={placement?.tone || "text-slate-200"}>
                  ctx {formatCtx(activeRunning.context_length)} · {placement?.label}
                </span>
              </div>
            )}
            {pingResult && (
              <p className={pingResult.ok ? "text-emerald-400" : "text-red-400"}>
                {pingResult.ok
                  ? `Reachable · ${pingResult.latency_ms ?? "?"}ms`
                  : pingResult.error || "Unreachable"}
              </p>
            )}
          </div>
          <p className="text-[10px] text-slate-500 mt-3">
            <Link to="/settings" className="text-emerald-400 hover:underline">
              Manage profiles in Settings →
            </Link>
          </p>
        </Card>

        <Card padded>
          <CardHeader eyebrow="Snapshot" title="Handshake" description="Latest /api/status payload." />
          <div className="grid grid-cols-2 gap-3">
            <MiniStat
              label="Status"
              value={
                <Pill tone={ollamaOK ? "neon" : "red"}>
                  <StatusDot tone={ollamaOK ? "neon" : "red"} /> {ollamaOK ? "Online" : "Offline"}
                </Pill>
              }
            />
            <MiniStat label="Profiles" value={status?.profile_count ?? "--"} />
            <MiniStat label="Sessions" value={status?.session_count ?? "--"} />
            <MiniStat label="Telemetry" value={status?.telemetry_enabled ? "On" : "Off"} />
          </div>
          {status?.default_model && (
            <p className="text-[10px] text-slate-500 mt-3 bg-slate-950 border border-white/5 rounded p-2">
              Recommended default: <span className="text-slate-300">{status.default_model}</span>
            </p>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <StatCard
          label="Hardware"
          value={hardwareValue}
          caption={hardwareCaption}
          tone={status?.hardware?.ram_gb != null ? "neon" : "slate"}
        />
        <StatCard
          label="Embedding Device"
          value={embedPill?.label.replace("Embed: ", "") || "--"}
          caption="From the local torch/CUDA probe."
          tone={embedPill?.tone || "slate"}
        />
        <StatCard
          label="Loaded Model"
          value={activeRunning ? formatCtx(activeRunning.context_length) + " ctx" : "idle"}
          caption="From Ollama's /api/ps."
          tone={activeRunning ? "neon" : "slate"}
        />
      </div>

      {offline && (
        <Card padded className="border-red-500/40">
          <p className="text-xs text-red-300 font-mono flex items-center gap-2">
            <WifiOff className="h-3 w-3" /> Backend unreachable — retrying…
          </p>
        </Card>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="bg-slate-950 p-2.5 rounded border border-white/5">
      <p className="av-section-eyebrow mb-1 text-[9px]">{label}</p>
      <div className="text-sm font-mono text-slate-200">{value}</div>
    </div>
  );
}
