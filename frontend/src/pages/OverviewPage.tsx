import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router";
import {
  AlertTriangle, CheckCircle2, Download, Loader2, Power,
  RefreshCw, Wifi, WifiOff, X,
} from "lucide-react";
import { useConnection } from "../store/connection";
import { useWorkspace } from "../store/workspace";
import { Card, CardHeader } from "../components/Card";
import { Pill, StatusDot } from "../components/Pill";
import { StatCard } from "../components/StatCard";
import { Select } from "../components/Input";
import { PullProgress } from "../components/settings/PullProgress";
import { modelInstalled, ollamaPullPlan } from "../components/settings/providerKinds";
import { cancelPull, startPull, usePullState } from "../lib/pullManager";
import { api, type PingResult, type ProfileSummary } from "../lib/api";
import { embedPillState, findActiveRunningModel, formatCtx, placementLabel } from "../lib/hardware";

const PROVIDER_LABELS: Record<string, string> = {
  ollama: "Ollama",
  gemini: "Gemini",
  "openai-compat": "OpenAI",
  huggingface: "Hugging Face",
  custom: "Custom",
};

// The single source of truth for "is the selected model usable right now?".
// It deliberately separates the daemon being reachable from the model being
// installed from the model being loaded, because the old single "Online" pill
// conflated all three and could read green while generation 404'd.
type ConnState =
  | "backend_down"   // CGX backend itself unreachable
  | "probing"        // still fetching the provider probe
  | "daemon_down"    // Ollama not answering at the active base_url
  | "model_missing"  // daemon up, but the selected tag isn't pulled
  | "ready"          // installed, will load on first request
  | "connected"      // installed AND currently resident in memory
  | "cloud_unknown"  // cloud provider, not yet tested
  | "cloud_ok"       // cloud provider, ping succeeded
  | "cloud_error";   // cloud provider, ping failed

const STATE_PILL: Record<ConnState, { tone: "neon" | "red" | "amber" | "slate"; label: string }> = {
  backend_down: { tone: "red", label: "Backend offline" },
  probing: { tone: "slate", label: "Checking…" },
  daemon_down: { tone: "red", label: "Ollama offline" },
  model_missing: { tone: "amber", label: "Model not installed" },
  ready: { tone: "neon", label: "Ready" },
  connected: { tone: "neon", label: "Connected" },
  cloud_unknown: { tone: "slate", label: "Untested" },
  cloud_ok: { tone: "neon", label: "Connected" },
  cloud_error: { tone: "red", label: "Unreachable" },
};

export default function OverviewPage() {
  const status = useConnection((s) => s.status);
  const offline = useConnection((s) => s.offline);
  const refresh = useConnection((s) => s.refresh);
  const provider = useWorkspace((s) => s.provider);
  const setProvider = useWorkspace((s) => s.setProvider);
  const applyProfile = useWorkspace((s) => s.applyProfile);

  const [pinging, setPinging] = useState(false);
  const [pingResult, setPingResult] = useState<PingResult | null>(null);
  const [probe, setProbe] = useState<{ reachable: boolean; installed: string[] } | null>(null);
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const activePull = usePullState();

  const isLocal = provider.kind === "ollama";

  // Probe the *active provider's* base_url (not the hardcoded localhost the
  // /api/status pill uses), so the indicator is correct for a remote Ollama.
  const refreshProbe = () => {
    if (!isLocal) {
      setProbe(null);
      return;
    }
    api
      .setupModels(provider.base_url)
      .then((r) => setProbe({ reachable: r.ollama_reachable ?? false, installed: r.installed || [] }))
      .catch(() => setProbe({ reachable: false, installed: [] }));
  };

  useEffect(() => {
    refreshProbe();
    setPingResult(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider.kind, provider.base_url]);

  useEffect(() => {
    api.listProfiles().then(setProfiles).catch(() => {});
  }, []);

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

  const running = status?.ollama?.running_models || [];
  const activeRunning = isLocal && provider.model ? findActiveRunningModel(running, provider.model) : undefined;
  const placement = activeRunning ? placementLabel(activeRunning) : null;
  const installedHas = probe ? modelInstalled(probe.installed, provider.model) : null;

  const connState: ConnState = useMemo(() => {
    if (offline) return "backend_down";
    if (isLocal) {
      if (probe === null) return "probing";
      if (!probe.reachable) return "daemon_down";
      if (!installedHas) return "model_missing";
      if (activeRunning) return "connected";
      return "ready";
    }
    if (pingResult === null) return "cloud_unknown";
    return pingResult.ok ? "cloud_ok" : "cloud_error";
  }, [offline, isLocal, probe, installedHas, activeRunning, pingResult]);

  const pill = STATE_PILL[connState];
  const pullPlan = ollamaPullPlan(provider.model);
  const isLocalhost = /localhost|127\.0\.0\.1/.test(provider.base_url);
  const defaultModel = status?.default_model;
  const activeProfile = provider.use_profile && provider.profile_name ? provider.profile_name : null;

  const doPull = () => {
    startPull(
      pullPlan.target,
      provider.base_url,
      () => {
        refreshProbe();
        // A re-aliased hf.co pull lands under a lowercased local tag; adopt it
        // so the exact-match generation path resolves the model.
        if (pullPlan.localName) setProvider({ model: pullPlan.localName, use_profile: false });
      },
      pullPlan.localName,
    );
  };

  const embedPill = embedPillState(status?.hardware);
  const hw = status?.hardware;
  const hardwareValue = hw?.ram_gb != null ? `${hw.ram_gb.toFixed(1)} GB RAM` : "--";
  const hardwareCaption = hw?.gpu_name
    ? `${hw.gpu_name}${hw.is_unified_memory ? ` · ${hw.gpu_vram_gb?.toFixed(1)} GB Unified` : hw.gpu_vram_gb != null ? ` · ${hw.gpu_vram_gb.toFixed(1)} GB VRAM` : ""}`
    : hw?.gpu_vram_gb != null
      ? `${hw.gpu_vram_gb.toFixed(1)} GB VRAM detected`
      : "No GPU detected";

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-6xl">
      <CardHeader
        title="Overview"
        description="At-a-glance system health and model connection."
        right={
          <button onClick={() => { refresh(); refreshProbe(); }} className="av-btn-ghost">
            <RefreshCw className="h-3 w-3" /> Refresh
          </button>
        }
      />

      {/* ── Connection panel: the one place to get a model wired up ── */}
      <Card padded>
        <CardHeader
          eyebrow="Model Connection"
          title={activeProfile ? `Profile: ${activeProfile}` : PROVIDER_LABELS[provider.kind] || provider.kind}
          description="Where the agent sends every request. Get this green before running Ask, Plan, or Agent."
          right={
            <div className="flex items-center gap-2">
              <button onClick={ping} disabled={pinging} className="av-btn-ghost" title="Send a tiny request to confirm the model responds (loads it into memory)">
                {pinging ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wifi className="h-3 w-3" />}
                {pinging ? "Testing…" : "Test"}
              </button>
              <Pill tone={pill.tone}>
                <StatusDot tone={pill.tone === "amber" ? "red" : pill.tone} /> {pill.label}
              </Pill>
            </div>
          }
        />

        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <KV label="model" value={provider.model || "--"} />
          <KV label="base_url" value={provider.base_url || "--"} />
          {activeRunning && (
            <KV
              label="loaded"
              value={`ctx ${formatCtx(activeRunning.context_length)} · ${placement?.label}`}
              tone={placement?.tone}
            />
          )}
          {isLocal && probe && (
            <KV label="installed models" value={String(probe.installed.length)} />
          )}
        </div>

        {/* ── State-specific guidance + call to action ── */}
        <div className="mt-4">
          <ConnectionAction
            state={connState}
            model={provider.model}
            baseUrl={provider.base_url}
            isLocalhost={isLocalhost}
            latencyMs={pingResult?.ok ? pingResult.latency_ms ?? null : null}
            errorText={connState === "cloud_error" ? pingResult?.error ?? null : null}
            installed={probe?.installed || []}
            defaultModel={defaultModel}
            activePull={activePull}
            pullTarget={pullPlan.target}
            onPull={doPull}
            onCancelPull={cancelPull}
            onPickModel={(m) => setProvider({ model: m, use_profile: false })}
            onUseDefault={defaultModel ? () => setProvider({ model: defaultModel, kind: "ollama", use_profile: false }) : undefined}
            onRetry={() => { refresh(); refreshProbe(); }}
          />
        </div>

        {/* ── Profile switcher: adopt any saved connection preset in one click ── */}
        {profiles.length > 0 && (
          <div className="mt-4 pt-3 border-t border-muted flex items-center gap-2 flex-wrap">
            <span className="av-section-eyebrow">Profiles</span>
            {profiles.map((p) => (
              <button
                key={p.name}
                onClick={() => { applyProfile(p); }}
                className={
                  "text-[11px] font-mono px-2 py-1 rounded border transition " +
                  (activeProfile === p.name
                    ? "border-emerald-500/40 bg-emerald-950/30 text-emerald-300"
                    : "border-white/10 bg-slate-950/40 text-slate-300 hover:border-white/20")
                }
                title={`${p.model} @ ${p.base_url}`}
              >
                {p.name}
              </button>
            ))}
            <Link to="/settings" className="text-[10px] text-emerald-400 hover:underline ml-auto">
              Manage profiles →
            </Link>
          </div>
        )}
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card padded>
          <CardHeader eyebrow="Snapshot" title="Handshake" description="Latest /api/status payload." />
          <div className="grid grid-cols-2 gap-3">
            <MiniStat
              label="Daemon"
              value={
                <Pill tone={!offline && !!status?.ollama?.ok ? "neon" : "red"}>
                  <StatusDot tone={!offline && !!status?.ollama?.ok ? "neon" : "red"} />
                  {!offline && !!status?.ollama?.ok ? "Reachable" : "Down"}
                </Pill>
              }
            />
            <MiniStat label="Profiles" value={status?.profile_count ?? "--"} />
            <MiniStat label="Sessions" value={status?.session_count ?? "--"} />
            <MiniStat label="Telemetry" value={status?.telemetry_enabled ? "On" : "Off"} />
          </div>
          {defaultModel && (
            <p className="text-[10px] text-slate-500 mt-3 bg-slate-950 border border-white/5 rounded p-2">
              Recommended for this hardware: <span className="text-slate-300">{defaultModel}</span>
            </p>
          )}
        </Card>

        <Card padded>
          <CardHeader eyebrow="Local inference" title="Devices" description="From the torch/CUDA + Ollama probes." />
          <div className="grid grid-cols-1 gap-3">
            <MiniStat
              label="Embedding device"
              value={embedPill?.label.replace("Embed: ", "") || "--"}
            />
            <MiniStat
              label="Loaded model"
              value={activeRunning ? `${formatCtx(activeRunning.context_length)} ctx · ${placement?.label}` : "idle"}
            />
          </div>
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

// The actionable body of the connection card: one clear next step per state.
function ConnectionAction({
  state, model, baseUrl, isLocalhost, latencyMs, errorText, installed,
  defaultModel, activePull, pullTarget, onPull, onCancelPull, onPickModel,
  onUseDefault, onRetry,
}: {
  state: ConnState;
  model: string;
  baseUrl: string;
  isLocalhost: boolean;
  latencyMs: number | null;
  errorText: string | null;
  installed: string[];
  defaultModel?: string;
  activePull: ReturnType<typeof usePullState>;
  pullTarget: string;
  onPull: () => void;
  onCancelPull: () => void;
  onPickModel: (m: string) => void;
  onUseDefault?: () => void;
  onRetry: () => void;
}) {
  const pulling = !!activePull && !activePull.done && !activePull.error;

  if (state === "connected") {
    return (
      <Banner tone="ok" icon={<CheckCircle2 className="h-4 w-4" />} title="Connected and loaded">
        <span className="text-white">{model}</span> is resident in memory and ready to serve requests.
      </Banner>
    );
  }

  if (state === "ready") {
    return (
      <Banner tone="ok" icon={<CheckCircle2 className="h-4 w-4" />} title="Ready">
        <span className="text-white">{model}</span> is installed. It loads into memory on the first request — or press
        {" "}<strong className="text-slate-200">Test</strong> to warm it up now.
      </Banner>
    );
  }

  if (state === "cloud_ok") {
    return (
      <Banner tone="ok" icon={<CheckCircle2 className="h-4 w-4" />} title="Connected">
        Reachable{latencyMs != null ? ` · ${Math.round(latencyMs)}ms` : ""}. Provider is responding.
      </Banner>
    );
  }

  if (state === "cloud_unknown") {
    return (
      <Banner tone="info" icon={<Wifi className="h-4 w-4" />} title="Not tested yet">
        Press <strong className="text-slate-200">Test</strong> to confirm the cloud provider and model are reachable.
      </Banner>
    );
  }

  if (state === "cloud_error") {
    return (
      <Banner tone="err" icon={<AlertTriangle className="h-4 w-4" />} title="Provider unreachable">
        <span className="break-all">{errorText || "The provider did not respond."}</span>
        <div className="mt-2">
          <Link to="/settings" className="av-btn-ghost inline-flex">Fix in Settings</Link>
        </div>
      </Banner>
    );
  }

  if (state === "probing") {
    return (
      <Banner tone="info" icon={<Loader2 className="h-4 w-4 animate-spin" />} title="Checking connection…">
        Probing Ollama at <span className="text-slate-300">{baseUrl}</span>.
      </Banner>
    );
  }

  if (state === "backend_down") {
    return (
      <Banner tone="err" icon={<WifiOff className="h-4 w-4" />} title="CGX backend unreachable">
        The CGX server isn't responding. It should auto-retry; use Refresh to check again.
        <div className="mt-2">
          <button onClick={onRetry} className="av-btn-ghost"><RefreshCw className="h-3 w-3" /> Retry</button>
        </div>
      </Banner>
    );
  }

  if (state === "daemon_down") {
    return (
      <Banner tone="err" icon={<Power className="h-4 w-4" />} title="Ollama isn't running">
        Nothing is answering at <span className="text-slate-300">{baseUrl}</span>.
        {isLocalhost ? (
          <>
            {" "}Start the local daemon, then retry:
            <pre className="mt-2 bg-slate-950 border border-white/10 rounded px-3 py-2 text-[11px] text-emerald-300 overflow-x-auto">ollama serve</pre>
          </>
        ) : (
          <> {" "}Check that Ollama is running and reachable at that host.</>
        )}
        <div className="mt-2">
          <button onClick={onRetry} className="av-btn-ghost"><RefreshCw className="h-3 w-3" /> Retry</button>
        </div>
      </Banner>
    );
  }

  // model_missing
  return (
    <Banner tone="warn" icon={<AlertTriangle className="h-4 w-4" />} title="Model not installed">
      <span className="text-white">{model}</span> isn't pulled into Ollama at{" "}
      <span className="text-slate-300">{baseUrl}</span>.
      <div className="mt-3 flex items-center gap-2 flex-wrap">
        {pulling ? (
          <button onClick={onCancelPull} className="av-btn-ghost"><X className="h-3 w-3" /> Cancel pull</button>
        ) : (
          <button onClick={onPull} className="av-btn-primary">
            <Download className="h-3 w-3" /> Pull {pullTarget}
          </button>
        )}
        {onUseDefault && defaultModel && defaultModel !== model && (
          <button onClick={onUseDefault} className="av-btn-ghost" title="Switch to the model recommended for your hardware">
            Use recommended: {defaultModel}
          </button>
        )}
        {installed.length > 0 && (
          <Select
            value=""
            onChange={(e) => {
              const v = (e.target as HTMLSelectElement).value;
              if (v) onPickModel(v);
            }}
            className="w-52"
          >
            <option value="">Pick an installed model…</option>
            {installed.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </Select>
        )}
      </div>
      <PullProgress pull={activePull} model={activePull?.model ?? pullTarget} />
    </Banner>
  );
}

function Banner({
  tone, icon, title, children,
}: {
  tone: "ok" | "warn" | "err" | "info";
  icon: ReactNode;
  title: string;
  children: ReactNode;
}) {
  const toneCls = {
    ok: "border-emerald-500/25 bg-emerald-500/5 text-emerald-300",
    warn: "border-amber-500/25 bg-amber-500/5 text-amber-300",
    err: "border-red-500/25 bg-red-500/5 text-red-300",
    info: "border-white/10 bg-slate-950/40 text-slate-300",
  }[tone];
  return (
    <div className={`flex items-start gap-3 rounded-lg border px-4 py-3 text-xs ${toneCls}`}>
      <span className="shrink-0 mt-0.5">{icon}</span>
      <div className="min-w-0">
        <p className="font-semibold">{title}</p>
        <div className="text-slate-400 mt-0.5 font-mono leading-relaxed">{children}</div>
      </div>
    </div>
  );
}

function KV({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="bg-slate-950 p-2.5 rounded border border-white/5 flex justify-between items-center gap-3">
      <span className="text-slate-500">{label}</span>
      <span className={`truncate ${tone || "text-slate-200"}`}>{value}</span>
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
