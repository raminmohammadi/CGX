import { Cpu } from "lucide-react";
import { useConnection } from "../store/connection";
import { useWorkspace } from "../store/workspace";
import { StatusDot } from "../components/Pill";
import type { HardwareInfo, RunningModel } from "../lib/api";

// Top platform header: brand + provider health pulse + mode badge.
// Reads from the shared connection store and active workspace provider.

const PROVIDER_LABELS: Record<string, string> = {
  ollama: "Ollama",
  gemini: "Gemini",
  "openai-compat": "OpenAI",
  custom: "Custom",
};

// Render a token count as a short human label: 4096 → "4K", 131072 → "131K".
function formatCtx(n: number | null | undefined): string {
  if (!n || n <= 0) return "—";
  if (n >= 1000) return `${Math.round(n / 1024)}K`;
  return String(n);
}

// Classify Ollama placement from the two byte counts in /api/ps.
function placementLabel(m: RunningModel): { label: string; tone: string } {
  const total = Number(m.size || 0);
  const vram = Number(m.size_vram || 0);
  if (total > 0 && vram >= total) return { label: "GPU", tone: "text-emerald-300" };
  if (vram === 0) return { label: "CPU", tone: "text-amber-300" };
  const pct = total > 0 ? Math.round((vram / total) * 100) : 0;
  return { label: `GPU ${pct}%`, tone: "text-amber-300" };
}

// Decide the Embed (torch/CUDA) pill state from the hardware probe.
//   - returns null when torch isn't installed (core-only install -- nothing
//     to surface; the user isn't running local embeddings)
//   - "warn" when nvidia-smi sees a GPU but torch can't use it (the regression
//     this whole check exists to catch)
//   - "gpu" / "cpu" for the healthy cases
function embedPillState(hw: HardwareInfo | undefined): {
  tone: "neon" | "red" | "amber" | "slate";
  label: string;
  title: string;
} | null {
  if (!hw || hw.torch_installed !== true) return null;
  if (hw.torch_cuda_warning) {
    return {
      tone: "red",
      label: "Embed: CPU ⚠",
      title: hw.torch_cuda_warning,
    };
  }
  if (hw.torch_cuda_available) {
    const buildTag = hw.torch_cuda_build ? ` (CUDA ${hw.torch_cuda_build})` : "";
    return {
      tone: "neon",
      label: "Embed: GPU",
      title: `torch ${hw.torch_version || "?"}${buildTag} -- embeddings run on the GPU`,
    };
  }
  return {
    tone: "slate",
    label: "Embed: CPU",
    title: hw.gpu_vram_gb
      ? "torch is CPU-only despite a GPU being present"
      : "No NVIDIA GPU detected; embeddings run on CPU",
  };
}

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
    : "--";

  // Match the active model against Ollama's currently-loaded set so the user
  // can see at a glance whether their picked model is actually resident, with
  // what effective context window and GPU/CPU placement.
  const running = (status?.ollama?.running_models || []) as RunningModel[];
  const activeRunning =
    isLocal && provider.model
      ? running.find(
          (m) =>
            (m.name || m.model || "").toLowerCase() ===
            provider.model.toLowerCase(),
        )
      : undefined;
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

  return (
    <header
      className="h-14 border-b flex items-center justify-between px-6 bg-header flex-shrink-0"
      style={{ borderColor: "rgba(255,255,255,0.06)" }}
    >
      <div className="flex items-center space-x-3">
        <div className="h-7 w-7 bg-emerald-500 rounded flex items-center justify-center text-slate-950 font-bold text-sm shadow-md shadow-emerald-500/10">
          C
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="font-bold tracking-tight text-white text-base font-mono">
            CGX
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
      </div>
    </header>
  );
}
