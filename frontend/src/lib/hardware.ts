import type { HardwareInfo, RunningModel } from "./api";

// Render a token count as a short human label: 4096 → "4K", 131072 → "131K".
export function formatCtx(n: number | null | undefined): string {
  if (!n || n <= 0) return "—";
  if (n >= 1000) return `${Math.round(n / 1024)}K`;
  return String(n);
}

// Classify Ollama placement from the two byte counts in /api/ps.
export function placementLabel(m: RunningModel): { label: string; tone: string } {
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
export function embedPillState(hw: HardwareInfo | undefined): {
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

// Find the currently-resident Ollama model matching the active provider's
// configured model name, if any (case-insensitive match on name/model field).
export function findActiveRunningModel(
  running: RunningModel[],
  modelName: string | null | undefined,
): RunningModel | undefined {
  if (!modelName) return undefined;
  const needle = modelName.toLowerCase();
  return running.find((m) => (m.name || m.model || "").toLowerCase() === needle);
}
