import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle,
  Cpu,
  Gauge,
  HardDrive,
  Loader2,
  Microchip,
  RefreshCcw,
  Search,
  XCircle,
} from "lucide-react";
import { api, type HardwareMatrixResponse, type HfModelFit } from "../../lib/api";
import { useWorkspace } from "../../store/workspace";
import { Card, CardHeader } from "../Card";
import { Pill } from "../Pill";
import { StatCard } from "../StatCard";
import { TextInput } from "../Input";

// GGUF / Ollama models live in the local daemon, so the installed-model probe
// targets the active Ollama host (or localhost when a cloud provider is set).
const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434";

export function HardwarePanel() {
  const provider = useWorkspace((s) => s.provider);
  const ollamaBaseUrl =
    provider.kind === "ollama" && provider.base_url ? provider.base_url : DEFAULT_OLLAMA_BASE_URL;

  const [data, setData] = useState<HardwareMatrixResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ad-hoc Hugging Face fit checker: paste a repo id, size it against the
  // detected hardware budget without pulling anything.
  const [hfRepo, setHfRepo] = useState("");
  const [hfFit, setHfFit] = useState<HfModelFit | null>(null);
  const [hfBusy, setHfBusy] = useState(false);
  const [hfError, setHfError] = useState<string | null>(null);

  const checkHfFit = async () => {
    const repo = hfRepo.trim();
    if (!repo) return;
    setHfBusy(true);
    setHfError(null);
    setHfFit(null);
    try {
      setHfFit(await api.hfFit(repo));
    } catch (e: any) {
      setHfError(String(e?.message || e));
    } finally {
      setHfBusy(false);
    }
  };

  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      const d = await api.hardwareMatrix(ollamaBaseUrl);
      setData(d);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ollamaBaseUrl]);

  const ram = data?.hardware?.ram_gb;
  const vram = data?.hardware?.gpu_vram_gb;
  const installedCount = data?.rows?.filter((r) => r.installed).length ?? 0;

  return (
    <div className="space-y-6">
      <CardHeader
        eyebrow="Hardware"
        title="Hardware-Aware Local Catalog"
        description="Cross-references localized system resources directly against 4-bit quantized GGUF inference thresholds. Models you've already pulled into Ollama are flagged as Downloaded."
        right={
          <button onClick={load} disabled={busy} className="av-btn-primary">
            <Microchip className="h-3 w-3" />
            {busy ? "Detecting…" : "Detect Hardware Budget"}
          </button>
        }
      />

      <div className="grid grid-cols-4 gap-3">
        <StatCard label="System RAM" value={ram != null ? `${ram.toFixed(1)} GB` : "--"} tone="neon" />
        <StatCard
          label={data?.hardware?.is_unified_memory ? "GPU Unified Memory" : "GPU VRAM"}
          value={vram != null ? `${vram.toFixed(1)} GB` : "--"}
          caption={data?.hardware?.gpu_name || undefined}
          tone={vram != null ? "neon" : "slate"}
        />
        <StatCard label="Catalog rows" value={data ? `${data.rows.length}` : "--"} tone="slate" />
        <StatCard
          label="Installed"
          value={data ? `${installedCount}` : "--"}
          tone={installedCount > 0 ? "neon" : "slate"}
        />
      </div>

      <div className="bg-surface rounded-xl border border-muted overflow-hidden">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-slate-950 text-slate-400 uppercase border-b border-white/5">
            <tr>
              <th className="p-3 text-[10px]">Model</th>
              <th className="p-3 text-[10px]">Params</th>
              <th className="p-3 text-[10px]">Min RAM</th>
              <th className="p-3 text-[10px]">Rec VRAM</th>
              <th className="p-3 text-[10px]">Family</th>
              <th className="p-3 text-[10px]">Status</th>
              <th className="p-3 text-[10px]">Fit</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5 text-slate-300">
            {data?.rows?.map((r) => (
              <tr key={r.model} title={r.reason} className={r.installed ? "bg-emerald-500/5" : undefined}>
                <td className="p-3 font-semibold text-white">{r.model}</td>
                <td className="p-3">{r.params_b > 0 ? `${r.params_b.toFixed(1)}B` : "—"}</td>
                <td className="p-3">{r.min_ram_gb > 0 ? `${r.min_ram_gb.toFixed(1)} GB` : "—"}</td>
                <td className="p-3">{r.rec_vram_gb > 0 ? `${r.rec_vram_gb.toFixed(1)} GB` : "—"}</td>
                <td className="p-3 text-slate-400">{r.family}</td>
                <td className="p-3">
                  {r.installed ? (
                    <Pill tone="neon">
                      <HardDrive className="h-3 w-3" /> Downloaded
                    </Pill>
                  ) : (
                    <span className="text-slate-600">—</span>
                  )}
                </td>
                <td className={`p-3 font-medium ${fitColor(r.fit)}`}>
                  <span className="flex items-center gap-1.5">
                    <FitIcon fit={r.fit} />
                    {r.fit}
                    {r.notes && <span className="text-slate-500 text-[10px]">({r.notes})</span>}
                  </span>
                </td>
              </tr>
            ))}
            {!data?.rows?.length && (
              <tr>
                <td colSpan={7} className="p-6 text-center text-slate-500">
                  {busy ? "Loading…" : error || "No catalog rows."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <Card padded>
        <CardHeader
          eyebrow="Hugging Face"
          title="Test a Hugging Face model"
          description="Paste any Hub repo id (e.g. Qwen/Qwen2.5-Coder-7B-Instruct) to estimate its requirements and check them against your hardware before pulling."
        />
        <form
          className="flex gap-2 mt-2"
          onSubmit={(e) => {
            e.preventDefault();
            checkHfFit();
          }}
        >
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-500" />
            <TextInput
              value={hfRepo}
              onChange={(e) => setHfRepo(e.target.value)}
              placeholder="owner/repo"
              className="w-full pl-8 font-mono text-xs"
            />
          </div>
          <button type="submit" className="av-btn-primary whitespace-nowrap" disabled={hfBusy || !hfRepo.trim()}>
            {hfBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Gauge className="h-3 w-3" />}
            Check fit
          </button>
        </form>
        {hfError && <p className="text-xs text-red-400 font-mono mt-2">{hfError}</p>}
        {hfFit && (
          <div className="mt-3">
            <FitResult fit={hfFit} />
          </div>
        )}
      </Card>

      <Card padded>
        <CardHeader title="Local-First vs Cloud Trade-offs" eyebrow="Matrix" />
        <div className="grid grid-cols-2 gap-3 text-xs">
          {data?.tradeoffs?.map((t) => (
            <div
              key={t.dimension}
              className="bg-slate-950 p-3 rounded-lg border border-white/5 flex justify-between items-center gap-3"
            >
              <div className="min-w-0">
                <p className="text-slate-200 font-medium truncate">{t.dimension}</p>
                <p className="text-[10px] text-slate-500 font-mono mt-0.5 truncate">Local: {t.local}</p>
                <p className="text-[10px] text-slate-500 font-mono truncate">Cloud: {t.cloud}</p>
              </div>
              <span
                className={`uppercase text-[10px] font-bold px-2 py-0.5 rounded border font-mono whitespace-nowrap ${winnerClasses(t.winner)}`}
              >
                {t.winner}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {error && (
        <Card padded className="border-red-500/40">
          <p className="text-xs text-red-300 font-mono flex items-center gap-2">
            <RefreshCcw className="h-3 w-3" /> {error}
          </p>
        </Card>
      )}
    </div>
  );
}

function fitColor(fit: string): string {
  const f = fit.toLowerCase();
  if (f.includes("fits")) return "text-emerald-400";
  if (f.includes("tight")) return "text-amber-400";
  return "text-red-400";
}

function FitIcon({ fit }: { fit: string }) {
  const f = fit.toLowerCase();
  if (f.includes("fits")) return <CheckCircle className="h-3.5 w-3.5 text-emerald-400 shrink-0" />;
  if (f.includes("tight")) return <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0" />;
  return <XCircle className="h-3.5 w-3.5 text-red-400 shrink-0" />;
}

// Inline spec + hardware verdict rendered under the HF fit checker.
function FitResult({ fit }: { fit: HfModelFit }) {
  const ram = fit.hardware?.ram_gb;
  const vram = fit.hardware?.gpu_vram_gb;
  const paramsLabel =
    fit.params_b > 0
      ? `${fit.params_b.toFixed(1)}B${fit.params_source === "name" ? " (est.)" : ""}`
      : "unknown";
  return (
    <div className="bg-slate-950 rounded-lg border border-white/5 p-3 space-y-2">
      <div className={`flex items-center gap-1.5 text-xs font-medium ${fitColor(fit.fit)}`}>
        {fit.fit.toLowerCase().includes("unknown") ? (
          <Cpu className="h-3.5 w-3.5 shrink-0" />
        ) : (
          <FitIcon fit={fit.fit} />
        )}
        <span className="uppercase font-mono">{fit.fit}</span>
        <span className="text-slate-500 font-mono text-[10px]">— {fit.reason}</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono">
        <Spec label="Params" value={paramsLabel} />
        <Spec label="Min RAM" value={fit.min_ram_gb > 0 ? `${fit.min_ram_gb.toFixed(1)} GB` : "—"} />
        <Spec label="Rec VRAM" value={fit.rec_vram_gb > 0 ? `${fit.rec_vram_gb.toFixed(1)} GB` : "—"} />
        <Spec
          label="Your budget"
          value={`${ram != null ? `${ram.toFixed(0)}G RAM` : "?"} / ${
            vram != null ? `${vram.toFixed(0)}G ${fit.hardware?.is_unified_memory ? "Unified" : "VRAM"}` : "no GPU"
          }`}
        />
      </div>
    </div>
  );
}

function Spec({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-900/60 px-2 py-1.5 rounded border border-white/5">
      <p className="text-slate-500">{label}</p>
      <p className="text-slate-200 truncate">{value}</p>
    </div>
  );
}

function winnerClasses(winner: string): string {
  const w = (winner || "").toLowerCase();
  if (w.includes("local")) return "text-emerald-400 bg-emerald-500/5 border-emerald-500/10";
  if (w.includes("cloud")) return "text-purple-400 bg-purple-500/5 border-purple-500/10";
  return "text-slate-400 bg-slate-500/5 border-white/5";
}
