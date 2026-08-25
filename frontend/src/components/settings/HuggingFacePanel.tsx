import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle,
  Cpu,
  Download,
  Gauge,
  Heart,
  Loader2,
  RefreshCw,
  Search,
  X,
  XCircle,
} from "lucide-react";
import { api, type HfHubModel, type HfModelFit } from "../../lib/api";
import { cancelPull, startPull, usePullState } from "../../lib/pullManager";
import { Card, CardHeader } from "../Card";
import { Pill } from "../Pill";
import { Select, TextInput } from "../Input";
import { PullProgress } from "./PullProgress";

// GGUF models are pulled through the local Ollama daemon, so downloads always
// target the default Ollama host regardless of the active cloud provider.
const OLLAMA_BASE_URL = "http://localhost:11434";

const SORTS = [
  { value: "trending_score", label: "Trending" },
  { value: "downloads", label: "Most downloaded" },
  { value: "likes", label: "Most liked" },
  { value: "lastModified", label: "Recently updated" },
];

function compact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

// Derive a short, human-friendly Ollama tag from a Hub repo id so the pulled
// model shows up as e.g. ``ornith-1.0-9b-gguf`` instead of the full
// ``hf.co/ornith-ai/Ornith-1.0-9B-GGUF`` web address. The quant, when chosen,
// becomes the tag (``…:q4_k_m``).
function localName(repoId: string, quant?: string): string {
  const base = (repoId.split("/").pop() || repoId)
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "");
  const tag = quant ? quant.toLowerCase().replace(/[^a-z0-9._-]+/g, "-") : "";
  return tag ? `${base}:${tag}` : base;
}

export function HuggingFacePanel() {
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("trending_score");
  const [models, setModels] = useState<HfHubModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [quant, setQuant] = useState<Record<string, string>>({});
  // Per-repo "Check fit" state: the resolved spec + verdict, an in-flight
  // flag, and any error, all keyed by the Hub repo id.
  const [fit, setFit] = useState<Record<string, HfModelFit>>({});
  const [fitLoading, setFitLoading] = useState<Record<string, boolean>>({});
  const [fitError, setFitError] = useState<Record<string, string>>({});
  const activePull = usePullState();

  const checkFit = (repo: string) => {
    setFitLoading((s) => ({ ...s, [repo]: true }));
    setFitError((s) => ({ ...s, [repo]: "" }));
    api
      .hfFit(repo)
      .then((r) => setFit((s) => ({ ...s, [repo]: r })))
      .catch((e: any) => setFitError((s) => ({ ...s, [repo]: String(e?.message || e) })))
      .finally(() => setFitLoading((s) => ({ ...s, [repo]: false })));
  };

  const load = () => {
    setLoading(true);
    setError(null);
    api
      .hfModels({ search: search.trim(), sort, limit: 40 })
      .then((r) => setModels(r.models))
      .catch((e: any) => setError(String(e?.message || e)))
      .finally(() => setLoading(false));
  };

  // Reload on sort change and on first mount; search is applied via the form.
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sort]);

  const pull = (m: HfHubModel) => {
    const q = quant[m.id];
    const tag = q ? `${m.pull_tag}:${q}` : m.pull_tag;
    // Re-alias to a clean local name so the model isn't stored under the full
    // hf.co/<repo> web address.
    startPull(tag, OLLAMA_BASE_URL, undefined, localName(m.id, q));
  };

  return (
    <Card padded>
      <CardHeader
        eyebrow="Hugging Face Hub"
        title="Browse GGUF models"
        description="Pull any GGUF repository straight into Ollama via hf.co/<repo>. Use Check fit to size a model against your hardware before pulling."
        right={
          <button className="av-btn-ghost" onClick={load} disabled={loading}>
            {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
            Refresh
          </button>
        }
      />

      <form
        className="flex gap-2 mb-4"
        onSubmit={(e) => {
          e.preventDefault();
          load();
        }}
      >
        <div className="relative flex-1">
          <Search className="h-3 w-3 text-slate-600 absolute left-2.5 top-1/2 -translate-y-1/2" />
          <TextInput
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search the Hub (e.g. llama, qwen, gemma)…"
            className="pl-7"
          />
        </div>
        <Select value={sort} onChange={(e) => setSort((e.target as any).value)} className="w-44">
          {SORTS.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </Select>
      </form>

      {error && <p className="text-xs text-red-400 font-mono mb-3">{error}</p>}

      <div className="space-y-2">
        {models.map((m) => {
          const isPulling =
            activePull && activePull.model.startsWith(m.pull_tag) && !activePull.done && !activePull.error;
          return (
            <div key={m.id} className="av-card p-3 flex flex-col gap-2">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <a
                    href={`https://huggingface.co/${m.id}`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-sm font-medium text-white hover:text-emerald-400 truncate block"
                  >
                    {m.id}
                  </a>
                  <div className="flex items-center gap-3 mt-1 text-[10px] font-mono text-slate-400">
                    <span className="flex items-center gap-1">
                      <Download className="h-3 w-3" /> {compact(m.downloads)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Heart className="h-3 w-3" /> {compact(m.likes)}
                    </span>
                    {m.gated && <Pill tone="slate">gated</Pill>}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {m.quants.length > 0 && (
                    <Select
                      value={quant[m.id] ?? ""}
                      onChange={(e) => setQuant((q) => ({ ...q, [m.id]: (e.target as any).value }))}
                      className="w-28 text-xs"
                    >
                      <option value="">default quant</option>
                      {m.quants.map((q) => (
                        <option key={q} value={q}>
                          {q}
                        </option>
                      ))}
                    </Select>
                  )}
                  <button
                    className="av-btn-ghost whitespace-nowrap"
                    onClick={() => checkFit(m.id)}
                    disabled={fitLoading[m.id]}
                    title="Estimate this model's requirements and check it against your hardware"
                  >
                    {fitLoading[m.id] ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Gauge className="h-3 w-3" />
                    )}
                    Check fit
                  </button>
                  {isPulling ? (
                    <button className="av-btn-ghost whitespace-nowrap" onClick={cancelPull}>
                      <X className="h-3 w-3" /> Cancel
                    </button>
                  ) : (
                    <button className="av-btn-primary whitespace-nowrap" onClick={() => pull(m)}>
                      <Download className="h-3 w-3" /> Pull
                    </button>
                  )}
                </div>
              </div>
              {fitError[m.id] && (
                <p className="text-xs text-red-400 font-mono">{fitError[m.id]}</p>
              )}
              {fit[m.id] && <FitResult fit={fit[m.id]} />}
              {activePull && activePull.model.startsWith(m.pull_tag) && (
                <PullProgress pull={activePull} model={activePull.model} />
              )}
            </div>
          );
        })}
        {!loading && models.length === 0 && !error && (
          <p className="text-xs text-slate-500 font-mono">No models found.</p>
        )}
      </div>
    </Card>
  );
}

function fitTone(fit: string): { color: string; Icon: typeof CheckCircle } {
  const f = fit.toLowerCase();
  if (f.includes("fits")) return { color: "text-emerald-400", Icon: CheckCircle };
  if (f.includes("tight")) return { color: "text-amber-400", Icon: AlertTriangle };
  if (f.includes("unknown")) return { color: "text-slate-400", Icon: Cpu };
  return { color: "text-red-400", Icon: XCircle };
}

// Inline spec + hardware verdict rendered under a model card after Check fit.
function FitResult({ fit }: { fit: HfModelFit }) {
  const { color, Icon } = fitTone(fit.fit);
  const ram = fit.hardware?.ram_gb;
  const vram = fit.hardware?.gpu_vram_gb;
  const paramsLabel =
    fit.params_b > 0
      ? `${fit.params_b.toFixed(1)}B${fit.params_source === "name" ? " (est.)" : ""}`
      : "unknown";
  return (
    <div className="bg-slate-950 rounded-lg border border-white/5 p-3 space-y-2">
      <div className={`flex items-center gap-1.5 text-xs font-medium ${color}`}>
        <Icon className="h-3.5 w-3.5 shrink-0" />
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
