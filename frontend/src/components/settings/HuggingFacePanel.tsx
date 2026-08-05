import { useEffect, useState } from "react";
import { Download, Heart, Loader2, RefreshCw, Search, X } from "lucide-react";
import { api, type HfHubModel } from "../../lib/api";
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

export function HuggingFacePanel() {
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("trending_score");
  const [models, setModels] = useState<HfHubModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [quant, setQuant] = useState<Record<string, string>>({});
  const activePull = usePullState();

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
    startPull(tag, OLLAMA_BASE_URL);
  };

  return (
    <Card padded>
      <CardHeader
        eyebrow="Hugging Face Hub"
        title="Browse GGUF models"
        description="Pull any GGUF repository straight into Ollama via hf.co/<repo>."
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
