import { useEffect, useRef, useState } from "react";
import { Database, HardDriveUpload, Play, RefreshCcw, Square } from "lucide-react";
import { api } from "../lib/api";
import { streamSSE } from "../lib/sse";
import { abortConnection, getConnection, setConnection } from "../lib/connections";
import { useTasks } from "../store/tasks";
import { useWorkspace } from "../store/workspace";
import { Card, CardHeader } from "../components/Card";
import { Field, Select, TextInput } from "../components/Input";
import { Pill } from "../components/Pill";

const PAGE_KEY = "index";

export default function IndexPage() {
  const { index, setIndex, projectRoot, setProjectRoot } = useWorkspace();
  const { index: indexState, setIndex: setIndexState, appendIndexProgress, resetIndex } = useTasks();
  const { busy, progress, result, error } = indexState;

  const [outDir, setOutDir] = useState("/tmp/cgx_index");
  const [embedModel, setEmbedModel] = useState(index.embed_model);
  const [metric, setMetric] = useState("cosine");
  const [indexType, setIndexType] = useState("flat");
  const [zipPath, setZipPath] = useState<string | null>(null);
  const [zipName, setZipName] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  // On mount: clear busy if no live connection.
  useEffect(() => {
    if (busy && !getConnection(PAGE_KEY)) {
      setIndexState({ busy: false });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onUpload = async (f: File) => {
    setIndexState({ error: null });
    try {
      const { path, original_name } = await api.uploadZip(f);
      setZipPath(path);
      setZipName(original_name);
    } catch (e: any) {
      setIndexState({ error: String(e?.message || e) });
    }
  };

  const run = (force: boolean = false) => {
    if (busy) return;
    if (!projectRoot && !zipPath) {
      setIndexState({ error: "Provide a project root or upload a .zip to index." });
      return;
    }
    void force;

    setIndexState({ error: null, progress: [], result: null, busy: true });

    abortConnection(PAGE_KEY);
    const conn = streamSSE(
      "/api/index/build",
      {
        project_root: projectRoot || null,
        out_dir: outDir,
        embed_model: embedModel,
        metric,
        index_type: indexType,
        zip_path: zipPath,
      },
      (ev, data) => {
        if (ev === "progress") {
          appendIndexProgress(data as any);
        } else if (ev === "result") {
          setIndexState({ result: data, busy: false });
          setIndex({
            index_dir: `${outDir.replace(/\/$/, "")}/indices`,
            records: `${outDir.replace(/\/$/, "")}/records.jsonl`,
            embed_model: embedModel,
          });
        } else if (ev === "cancelled") {
          setIndexState({ busy: false, error: "Cancelled." });
        } else if (ev === "error") {
          setIndexState({ error: String(data?.message || "build error"), busy: false });
        }
      },
      (err) => {
        setIndexState({ error: String((err as any)?.message || err), busy: false });
      },
    );

    setConnection(PAGE_KEY, conn);
    conn.done.finally(() => {
      setIndexState({ busy: false });
      abortConnection(PAGE_KEY);
    });
  };

  const cancel = () => {
    abortConnection(PAGE_KEY);
    setIndexState({ busy: false });
  };

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-4xl">
      <CardHeader
        title="Content-Addressed Embedding Cache"
        description="Tracks repository updates via SHA-256 validation mapping, processing only altered chunks."
        right={
          busy ? (
            <button onClick={cancel} className="av-btn-ghost">
              <Square className="h-3 w-3" /> Cancel
            </button>
          ) : null
        }
      />

      <Card padded>
        <div className="flex justify-between items-center border-b border-muted pb-4 mb-4">
          <div>
            <span className="av-section-eyebrow block">Target Directory Path</span>
            <span className="text-xs text-slate-200 font-medium font-mono">
              {projectRoot || zipName || "—"}
            </span>
          </div>
          <div className="flex gap-2">
            <button onClick={resetIndex} disabled={busy} className="av-btn-ghost">
              Clear
            </button>
            <button onClick={() => run(true)} disabled={busy} className="av-btn-ghost">
              <RefreshCcw className="h-3 w-3" /> Force Clean Rebuild
            </button>
            <button onClick={() => run(false)} disabled={busy} className="av-btn-primary">
              <Play className="h-3 w-3" /> {busy ? "Indexing…" : "Run Incremental Sync"}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="Project root">
            <TextInput
              value={projectRoot}
              onChange={(e) => setProjectRoot(e.target.value)}
              placeholder="/abs/path/to/repo"
            />
          </Field>
          <Field label="Or upload a .zip">
            <div className="flex gap-2 items-center">
              <input
                ref={fileRef}
                type="file"
                accept=".zip"
                hidden
                onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
              />
              <button className="av-btn-ghost" onClick={() => fileRef.current?.click()}>
                <HardDriveUpload className="h-3 w-3" /> Choose .zip
              </button>
              <span className="text-[10px] text-slate-500 font-mono truncate">
                {zipName || "no file chosen"}
              </span>
            </div>
          </Field>
          <Field label="Output dir">
            <TextInput value={outDir} onChange={(e) => setOutDir(e.target.value)} />
          </Field>
          <Field label="Embed model">
            <TextInput value={embedModel} onChange={(e) => setEmbedModel(e.target.value)} />
          </Field>
          <Field label="Metric">
            <Select value={metric} onChange={(e) => setMetric((e.target as any).value)}>
              <option value="cosine">cosine</option>
              <option value="dot">dot</option>
              <option value="l2">l2</option>
            </Select>
          </Field>
          <Field label="Index type">
            <Select value={indexType} onChange={(e) => setIndexType((e.target as any).value)}>
              <option value="flat">flat</option>
              <option value="hnsw">hnsw</option>
              <option value="ivf">ivf</option>
            </Select>
          </Field>
        </div>
      </Card>

      {(progress.length > 0 || busy) && (
        <Card padded>
          <p className="av-section-eyebrow mb-2 flex items-center gap-2">
            <span className="av-dot" /> Build stream
          </p>
          <div className="bg-slate-950 border border-muted rounded-lg p-3 font-mono text-[11px] text-slate-400 h-40 overflow-y-auto space-y-1">
            {progress.map((p, i) => (
              <p key={i}>
                <span className="text-emerald-400">[{p.stage || "step"}]</span>{" "}
                {p.message || ""}
              </p>
            ))}
            {busy && <p className="text-slate-600 italic">…streaming…</p>}
          </div>
        </Card>
      )}

      {result && (
        <Card padded>
          <CardHeader title="Index built" eyebrow="Result" right={<Pill tone="neon">OK</Pill>} />
          <div className="grid grid-cols-2 gap-3 text-xs font-mono">
            <KV label="project_root" value={result.project_root} />
            <KV label="out_dir" value={result.out_dir} />
            {result.summary &&
              Object.entries(result.summary).map(([k, v]) => (
                <KV key={k} label={k} value={String(v)} />
              ))}
          </div>
        </Card>
      )}

      {error && (
        <Card padded className="border-red-500/40">
          <p className="text-xs text-red-300 font-mono">{error}</p>
        </Card>
      )}

      <Card padded>
        <p className="av-section-eyebrow mb-2 flex items-center gap-1">
          <Database className="h-3 w-3" /> Active index location
        </p>
        <div className="grid grid-cols-2 gap-3 text-xs font-mono">
          <KV label="index_dir" value={index.index_dir} />
          <KV label="records" value={index.records} />
          <KV label="embed_model" value={index.embed_model} />
        </div>
      </Card>
    </div>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-950 p-3 rounded border border-white/5 flex justify-between items-center gap-3">
      <span className="text-slate-500 truncate">{label}</span>
      <span className="text-slate-200 truncate text-right">{value}</span>
    </div>
  );
}
