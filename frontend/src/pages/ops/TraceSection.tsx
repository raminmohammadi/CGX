import { useEffect, useMemo, useState } from "react";
import { Card, CardHeader } from "../../components/Card";
import { Pill } from "../../components/Pill";
import { BarList, type BarRow } from "../../components/charts";
import { api, type AdminLogEntry } from "../../lib/api";
import { useWorkspace } from "../../store/workspace";
import { cn, formatRelative } from "../../lib/utils";
import { ErrorLine, fmtMs, useAsync, type SectionProps } from "./common";
import { TraceDetail, catChartTone, catTone, deriveCat, eventTone, recLabel } from "./traceDetail";

// Trailing path segments for a compact source label (full path kept in title).
function shortRoot(root: string): string {
  const parts = root.replace(/\/+$/, "").split("/").filter(Boolean);
  return parts.length <= 2 ? root : "…/" + parts.slice(-2).join("/");
}

export default function TraceSection({ refreshKey }: SectionProps) {
  const projectRoot = useWorkspace((s) => s.projectRoot);
  const [event, setEvent] = useState("");
  const [cat, setCat] = useState("all");
  const [source, setSource] = useState(""); // "" = global fallback log
  const [hideHttp, setHideHttp] = useState(true);
  const [sel, setSel] = useState<AdminLogEntry | null>(null);
  const { data, error } = useAsync(
    () => api.adminLogs({ event: event || undefined, limit: 500, project_root: source || undefined }),
    [refreshKey, event, source],
  );
  // Project roots seen in recent activity runs -> selectable trace sources.
  const { data: runsData } = useAsync(() => api.activityRuns({ limit: 300 }), [refreshKey]);
  const roots = useMemo(() => {
    const set = new Set<string>();
    if (projectRoot) set.add(projectRoot);
    for (const r of runsData?.runs ?? []) if (r.project_root) set.add(r.project_root);
    return [...set];
  }, [projectRoot, runsData]);

  const allLogs = data?.logs ?? [];
  const logs = hideHttp ? allLogs.filter((r) => deriveCat(r) !== "http") : allLogs;
  useEffect(() => setSel(null), [refreshKey, event, cat, source, hideHttp]);

  // Category breakdown (client-side) + the chips used to filter the list.
  const cats = useMemo(() => {
    const m: Record<string, number> = {};
    for (const r of logs) m[deriveCat(r)] = (m[deriveCat(r)] ?? 0) + 1;
    return m;
  }, [logs]);
  const catBars: BarRow[] = Object.entries(cats)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label, value, tone: catChartTone(label) }));
  const rows = cat === "all" ? logs : logs.filter((r) => deriveCat(r) === cat);

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <Card padded>
        <CardHeader
          eyebrow="Admin explorer (D) · Tracing (B)"
          title="Function-call trace"
          description="Pick a project to read its agent.log — LLM calls (full prompt + response), router, executors, codegen, scaffold & repair. The Global source holds only HTTP/CLI traces. Newest first · secrets redacted server-side · click a record for detail."
          right={
            <div className="flex items-center gap-2">
              <select
                value={source}
                onChange={(e) => setSource(e.target.value)}
                className="av-input text-xs w-52"
                title="Trace source (project agent.log vs. global fallback)"
              >
                <option value="">Global (HTTP · CLI)</option>
                {roots.map((r) => (
                  <option key={r} value={r} title={r}>
                    {shortRoot(r)}
                  </option>
                ))}
              </select>
              <input
                value={event}
                onChange={(e) => setEvent(e.target.value)}
                placeholder="filter event/fn…"
                className="av-input text-xs w-40"
              />
              <button
                onClick={() => setHideHttp((v) => !v)}
                className={cn("av-btn-ghost whitespace-nowrap", hideHttp && "text-emerald-400")}
                title="Hide high-volume HTTP request traces"
              >
                {hideHttp ? "HTTP hidden" : "HTTP shown"}
              </button>
            </div>
          }
        />
        {logs.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 mb-3">
            <Chip label="all" count={logs.length} active={cat === "all"} onClick={() => setCat("all")} />
            {Object.entries(cats)
              .sort((a, b) => b[1] - a[1])
              .map(([c, n]) => (
                <Chip key={c} label={c} count={n} active={cat === c} onClick={() => setCat(c)} />
              ))}
          </div>
        )}

        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2 space-y-1 max-h-[32rem] overflow-y-auto text-[11px] font-mono">
            {rows.length === 0 && (
              <p className="text-slate-500 italic">
                No trace records here. Enable tracing (CGX_TRACE=1 or the Settings toggle), run an
                ask/plan/agent, then select its project as the source above — the Global source only
                collects HTTP/CLI traces.
              </p>
            )}
            {rows.map((r, i) => {
              const elapsed = r.elapsed_ms ?? r.latency_ms;
              return (
                <button
                  key={i}
                  onClick={() => setSel(r)}
                  className={cn(
                    "w-full text-left p-2 rounded border flex items-center gap-2 transition",
                    sel === r ? "bg-slate-950 border-emerald-500/30" : "bg-slate-950/40 border-white/5 hover:border-white/10",
                  )}
                >
                  <Pill tone={eventTone(String(r.event))}>{String(r.event ?? "?").replace("trace_", "")}</Pill>
                  <Pill tone={catTone(deriveCat(r))}>{deriveCat(r)}</Pill>
                  <span className="text-slate-300 truncate flex-1">{recLabel(r)}</span>
                  {elapsed != null && <span className="text-slate-500 shrink-0">{fmtMs(Number(elapsed))}</span>}
                  <span className="text-slate-600 shrink-0">{formatRelative(r.ts)}</span>
                </button>
              );
            })}
          </div>

          <div className="space-y-4">
            <Card padded className="!p-3">
              <p className="av-section-eyebrow mb-2">By category</p>
              <BarList data={catBars} />
            </Card>
            <Card padded className="!p-3">
              <p className="av-section-eyebrow mb-2">Record detail</p>
              {!sel && <p className="text-[11px] text-slate-500 font-mono italic">Select a trace record.</p>}
              {sel && <TraceDetail rec={sel} />}
            </Card>
          </div>
        </div>
        <p className="text-[10px] text-slate-500 font-mono mt-3 truncate">source: {data?.source ?? "--"}</p>
      </Card>
    </div>
  );
}

function Chip({ label, count, active, onClick }: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "px-2 py-0.5 rounded border text-[10px] font-mono font-bold uppercase tracking-wider transition",
        active ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" : "bg-slate-900 text-slate-400 border-white/5 hover:border-white/10",
      )}
    >
      {label} <span className="text-slate-600">{count}</span>
    </button>
  );
}
