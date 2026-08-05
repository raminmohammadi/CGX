import { useEffect, useMemo, useState } from "react";
import { Card, CardHeader } from "../../components/Card";
import { Pill, type PillTone } from "../../components/Pill";
import { BarList, type BarRow } from "../../components/charts";
import { api, type AdminLogEntry } from "../../lib/api";
import { cn, formatRelative } from "../../lib/utils";
import { ErrorLine, fmtMs, useAsync, type SectionProps } from "./common";

// Trace category -> pill tone (matches the @traced() categories in cgx.trace).
function catTone(cat: string): PillTone {
  if (cat.startsWith("repair")) return "red";
  if (cat === "llm") return "neon";
  if (cat === "retrieval") return "amber";
  if (cat === "router" || cat === "pipeline") return "purple";
  return "slate"; // codegen + anything else
}
function eventTone(ev: string): PillTone {
  if (ev === "trace_error") return "red";
  if (ev === "trace_exit") return "neon";
  return "slate"; // trace_enter / other
}
const META = new Set(["event", "ts", "category", "fn", "elapsed_ms"]);
const CORRELATION = ["run_id", "request_id", "session_id", "task_id"];

export default function TraceSection({ refreshKey }: SectionProps) {
  const [event, setEvent] = useState("");
  const [cat, setCat] = useState("all");
  const [sel, setSel] = useState<AdminLogEntry | null>(null);
  const { data, error } = useAsync(
    () => api.adminLogs({ event: event || undefined, limit: 500 }),
    [refreshKey, event],
  );
  const logs = data?.logs ?? [];
  useEffect(() => setSel(null), [refreshKey, event, cat]);

  // Category breakdown (client-side) + the chips used to filter the list.
  const cats = useMemo(() => {
    const m: Record<string, number> = {};
    for (const r of logs) m[String(r.category ?? "?")] = (m[String(r.category ?? "?")] ?? 0) + 1;
    return m;
  }, [logs]);
  const catBars: BarRow[] = Object.entries(cats)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label, value, tone: "purple" as const }));
  const rows = cat === "all" ? logs : logs.filter((r) => String(r.category) === cat);

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <Card padded>
        <CardHeader
          eyebrow="Admin explorer (D) · Tracing (B)"
          title="Function-call trace"
          description="Newest first · secrets redacted server-side · click a record for full detail."
          right={
            <input
              value={event}
              onChange={(e) => setEvent(e.target.value)}
              placeholder="filter event/fn…"
              className="av-input text-xs w-44"
            />
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
                No trace records. Enable tracing with CGX_TRACE=1 or the trace toggle in Settings.
              </p>
            )}
            {rows.map((r, i) => (
              <button
                key={i}
                onClick={() => setSel(r)}
                className={cn(
                  "w-full text-left p-2 rounded border flex items-center gap-2 transition",
                  sel === r ? "bg-slate-950 border-emerald-500/30" : "bg-slate-950/40 border-white/5 hover:border-white/10",
                )}
              >
                <Pill tone={eventTone(String(r.event))}>{String(r.event ?? "?").replace("trace_", "")}</Pill>
                {r.category != null && <Pill tone={catTone(String(r.category))}>{String(r.category)}</Pill>}
                <span className="text-slate-300 truncate flex-1">{String(r.fn ?? "")}</span>
                {r.elapsed_ms != null && <span className="text-slate-500 shrink-0">{fmtMs(Number(r.elapsed_ms))}</span>}
                <span className="text-slate-600 shrink-0">{formatRelative(r.ts)}</span>
              </button>
            ))}
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

function TraceDetail({ rec }: { rec: AdminLogEntry }) {
  const ids = CORRELATION.filter((k) => rec[k] != null);
  const extra = Object.entries(rec).filter(([k]) => !META.has(k) && !CORRELATION.includes(k));
  return (
    <div className="space-y-1.5 text-[11px] font-mono">
      <Row label="event" value={String(rec.event ?? "--")} />
      <Row label="category" value={String(rec.category ?? "--")} />
      <Row label="fn" value={String(rec.fn ?? "--")} />
      <Row label="elapsed" value={rec.elapsed_ms != null ? fmtMs(Number(rec.elapsed_ms)) : "--"} />
      {ids.map((k) => (
        <Row key={k} label={k} value={String(rec[k])} accent />
      ))}
      {extra.map(([k, v]) => (
        <Row key={k} label={k} value={typeof v === "object" ? JSON.stringify(v) : String(v)} />
      ))}
    </div>
  );
}

function Row({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="bg-slate-950 p-2 rounded border border-white/5 flex justify-between gap-3">
      <span className="text-slate-500 shrink-0">{label}</span>
      <span className={cn("truncate text-right", accent ? "text-emerald-400" : "text-slate-200")}>{value}</span>
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
