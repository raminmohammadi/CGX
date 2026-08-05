import { Pill, type PillTone } from "../../components/Pill";
import { type ChartTone } from "../../components/charts";
import type { AdminLogEntry } from "../../lib/api";
import { cn } from "../../lib/utils";
import { fmtMs } from "./common";

// A record's logical category. @traced records carry ``category``; llm_call
// records don't, so derive one so they group + colour with the LLM traces.
export function deriveCat(rec: AdminLogEntry): string {
  if (rec.category != null) return String(rec.category);
  const ev = String(rec.event ?? "");
  if (ev === "llm_call") return "llm";
  return "event";
}

export function catTone(cat: string): PillTone {
  if (cat.startsWith("repair")) return "red";
  if (cat === "llm") return "neon";
  if (cat === "retrieval") return "amber";
  if (cat === "router" || cat === "pipeline") return "purple";
  return "slate"; // codegen / http / event / anything else
}
// BarList/charts use ChartTone (no "neon"); map the pill tone across.
export function catChartTone(cat: string): ChartTone {
  const t = catTone(cat);
  return t === "neon" ? "emerald" : t;
}
export function eventTone(ev: string): PillTone {
  if (ev === "trace_error" || ev.includes("error")) return "red";
  if (ev === "llm_call") return "neon";
  if (ev === "trace_exit") return "neon";
  return "slate";
}

// One-line label for a record row: the fn for @traced records, model for an
// llm_call, else the raw event name.
export function recLabel(rec: AdminLogEntry): string {
  const ev = String(rec.event ?? "");
  if (ev === "llm_call") return String(rec.model ?? rec.component ?? "llm call");
  if (rec.fn != null) return String(rec.fn);
  return ev || "(record)";
}

const CORRELATION = ["fact_id", "run_id", "request_id", "session_id", "task_id"];
const META = new Set([
  ...CORRELATION, "event", "ts", "kind", "category", "fn", "elapsed_ms",
  "args", "error", "error_type", "component", "model", "latency_ms", "streamed",
  "sampling", "prompt_full", "response_full", "prompt_preview",
  "response_preview", "prompt_chars", "response_chars",
]);

export function TraceDetail({ rec }: { rec: AdminLogEntry }) {
  const ev = String(rec.event ?? "");
  return ev === "llm_call" ? <LlmDetail rec={rec} /> : <GenericDetail rec={rec} />;
}

function LlmDetail({ rec }: { rec: AdminLogEntry }) {
  const prompt = String(rec.prompt_full ?? rec.prompt_preview ?? "");
  const response = String(rec.response_full ?? rec.response_preview ?? "");
  const s = (rec.sampling as Record<string, unknown>) ?? {};
  return (
    <div className="space-y-1.5 text-[11px] font-mono">
      <div className="flex items-center gap-1.5">
        <Pill tone="neon">llm call</Pill>
        {rec.streamed ? <Pill tone="purple">streamed</Pill> : null}
      </div>
      <Row label="component" value={String(rec.component ?? "--")} />
      <Row label="model" value={String(rec.model ?? "--")} />
      <Row label="latency" value={rec.latency_ms != null ? fmtMs(Number(rec.latency_ms)) : "--"} />
      <Row label="prompt / response" value={`${rec.prompt_chars ?? "?"} → ${rec.response_chars ?? "?"} chars`} />
      {(s.temperature != null || s.max_tokens != null) && (
        <Row label="sampling" value={`temp=${s.temperature ?? "?"} · max_tokens=${s.max_tokens ?? "?"}`} />
      )}
      <Ids rec={rec} />
      {rec.error != null && <Row label="error" value={String(rec.error)} tone="red" />}
      <Block label="prompt" text={prompt} />
      <Block label="response" text={response} tone="emerald" />
    </div>
  );
}

function GenericDetail({ rec }: { rec: AdminLogEntry }) {
  const args = rec.args as Record<string, unknown> | undefined;
  const extra = Object.entries(rec).filter(([k]) => !META.has(k));
  return (
    <div className="space-y-1.5 text-[11px] font-mono">
      <Row label="event" value={String(rec.event ?? "--")} />
      <Row label="category" value={deriveCat(rec)} />
      {rec.fn != null && <Row label="fn" value={String(rec.fn)} />}
      {rec.elapsed_ms != null && <Row label="elapsed" value={fmtMs(Number(rec.elapsed_ms))} />}
      <Ids rec={rec} />
      {rec.error_type != null && <Row label="error_type" value={String(rec.error_type)} tone="red" />}
      {rec.error != null && <Row label="error" value={String(rec.error)} tone="red" />}
      {args && Object.keys(args).length > 0 && (
        <Block label="args" text={Object.entries(args).map(([k, v]) => `${k} = ${String(v)}`).join("\n")} />
      )}
      {extra.map(([k, v]) => (
        <Row key={k} label={k} value={typeof v === "object" ? JSON.stringify(v) : String(v)} />
      ))}
    </div>
  );
}

function Ids({ rec }: { rec: AdminLogEntry }) {
  const ids = CORRELATION.filter((k) => rec[k] != null);
  if (ids.length === 0) return null;
  return (
    <>
      {ids.map((k) => (
        <Row key={k} label={k} value={String(rec[k])} accent />
      ))}
    </>
  );
}

function Row({ label, value, accent, tone }: { label: string; value: string; accent?: boolean; tone?: "red" }) {
  return (
    <div className="bg-slate-950 p-2 rounded border border-white/5 flex justify-between gap-3">
      <span className="text-slate-500 shrink-0">{label}</span>
      <span className={cn("truncate text-right", tone === "red" ? "text-red-400" : accent ? "text-emerald-400" : "text-slate-200")}>
        {value}
      </span>
    </div>
  );
}

function Block({ label, text, tone }: { label: string; text: string; tone?: "emerald" }) {
  return (
    <div className="bg-slate-950 rounded border border-white/5">
      <p className="av-section-eyebrow px-2 pt-2">{label}</p>
      {text ? (
        <pre className={cn("p-2 text-[10px] whitespace-pre-wrap break-words max-h-56 overflow-y-auto", tone === "emerald" ? "text-emerald-300/90" : "text-slate-300")}>
          {text}
        </pre>
      ) : (
        <p className="p-2 text-slate-600 italic">(empty)</p>
      )}
    </div>
  );
}
