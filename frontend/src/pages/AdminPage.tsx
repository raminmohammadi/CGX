import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Card, CardHeader } from "../components/Card";
import { Pill, type PillTone } from "../components/Pill";
import { StatCard } from "../components/StatCard";
import {
  api,
  type AdminLogEntry,
  type AdminOverview,
  type MetricsSnapshot,
} from "../lib/api";
import { cn, formatRelative } from "../lib/utils";

const SEV_TONE: Record<string, PillTone> = {
  critical: "red",
  warning: "amber",
  info: "slate",
};

export default function AdminPage() {
  const [ov, setOv] = useState<AdminOverview | null>(null);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [logs, setLogs] = useState<AdminLogEntry[]>([]);
  const [event, setEvent] = useState("");
  const [tab, setTab] = useState<"logs" | "metrics">("logs");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [o, m, l] = await Promise.all([
        api.adminOverview(),
        api.adminMetrics(),
        api.adminLogs({ event: event || undefined, limit: 300 }),
      ]);
      setOv(o);
      setMetrics(m);
      setLogs(l.logs);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }, [event]);

  useEffect(() => {
    void load();
  }, [load]);

  const sat = ov?.feedback?.satisfaction;
  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-6xl">
      <CardHeader
        title="Admin"
        description="Operator view — trace explorer, metrics and audit-lite health, from /api/admin."
        right={
          <button onClick={() => void load()} disabled={loading} className="av-btn-ghost">
            <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} /> Refresh
          </button>
        }
      />

      <div className="grid grid-cols-4 gap-4">
        <StatCard label="HTTP requests" value={ov?.http.requests ?? "--"} tone="neon" />
        <StatCard
          label="HTTP 5xx"
          value={ov?.http.errors ?? "--"}
          tone={ov && ov.http.errors > 0 ? "red" : "slate"}
        />
        <StatCard
          label="Alerts"
          value={ov?.alerts.total ?? "--"}
          tone={ov && ov.alerts.total > 0 ? "amber" : "slate"}
          caption={ov ? Object.entries(ov.alerts.by_severity).map(([k, v]) => `${k}:${v}`).join("  ") : undefined}
        />
        <StatCard
          label="Satisfaction"
          value={sat == null ? "--" : `${Math.round(sat * 100)}%`}
          caption={ov ? `${ov.feedback.up ?? 0}▲ / ${ov.feedback.down ?? 0}▼` : undefined}
        />
      </div>

      {error && <p className="text-xs text-red-400 font-mono">{error}</p>}

      <Card padded>
        <CardHeader eyebrow="Recent" title="Alerts" description="Latest quality/drift/cost incidents." />
        <div className="space-y-1 max-h-40 overflow-y-auto text-[11px] font-mono">
          {(ov?.alerts.recent ?? []).length === 0 && (
            <p className="text-slate-500 italic">No alerts recorded.</p>
          )}
          {(ov?.alerts.recent ?? []).map((a, i) => (
            <div key={i} className="flex items-center gap-2 bg-slate-950 p-2 rounded border border-white/5">
              <Pill tone={SEV_TONE[String(a.severity)] ?? "slate"}>{a.severity}</Pill>
              <span className="text-slate-300">{a.code}</span>
              <span className="text-slate-500 truncate flex-1">{a.message}</span>
              <span className="text-slate-600">{formatRelative(a.created_at)}</span>
            </div>
          ))}
        </div>
      </Card>

      <div className="flex items-center gap-2">
        {(["logs", "metrics"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn("av-btn-ghost capitalize", tab === t && "text-emerald-400")}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "logs" && (
        <Card padded>
          <CardHeader
            eyebrow="Trace"
            title="Logs"
            description="Newest first · secrets redacted server-side."
            right={
              <input
                value={event}
                onChange={(e) => setEvent(e.target.value)}
                placeholder="filter event…"
                className="av-input text-xs w-40"
              />
            }
          />
          <div className="space-y-1 max-h-[26rem] overflow-y-auto text-[11px] font-mono">
            {logs.length === 0 && <p className="text-slate-500 italic">No trace records.</p>}
            {logs.map((r, i) => (
              <div key={i} className="bg-slate-950 p-2 rounded border border-white/5 flex gap-2">
                <span className="text-emerald-400 shrink-0">{String(r.event ?? "?")}</span>
                <span className="text-slate-400 truncate flex-1">
                  {JSON.stringify(Object.fromEntries(Object.entries(r).filter(([k]) => k !== "event" && k !== "ts")))}
                </span>
                <span className="text-slate-600 shrink-0">{formatRelative(r.ts)}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {tab === "metrics" && <MetricsTable snapshot={metrics} />}
    </div>
  );
}

function MetricsTable({ snapshot }: { snapshot: MetricsSnapshot | null }) {
  const series = [...(snapshot?.counters ?? []), ...(snapshot?.gauges ?? [])];
  return (
    <Card padded>
      <CardHeader eyebrow="Registry" title="Metrics" description="Counters and gauges (in-process)." />
      <div className="space-y-1 max-h-[26rem] overflow-y-auto text-[11px] font-mono">
        {series.length === 0 && <p className="text-slate-500 italic">No metrics recorded.</p>}
        {series.map((s, i) => (
          <div key={i} className="bg-slate-950 p-2 rounded border border-white/5 flex justify-between gap-3">
            <span className="text-slate-300 truncate">
              {s.name}
              <span className="text-slate-600">
                {Object.keys(s.labels).length ? ` {${Object.entries(s.labels).map(([k, v]) => `${k}=${v}`).join(",")}}` : ""}
              </span>
            </span>
            <span className="text-emerald-400 shrink-0">{s.value}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}
