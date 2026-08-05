import { useState } from "react";
import { Card, CardHeader } from "../../components/Card";
import { Pill } from "../../components/Pill";
import { StatCard } from "../../components/StatCard";
import { BarList, type BarRow } from "../../components/charts";
import { api } from "../../lib/api";
import { cn, formatRelative } from "../../lib/utils";
import { ErrorLine, SEV_TONE, useAsync, type SectionProps } from "./common";

const SEVERITIES = ["all", "critical", "warning", "info"] as const;

export default function MonitoringSection({ refreshKey }: SectionProps) {
  const [sev, setSev] = useState<(typeof SEVERITIES)[number]>("all");
  const { data, error, loading } = useAsync(
    () => api.monitorAlerts({ limit: 300, severity: sev === "all" ? undefined : sev }),
    [refreshKey, sev],
  );
  const alerts = data?.alerts ?? [];

  const byCode = alerts.reduce<Record<string, number>>((m, a) => {
    m[a.code] = (m[a.code] ?? 0) + 1;
    return m;
  }, {});
  const codeBars: BarRow[] = Object.entries(byCode)
    .sort((a, b) => b[1] - a[1])
    .map(([code, n]) => ({
      label: code,
      value: n,
      tone: code.startsWith("guardrail") ? "purple" : "amber",
    }));
  const counts = {
    critical: alerts.filter((a) => a.severity === "critical").length,
    warning: alerts.filter((a) => a.severity === "warning").length,
    info: alerts.filter((a) => a.severity === "info").length,
  };

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Alerts" value={loading ? "…" : alerts.length} tone="amber" />
        <StatCard label="Critical" value={counts.critical} tone={counts.critical ? "red" : "slate"} />
        <StatCard label="Warning" value={counts.warning} tone={counts.warning ? "amber" : "slate"} />
        <StatCard label="Info" value={counts.info} />
      </div>

      <Card padded>
        <CardHeader
          eyebrow="AIOps (G) + Guardrails (K)"
          title="Alerts by code"
          description="Groundedness, drift, cost-anomaly, repair-health and guardrail findings."
        />
        <BarList data={codeBars} />
      </Card>

      <Card padded>
        <CardHeader
          eyebrow="Recent"
          title="Alert stream"
          right={
            <div className="flex items-center gap-1.5">
              {SEVERITIES.map((s) => (
                <button
                  key={s}
                  onClick={() => setSev(s)}
                  className={cn("av-btn-ghost capitalize", sev === s && "text-emerald-400")}
                >
                  {s}
                </button>
              ))}
            </div>
          }
        />
        <div className="space-y-1 max-h-[26rem] overflow-y-auto text-[11px] font-mono">
          {alerts.length === 0 && <p className="text-slate-500 italic">No alerts recorded.</p>}
          {alerts.map((a) => (
            <div
              key={a.alert_id}
              className="flex items-center gap-2 bg-slate-950 p-2 rounded border border-white/5"
            >
              <Pill tone={SEV_TONE[a.severity] ?? "slate"}>{a.severity}</Pill>
              <span className="text-slate-300 shrink-0">{a.code}</span>
              <span className="text-slate-500 truncate flex-1">{a.message}</span>
              {a.value != null && (
                <span className="text-amber-400 shrink-0">
                  {a.value}
                  {a.threshold != null && <span className="text-slate-600">/{a.threshold}</span>}
                </span>
              )}
              {a.run_id && <span className="text-slate-600 shrink-0 truncate max-w-[8rem]">{a.run_id}</span>}
              <span className="text-slate-600 shrink-0">{formatRelative(a.created_at)}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
