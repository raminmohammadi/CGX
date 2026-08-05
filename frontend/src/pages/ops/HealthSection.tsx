import { Activity, CheckCircle2, XCircle } from "lucide-react";
import { Card, CardHeader } from "../../components/Card";
import { Pill } from "../../components/Pill";
import { StatCard } from "../../components/StatCard";
import { Gauge } from "../../components/charts";
import { api } from "../../lib/api";
import { ErrorLine, useAsync, type SectionProps } from "./common";

export default function HealthSection({ refreshKey }: SectionProps) {
  const { data, error } = useAsync(
    () => Promise.all([api.liveness().catch(() => null), api.readiness().catch(() => null)]),
    [refreshKey],
  );
  const [live, ready] = data ?? [];
  const checks = ready?.checks ?? [];
  const passed = checks.filter((c) => c.ok).length;

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Liveness"
          value={live?.status ?? "--"}
          tone={live?.status === "ok" ? "neon" : "red"}
        />
        <StatCard
          label="Readiness"
          value={ready ? (ready.ready ? "ready" : "not ready") : "--"}
          tone={ready?.ready ? "neon" : "red"}
        />
        <StatCard label="Checks passing" value={ready ? `${passed}/${checks.length}` : "--"} />
        <StatCard
          label="Critical failing"
          value={checks.filter((c) => c.critical && !c.ok).length}
          tone={checks.some((c) => c.critical && !c.ok) ? "red" : "slate"}
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card padded className="flex items-center justify-center">
          <Gauge
            value={checks.length ? passed / checks.length : null}
            display={ready ? `${passed}/${checks.length}` : "--"}
            label="Readiness (J)"
            tone={ready?.ready ? "emerald" : "red"}
          />
        </Card>
        <Card padded className="col-span-2">
          <CardHeader
            eyebrow="Reliability & health (J)"
            title="Readiness probes"
            description="Critical checks gate /readyz (HTTP 503 when failing); others are informational."
          />
          <div className="space-y-1 text-[11px] font-mono">
            {checks.length === 0 && <p className="text-slate-500 italic">No readiness report.</p>}
            {checks.map((c) => (
              <div key={c.name} className="flex items-center gap-2 bg-slate-950 p-2 rounded border border-white/5">
                {c.ok ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                ) : (
                  <XCircle className="h-3.5 w-3.5 text-red-400 shrink-0" />
                )}
                <span className="text-slate-300 shrink-0">{c.name}</span>
                {c.critical ? (
                  <Pill tone="purple">critical</Pill>
                ) : (
                  <Pill tone="slate">info</Pill>
                )}
                <span className="text-slate-500 truncate flex-1">
                  {Object.entries(c.detail)
                    .map(([k, v]) => `${k}=${v}`)
                    .join("  ")}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <p className="text-[10px] text-slate-500 font-mono flex items-center gap-1.5">
        <Activity className="h-3 w-3" /> Probes: <span className="text-slate-400">GET /healthz</span> ·{" "}
        <span className="text-slate-400">GET /readyz</span> — wired to Kubernetes liveness/readiness in
        the Helm chart.
      </p>
    </div>
  );
}
