import { Card, CardHeader } from "../../components/Card";
import { StatCard } from "../../components/StatCard";
import { BarList, Donut, Gauge, type DonutSlice } from "../../components/charts";
import { api } from "../../lib/api";
import { ErrorLine, fmtCost, fmtNum, useAsync, type SectionProps } from "./common";

const KIND_TONE: Record<string, DonutSlice["tone"]> = {
  ask: "emerald",
  plan: "blue",
  agent: "purple",
};
const SEV_TONE: Record<string, DonutSlice["tone"]> = {
  critical: "red",
  warning: "amber",
  info: "slate",
};

export default function OverviewSection({ refreshKey, goto }: SectionProps) {
  const { data, error } = useAsync(
    () =>
      Promise.all([
        api.activitySummary(),
        api.feedbackStats(),
        api.monitorAlerts({ limit: 200 }),
        api.usageSummary(),
        api.readiness().catch(() => null),
      ]),
    [refreshKey],
  );
  const [act, fb, alerts, usage, ready] = data ?? [];

  const kindSlices: DonutSlice[] = Object.entries(act?.by_kind ?? {}).map(([k, v]) => ({
    label: k,
    value: v.runs,
    tone: KIND_TONE[k] ?? "slate",
  }));
  const sevCounts = (alerts?.alerts ?? []).reduce<Record<string, number>>((m, a) => {
    m[a.severity] = (m[a.severity] ?? 0) + 1;
    return m;
  }, {});
  const sevSlices: DonutSlice[] = Object.entries(sevCounts).map(([k, v]) => ({
    label: k,
    value: v,
    tone: SEV_TONE[k] ?? "slate",
  }));
  const readyOk = (ready?.checks ?? []).filter((c) => c.ok).length;
  const readyN = (ready?.checks ?? []).length;
  const costByOwner = [...(usage?.usage ?? [])]
    .sort((a, b) => b.cost_usd - a.cost_usd)
    .slice(0, 6)
    .map((u) => ({ label: u.owner, value: u.cost_usd, sub: `${u.calls} calls` }));

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Runs" value={fmtNum(act?.total)} tone="neon" />
        <StatCard label="Cost recorded" value={fmtCost(act?.cost_usd)} caption="all runs" />
        <StatCard label="Tokens" value={fmtNum(act?.tokens_total)} />
        <StatCard
          label="Errors"
          value={fmtNum(act?.errors)}
          tone={act && act.errors > 0 ? "red" : "slate"}
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card padded>
          <CardHeader eyebrow="Activity (C)" title="Runs by kind" />
          <Donut data={kindSlices} centerValue={act?.total ?? 0} centerLabel="runs" />
        </Card>
        <Card padded>
          <CardHeader
            eyebrow="AIOps (G)"
            title="Alerts by severity"
            right={
              <button className="av-btn-ghost" onClick={() => goto?.("monitoring")}>
                View all
              </button>
            }
          />
          <Donut data={sevSlices} centerValue={alerts?.count ?? 0} centerLabel="alerts" />
        </Card>
        <Card padded className="flex items-center justify-around">
          <Gauge value={fb?.satisfaction ?? null} label="Satisfaction (H)" tone="emerald" />
          <Gauge
            value={readyN ? readyOk / readyN : null}
            display={ready ? `${readyOk}/${readyN}` : "--"}
            label="Ready (J)"
            tone={ready?.ready ? "emerald" : "red"}
          />
        </Card>
      </div>

      <Card padded>
        <CardHeader
          eyebrow="Cost & quota (I)"
          title="Spend by owner (today)"
          description="Metered per-owner cost for the current UTC day."
          right={
            <button className="av-btn-ghost" onClick={() => goto?.("cost")}>
              Details
            </button>
          }
        />
        <BarList data={costByOwner} format={(v) => fmtCost(v)} />
      </Card>
    </div>
  );
}
