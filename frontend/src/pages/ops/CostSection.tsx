import { Card, CardHeader } from "../../components/Card";
import { Pill } from "../../components/Pill";
import { StatCard } from "../../components/StatCard";
import { BarList, Gauge } from "../../components/charts";
import { api } from "../../lib/api";
import { ErrorLine, fmtCost, fmtNum, useAsync, type SectionProps } from "./common";

const STATE_TONE = { ok: "neon", warn: "amber", exceeded: "red" } as const;

export default function CostSection({ refreshKey }: SectionProps) {
  const { data, error } = useAsync(
    () => Promise.all([api.usageSummary(), api.usage().catch(() => null)]),
    [refreshKey],
  );
  const [summary, own] = data ?? [];
  const rows = [...(summary?.usage ?? [])].sort((a, b) => b.cost_usd - a.cost_usd);

  const totalCost = rows.reduce((s, r) => s + r.cost_usd, 0);
  const totalTokens = rows.reduce((s, r) => s + r.tokens_total, 0);
  const totalCalls = rows.reduce((s, r) => s + r.calls, 0);

  const costBudget =
    own && own.cost_limit > 0 ? own.cost_used / own.cost_limit : null;
  const tokenBudget =
    own && own.tokens_limit > 0 ? own.tokens_used / own.tokens_limit : null;

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Spend today" value={fmtCost(totalCost)} tone="neon" />
        <StatCard label="Tokens today" value={fmtNum(totalTokens)} />
        <StatCard label="Calls today" value={fmtNum(totalCalls)} />
        <StatCard label="Owners" value={rows.length} />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card padded className="col-span-2">
          <CardHeader
            eyebrow="Cost & quota (I)"
            title="Spend by owner"
            description="Metered per-owner cost for the current UTC day."
          />
          <BarList
            data={rows.slice(0, 8).map((r) => ({ label: r.owner, value: r.cost_usd, sub: `${fmtNum(r.tokens_total)} tok` }))}
            format={(v) => fmtCost(v)}
          />
        </Card>
        <Card padded>
          <CardHeader eyebrow="You" title="Budget" description={own?.owner ? own.owner : undefined} />
          {own == null ? (
            <p className="text-[11px] text-slate-500 font-mono italic">Usage unavailable.</p>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="flex gap-4">
                <Gauge
                  value={costBudget}
                  label="cost"
                  tone={own.state === "exceeded" ? "red" : own.state === "warn" ? "amber" : "emerald"}
                  size={104}
                />
                <Gauge value={tokenBudget} label="tokens" tone="blue" size={104} />
              </div>
              <Pill tone={STATE_TONE[own.state as keyof typeof STATE_TONE] ?? "slate"}>{own.state}</Pill>
              <p className="text-[10px] text-slate-500 font-mono text-center">
                {fmtCost(own.cost_used)} / {own.cost_limit > 0 ? fmtCost(own.cost_limit) : "∞"}
              </p>
            </div>
          )}
        </Card>
      </div>

      <Card padded>
        <CardHeader eyebrow="Detail" title="Per-owner usage" />
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] font-mono">
            <thead className="text-slate-500 uppercase tracking-wider">
              <tr className="border-b border-white/5">
                <th className="text-left py-1.5">Owner</th>
                <th className="text-right">Cost</th>
                <th className="text-right">Tokens in</th>
                <th className="text-right">Tokens out</th>
                <th className="text-right">Calls</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-3 text-slate-500 italic text-center">
                    No metered usage yet.
                  </td>
                </tr>
              )}
              {rows.map((r) => (
                <tr key={r.owner} className="border-b border-white/5">
                  <td className="text-left py-1.5 text-slate-300">{r.owner}</td>
                  <td className="text-right text-emerald-400">{fmtCost(r.cost_usd)}</td>
                  <td className="text-right text-slate-300">{fmtNum(r.tokens_in)}</td>
                  <td className="text-right text-slate-300">{fmtNum(r.tokens_out)}</td>
                  <td className="text-right text-slate-300">{fmtNum(r.calls)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
