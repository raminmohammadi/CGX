import type { ReactNode } from "react";
import { Card, CardHeader } from "../../components/Card";
import { Pill } from "../../components/Pill";
import { BarList, type BarRow } from "../../components/charts";
import { api, type MetricsSnapshot } from "../../lib/api";
import { ErrorLine, useAsync, type SectionProps } from "./common";
import { IndexingCard, RetrievalCard, ThroughputCard } from "./pipelineCards";

const EMPTY_SNAP: MetricsSnapshot = { counters: [], gauges: [], histograms: [] };

function countBy(items: (string | null | undefined)[]): BarRow[] {
  const m: Record<string, number> = {};
  for (const it of items) {
    const key = it || "(none)";
    m[key] = (m[key] ?? 0) + 1;
  }
  return Object.entries(m)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([label, value]) => ({ label, value, tone: "purple" as const }));
}

export default function PipelineSection({ refreshKey }: SectionProps) {
  const { data, error } = useAsync(
    () =>
      Promise.all([
        api.activityRuns({ limit: 300 }),
        api.monitorAlerts({ limit: 300 }),
        api.adminMetrics(),
        api.activitySummary(),
      ]),
    [refreshKey],
  );
  const [runs, alerts, metrics, summary] = data ?? [];
  const snap = metrics ?? EMPTY_SNAP;
  const byKind = summary?.by_kind ?? {};
  const models = countBy((runs?.runs ?? []).map((r) => r.model));
  const prompts = countBy((runs?.runs ?? []).map((r) => r.prompt_version));
  const guardrailAlerts = (alerts?.alerts ?? []).filter((a) => a.code.startsWith("guardrail"));
  const guardrailEvents = (metrics?.counters ?? [])
    .filter((c) => c.name === "cgx_guardrail_events_total")
    .reduce((t, c) => t + c.value, 0);

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />

      <div>
        <p className="av-section-eyebrow mb-2">Live pipelines — driven by always-on metrics + activity</p>
        <div className="grid grid-cols-2 gap-4">
          <IndexingCard snap={snap} />
          <RetrievalCard snap={snap} />
        </div>
        <div className="grid grid-cols-3 gap-4 mt-4">
          <ThroughputCard
            eyebrow="Ask / QA"
            title="Contextual answers"
            description="Grounded QA runs (kind=ask): retrieve → rerank → answer with citations."
            entry={byKind["ask"]}
          />
          <ThroughputCard
            eyebrow="Codegen / Plan"
            title="Self-testing plans"
            description="Plan + codegen runs (kind=plan): diffs, tests and guardrail scans."
            entry={byKind["plan"]}
          />
          <ThroughputCard
            eyebrow="Agent loop"
            title="Autonomous runs"
            description="Router-driven agent sessions (kind=agent): explore → act → verify → repair."
            entry={byKind["agent"]}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card padded>
          <CardHeader eyebrow="Lineage & registry (F)" title="Models observed" description="Distinct models across recent runs (provenance join key: run_id)." />
          <BarList data={models} />
        </Card>
        <Card padded>
          <CardHeader eyebrow="Lineage & registry (F)" title="Prompt versions" description="Fingerprinted prompt templates seen in recent runs." />
          <BarList data={prompts} />
        </Card>
      </div>

      <Card padded>
        <CardHeader
          eyebrow="Guardrails & safety (K)"
          title="Defensive layers"
          right={<Pill tone={guardrailAlerts.length ? "amber" : "neon"}>{guardrailAlerts.length} findings</Pill>}
          description="Prompt-injection (input), secret/path leakage (output), and the CGX_LLM_DISABLED kill-switch. Findings are advisory: mirrored to metrics + the alert store."
        />
        <div className="grid grid-cols-3 gap-3 text-[11px] font-mono">
          <Metric label="Guardrail events (metric)" value={guardrailEvents} />
          <Metric label="Guardrail alerts" value={guardrailAlerts.length} />
          <Metric label="Kill-switch" value="CGX_LLM_DISABLED" />
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <InfoCard eyebrow="Evaluation (E)" title="Offline eval + CI gate">
          <p>Golden sets live under <code className="text-slate-300">evals/</code>; the harness runs as a module and gates CI when a metric regresses.</p>
          <pre className="bg-slate-950 border border-white/5 rounded p-2 mt-2 text-slate-300 whitespace-pre-wrap">python -m cgx.eval retrieval --golden evals/retrieval_golden.jsonl
python -m cgx.eval codegen   --golden evals/codegen_golden.jsonl</pre>
          <p className="mt-2">Down-votes from Feedback (H) drain into eval candidates via the flywheel.</p>
        </InfoCard>
        <InfoCard eyebrow="Packaging & deployment (L)" title="Containers & Kubernetes">
          <p>A multi-stage <code className="text-slate-300">Dockerfile</code>, a <code className="text-slate-300">docker-compose.yml</code> bundling CGX + Prometheus + Grafana, and a Helm chart under <code className="text-slate-300">deploy/helm/cgx</code>.</p>
          <p className="mt-2">Probes wire to <code className="text-slate-300">/healthz</code> + <code className="text-slate-300">/readyz</code>; Grafana ships the time-series dashboards for <code className="text-slate-300">/api/metrics</code>. See <code className="text-slate-300">deploy/README.md</code>.</p>
        </InfoCard>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="bg-slate-950 p-2.5 rounded border border-white/5">
      <p className="av-section-eyebrow mb-1">{label}</p>
      <p className="text-emerald-400 font-bold truncate">{value}</p>
    </div>
  );
}

function InfoCard({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return (
    <Card padded>
      <CardHeader eyebrow={eyebrow} title={title} />
      <div className="text-[11px] font-mono text-slate-400 leading-relaxed">{children}</div>
    </Card>
  );
}
