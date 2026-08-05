import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { CardHeader } from "../components/Card";
import { TabBar } from "../components/TabBar";
import { cn } from "../lib/utils";
import OverviewSection from "./ops/OverviewSection";
import ActivitySection from "./ops/ActivitySection";
import MonitoringSection from "./ops/MonitoringSection";
import CostSection from "./ops/CostSection";
import FeedbackSection from "./ops/FeedbackSection";
import MetricsSection from "./ops/MetricsSection";
import GovernanceSection from "./ops/GovernanceSection";
import HealthSection from "./ops/HealthSection";
import TraceSection from "./ops/TraceSection";
import PipelineSection from "./ops/PipelineSection";
import type { SectionProps } from "./ops/common";

type TabKey =
  | "overview"
  | "activity"
  | "monitoring"
  | "cost"
  | "feedback"
  | "metrics"
  | "governance"
  | "health"
  | "trace"
  | "pipeline";

const TABS: { key: TabKey; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "pipeline", label: "Pipelines" },
  { key: "activity", label: "Activity" },
  { key: "monitoring", label: "Monitoring" },
  { key: "cost", label: "Cost & Quota" },
  { key: "feedback", label: "Feedback" },
  { key: "metrics", label: "Metrics" },
  { key: "governance", label: "Governance" },
  { key: "health", label: "Health" },
  { key: "trace", label: "Trace" },
];

const SECTIONS: Record<TabKey, (p: SectionProps) => React.ReactElement> = {
  overview: OverviewSection,
  activity: ActivitySection,
  monitoring: MonitoringSection,
  cost: CostSection,
  feedback: FeedbackSection,
  metrics: MetricsSection,
  governance: GovernanceSection,
  health: HealthSection,
  trace: TraceSection,
  pipeline: PipelineSection,
};

export default function OpsPage() {
  const [tab, setTab] = useState<TabKey>("overview");
  const [refreshKey, setRefreshKey] = useState(0);
  const [auto, setAuto] = useState(false);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => setRefreshKey((k) => k + 1), 10000);
    return () => clearInterval(id);
  }, [auto]);

  const Section = SECTIONS[tab];
  return (
    <div className="p-6 space-y-5 overflow-y-auto h-full max-w-6xl">
      <CardHeader
        eyebrow="MLOps & production"
        title="Ops & Observability"
        description="Every production subsystem from docs/mlops.md — metrics, activity, alerts, cost, feedback, governance and health — in one place."
        right={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setAuto((a) => !a)}
              className={cn("av-btn-ghost", auto && "text-emerald-400")}
              title="Auto-refresh every 10s"
            >
              Auto {auto ? "on" : "off"}
            </button>
            <button onClick={() => setRefreshKey((k) => k + 1)} className="av-btn-ghost">
              <RefreshCw className="h-3 w-3" /> Refresh
            </button>
          </div>
        }
      />

      <TabBar tabs={TABS} active={tab} onChange={(k) => setTab(k)} />

      <Section refreshKey={refreshKey} goto={(t) => setTab(t as TabKey)} />
    </div>
  );
}
