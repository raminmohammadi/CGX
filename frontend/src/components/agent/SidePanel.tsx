import { useState } from "react";
import { PanelRightClose } from "lucide-react";
import type { ArtifactDTO, DecisionDTO, FactDTO } from "../../lib/api";
import { cn, formatRelative } from "../../lib/utils";

type Tab = "facts" | "artifacts" | "decisions";

// Right-rail tabbed panel: surfaces the session's KB (facts), all
// emitted artifacts, and the audit trail of resolved decisions. Each
// tab is intentionally lightweight; clicking an artifact selects it in
// the parent so the centre pane swaps in the matching task.
export function SidePanel({
  facts, artifacts, decisions, onSelectArtifact, width, onCollapse,
}: {
  facts: FactDTO[];
  artifacts: ArtifactDTO[];
  decisions: DecisionDTO[];
  onSelectArtifact: (taskId: string) => void;
  width?: number;
  onCollapse?: () => void;
}) {
  const [tab, setTab] = useState<Tab>("artifacts");
  return (
    <aside
      style={width ? { width } : undefined}
      className={cn(
        "shrink-0 border-l border-muted bg-slate-950/30 flex flex-col min-w-0",
        width === undefined && "w-72",
      )}
    >
      <div className="flex border-b border-muted">
        <TabBtn active={tab === "artifacts"} onClick={() => setTab("artifacts")}>
          Artifacts <Count n={artifacts.length} />
        </TabBtn>
        <TabBtn active={tab === "facts"} onClick={() => setTab("facts")}>
          Facts <Count n={facts.length} />
        </TabBtn>
        <TabBtn active={tab === "decisions"} onClick={() => setTab("decisions")}>
          Decisions <Count n={decisions.length} />
        </TabBtn>
        {onCollapse && (
          <button
            type="button" onClick={onCollapse} title="Hide artifacts panel"
            className="av-btn-icon h-7 w-7 ml-auto mr-1 self-center"
          ><PanelRightClose className="h-3 w-3" /></button>
        )}
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
        {tab === "artifacts" && (
          <ArtifactsTab artifacts={artifacts} onSelect={onSelectArtifact} />
        )}
        {tab === "facts" && <FactsTab facts={facts} />}
        {tab === "decisions" && <DecisionsTab decisions={decisions} />}
      </div>
    </aside>
  );
}

function TabBtn({
  active, children, onClick,
}: { active: boolean; children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex-1 px-3 py-2 text-[10px] font-mono uppercase tracking-wider transition",
        active
          ? "text-emerald-300 border-b-2 border-emerald-500/50 bg-emerald-500/5"
          : "text-slate-500 hover:text-slate-300 border-b-2 border-transparent",
      )}
    >
      {children}
    </button>
  );
}

function Count({ n }: { n: number }) {
  if (n === 0) return null;
  return (
    <span className="ml-1 px-1 py-px rounded bg-slate-800 text-slate-300 text-[9px]">
      {n}
    </span>
  );
}

function ArtifactsTab({
  artifacts, onSelect,
}: { artifacts: ArtifactDTO[]; onSelect: (taskId: string) => void }) {
  if (artifacts.length === 0) return <Empty>No artifacts yet.</Empty>;
  const sorted = [...artifacts].sort((a, b) => b.created_at - a.created_at);
  return (
    <ul className="space-y-1.5">
      {sorted.map((a) => (
        <li key={a.artifact_id}>
          <button
            type="button"
            onClick={() => onSelect(a.produced_by_task_id)}
            className="w-full text-left rounded border border-white/5 bg-slate-950/60 px-2 py-1.5 hover:border-emerald-500/30"
          >
            <p className="text-[11px] font-mono text-slate-200 truncate">
              {a.kind}
            </p>
            <p className="text-[9px] font-mono text-slate-500">
              {formatRelative(a.created_at)} · {a.artifact_id.slice(0, 8)}
            </p>
          </button>
        </li>
      ))}
    </ul>
  );
}

// Per-swarm-phase accent so the beat feed reads as grouped phases (planning →
// generating → verifying) rather than an undifferentiated "swarm_beat" pile.
const SWARM_ROLE_ACCENT: Record<string, string> = {
  tech_lead: "text-fuchsia-300",
  developer: "text-emerald-300",
  verify: "text-cyan-300",
};

// One-line summary of a swarm beat's payload, so each row says what happened
// (file written, tool called, plan rejected) instead of just its kind.
function summarizeBeat(c: Record<string, unknown>): string {
  const phase = String(c.phase ?? "");
  if (c.file) return `${phase}: ${String(c.file)}`;
  if (c.tool) return `${phase}: ${String(c.tool)}`;
  if (c.decision) return `${phase}: chose ${String(c.decision)}`;
  if (typeof c.file_count === "number") return `${phase}: ${c.file_count} files`;
  if (c.reason) return `${phase}: ${String(c.reason)}`;
  if (c.ok !== undefined) return `${phase}: ${c.ok ? "ok" : "failed"}`;
  return phase || "beat";
}

function FactsTab({ facts }: { facts: FactDTO[] }) {
  if (facts.length === 0) return <Empty>No facts surfaced yet.</Empty>;
  const sorted = [...facts].sort((a, b) => b.updated_at - a.updated_at);
  return (
    <ul className="space-y-1.5">
      {sorted.map((f) => {
        const isBeat = f.kind === "swarm_beat";
        const role = isBeat ? String(f.content?.role ?? "") : "";
        const accent = SWARM_ROLE_ACCENT[role] ?? "text-slate-400";
        return (
          <li
            key={f.fact_id}
            className={cn(
              "rounded border px-2 py-1.5",
              f.stale
                ? "border-amber-500/20 bg-amber-950/10"
                : "border-white/5 bg-slate-950/60",
            )}
          >
            <p className={cn(
              "text-[10px] font-mono uppercase tracking-wider",
              isBeat ? accent : "text-slate-400",
            )}>
              {isBeat ? `${role} · ${String(f.content?.phase ?? "")}` : f.kind}
              {f.stale && " · stale"}
              {f.kind === "llm_call" && typeof f.content?.latency_ms === "number"
                && ` · ${Math.round(f.content.latency_ms)}ms`}
            </p>
            <p className="text-[11px] text-slate-200 truncate">
              {isBeat
                ? summarizeBeat(f.content ?? {})
                : f.kind === "llm_call"
                  ? String(f.content?.model || "model?")
                  : String(f.content?.title || f.content?.path || f.content?.symbol
                    || f.content?.chunk_id || f.fact_id)}
            </p>
          </li>
        );
      })}
    </ul>
  );
}

function DecisionsTab({ decisions }: { decisions: DecisionDTO[] }) {
  if (decisions.length === 0) return <Empty>No decisions recorded yet.</Empty>;
  const sorted = [...decisions].sort((a, b) => b.made_at - a.made_at);
  return (
    <ul className="space-y-1.5">
      {sorted.map((d) => (
        <li key={d.decision_id}
            className="rounded border border-white/5 bg-slate-950/60 px-2 py-1.5">
          <p className="text-[10px] font-mono text-emerald-300 uppercase tracking-wider">
            {d.kind} · {formatRelative(d.made_at)}
          </p>
          <p className="text-[11px] text-slate-300 truncate">
            {String(d.chosen?.title || d.chosen?.text
              || (d.chosen?.approved !== undefined ? `approved=${d.chosen.approved}` : "")
              || d.decision_id)}
          </p>
        </li>
      ))}
    </ul>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[11px] text-slate-500 italic font-mono px-1">{children}</p>
  );
}
