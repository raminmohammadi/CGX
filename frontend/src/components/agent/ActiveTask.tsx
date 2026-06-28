import { useMemo } from "react";
import { Check, Loader2 } from "lucide-react";
import type {
  ArtifactDTO, DecisionDTO, TaskNodeDTO,
} from "../../lib/api";
import { Pill } from "../Pill";
import { AskUserForm } from "./AskUserForm";
import { ArtifactPreview } from "./ArtifactPreview";

export { AskUserForm, ArtifactPreview };

export interface ActiveTaskProps {
  task: TaskNodeDTO | null;
  artifacts: ArtifactDTO[];
  decisions: DecisionDTO[];
  onDecide: (payload: { chosen: Record<string, any>; rationale?: string }) => Promise<void> | void;
  pending: boolean;
}

export function ActiveTaskPanel(props: ActiveTaskProps) {
  const { task, artifacts, decisions, onDecide, pending } = props;
  if (task === null) {
    return (
      <div className="text-[12px] text-slate-500 font-mono italic px-4 py-8 text-center">
        Pick a task from the tree to inspect it.
      </div>
    );
  }
  // Resolve the linked artifact once: every ASK_USER input carries the
  // upstream artifact id under a kind-specific key.
  const linked = useMemo(
    () => resolveLinkedArtifact(task, artifacts),
    [task, artifacts],
  );
  const resolvedDecision = decisions.find(
    (d) => d.resolved_task_id === task.task_id,
  );
  return (
    <div className="space-y-4">
      <TaskHeader task={task} />
      {task.error && (
        <div className="rounded-lg border border-red-500/30 bg-red-950/30 p-3 text-[11px] font-mono text-red-300 whitespace-pre-wrap">
          {task.error}
        </div>
      )}
      {task.kind === "ask_user" && task.status === "in_progress" && !resolvedDecision && (
        <AskUserForm
          task={task} linked={linked}
          onDecide={onDecide} pending={pending}
        />
      )}
      {resolvedDecision && (
        <DecisionSummary decision={resolvedDecision} />
      )}
      {linked && (task.kind !== "ask_user" || resolvedDecision !== undefined) && (
        <ArtifactPreview artifact={linked} />
      )}
    </div>
  );
}

function TaskHeader({ task }: { task: TaskNodeDTO }) {
  const tone =
    task.status === "done" ? "neon"
    : task.status === "failed" ? "red"
    : task.status === "in_progress" ? "amber"
    : task.status === "ready" ? "purple" : "slate";
  return (
    <div>
      <div className="flex items-center gap-2 mb-1 flex-wrap">
        <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500">
          {task.kind}
        </span>
        <Pill tone={tone as any}>{task.status}</Pill>
        {task.status === "in_progress" && task.kind !== "ask_user" && (
          <Loader2 className="h-3 w-3 text-amber-400 animate-spin" />
        )}
      </div>
      <h2 className="text-sm font-semibold text-slate-100">{task.name}</h2>
      {task.description && task.description !== task.name && (
        <p className="text-[12px] text-slate-400 mt-1">{task.description}</p>
      )}
    </div>
  );
}

function DecisionSummary({ decision }: { decision: DecisionDTO }) {
  return (
    <div className="rounded-lg border border-emerald-500/20 bg-emerald-950/20 p-3 space-y-1">
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider font-mono text-emerald-300">
        <Check className="h-3 w-3" /> decision recorded · {decision.kind}
      </div>
      <pre className="text-[11px] font-mono text-slate-300 whitespace-pre-wrap break-words">
        {JSON.stringify(decision.chosen, null, 2)}
      </pre>
      {decision.rationale && (
        <p className="text-[11px] text-slate-400 italic">{decision.rationale}</p>
      )}
    </div>
  );
}

function resolveLinkedArtifact(task: TaskNodeDTO, artifacts: ArtifactDTO[]): ArtifactDTO | null {
  const direct = task.produced_artifact_id
    ? artifacts.find((a) => a.artifact_id === task.produced_artifact_id)
    : null;
  if (direct) return direct;
  // Every ASK_USER stores its upstream artifact under a kind-specific
  // ``*_artifact_id`` key in ``inputs`` (e.g. ``directions_artifact_id``
  // for choose_path, ``requirements_artifact_id`` for clarify_answers,
  // ``work_plan_artifact_id`` for approve_plan). Scan all matching
  // entries instead of hard-coding kinds so new flows pick up
  // automatically.
  const inputs = task.inputs || {};
  for (const k of Object.keys(inputs)) {
    if (!k.endsWith("_artifact_id")) continue;
    const id = inputs[k];
    if (typeof id === "string" && id) {
      const a = artifacts.find((x) => x.artifact_id === id);
      if (a) return a;
    }
  }
  return null;
}


