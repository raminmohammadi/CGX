import { useMemo, useState } from "react";
import { Brain, Check, ChevronDown, ChevronRight, Loader2, Activity, CheckCircle2, XCircle, Globe, Wrench } from "lucide-react";
import type {
  ArtifactDTO, DecisionDTO, FactDTO, TaskNodeDTO,
} from "../../lib/api";
import { Pill } from "../Pill";
import { ErrorBoundary } from "../ErrorBoundary";
import { AskUserForm } from "./AskUserForm";
import { ArtifactPreview } from "./ArtifactPreview";

export { AskUserForm, ArtifactPreview };

export interface ActiveTaskProps {
  task: TaskNodeDTO | null;
  artifacts: ArtifactDTO[];
  decisions: DecisionDTO[];
  facts?: FactDTO[];
  onDecide: (payload: { chosen: Record<string, any>; rationale?: string }) => Promise<void> | void;
  pending: boolean;
}

export function ActiveTaskPanel(props: ActiveTaskProps) {
  const { task, artifacts, decisions, facts, onDecide, pending } = props;
  const llmFacts = useMemo(
    () => (task && facts
      ? facts.filter(
          (f) => f.kind === "llm_call" && f.surfaced_in_task_id === task.task_id,
        )
      : []),
    [task, facts],
  );
  const swarmBeats = useMemo(
    () => (task && facts
      ? facts.filter(
          (f) => f.kind === "swarm_beat" && f.surfaced_in_task_id === task.task_id,
        )
      : []),
    [task, facts],
  );
  // Resolve the linked artifact once: every ASK_USER input carries the
  // upstream artifact id under a kind-specific key. Hook runs every
  // render so it must stay above any early return.
  const linked = useMemo(
    () => (task ? resolveLinkedArtifact(task, artifacts) : null),
    [task, artifacts],
  );
  if (task === null) {
    return (
      <div className="text-[12px] text-slate-500 font-mono italic px-4 py-8 text-center">
        Pick a task from the tree to inspect it.
      </div>
    );
  }
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
      {llmFacts.length > 0 && (
        <ErrorBoundary label="llm-traces">
          <LLMTraces facts={llmFacts} />
        </ErrorBoundary>
      )}
      {swarmBeats.length > 0 && (
        <ErrorBoundary label="swarm-beats">
          <SwarmBeats beats={swarmBeats} />
        </ErrorBoundary>
      )}
    </div>
  );
}

function LLMTraces({ facts }: { facts: FactDTO[] }) {
  const sorted = useMemo(
    () => [...facts].sort((a, b) => a.created_at - b.created_at),
    [facts],
  );
  return (
    <details className="rounded-lg border border-white/5 bg-slate-950/40">
      <summary className="cursor-pointer select-none px-3 py-2 text-[10px] font-mono uppercase tracking-wider text-slate-400 hover:text-slate-200">
        LLM calls <span className="ml-1 text-slate-500">({sorted.length})</span>
      </summary>
      <ul className="px-3 pb-3 pt-1 space-y-1.5">
        {sorted.map((f, i) => (
          <LLMTraceRow key={f.fact_id} fact={f} index={i + 1} />
        ))}
      </ul>
    </details>
  );
}

function LLMTraceRow({ fact, index }: { fact: FactDTO; index: number }) {
  const [open, setOpen] = useState(false);
  const c = fact.content || {};
  const latency = typeof c.latency_ms === "number" ? `${c.latency_ms.toFixed(0)}ms` : null;
  const model = typeof c.model === "string" ? c.model : "model?";
  const samp = c.sampling || {};
  return (
    <li className="rounded border border-white/5 bg-slate-950/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-2 py-1.5 text-left hover:bg-white/5"
      >
        {open ? <ChevronDown className="h-3 w-3 text-slate-500" />
              : <ChevronRight className="h-3 w-3 text-slate-500" />}
        <Brain className="h-3 w-3 text-purple-400 shrink-0" />
        <span className="text-[10px] font-mono text-slate-500">#{index}</span>
        <span className="text-[11px] font-mono text-slate-200 truncate">{model}</span>
        {latency && (
          <span className="text-[10px] font-mono text-emerald-400 ml-auto">{latency}</span>
        )}
        {c.streamed && (
          <span className="text-[9px] font-mono uppercase tracking-wider text-sky-400">stream</span>
        )}
        {c.error && (
          <span className="text-[9px] font-mono uppercase tracking-wider text-red-400">error</span>
        )}
      </button>
      {open && (
        <div className="border-t border-white/5 px-2 py-2 space-y-2 text-[11px] font-mono">
          <div className="text-slate-500 text-[10px] uppercase tracking-wider">
            sampling · temp={samp.temperature ?? "?"} · max_tokens={samp.max_tokens ?? "?"}
            {samp.force_json !== undefined && ` · force_json=${samp.force_json}`}
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
              prompt <span className="text-slate-600">({c.prompt_chars ?? 0} chars)</span>
            </div>
            <pre className="bg-slate-950/80 rounded p-2 text-slate-300 whitespace-pre-wrap break-words max-h-64 overflow-auto">
              {String(c.prompt || "")}
            </pre>
          </div>
          {c.error ? (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-red-400 mb-1">error</div>
              <pre className="bg-red-950/30 rounded p-2 text-red-300 whitespace-pre-wrap break-words">
                {String(c.error)}
              </pre>
            </div>
          ) : (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
                response <span className="text-slate-600">({c.response_chars ?? 0} chars)</span>
              </div>
              <pre className="bg-slate-950/80 rounded p-2 text-emerald-200 whitespace-pre-wrap break-words max-h-64 overflow-auto">
                {String(c.response || "")}
              </pre>
            </div>
          )}
        </div>
      )}
    </li>
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


function SwarmBeats({ beats }: { beats: FactDTO[] }) {
  const sorted = useMemo(
    () => [...beats].sort((a, b) => a.created_at - b.created_at),
    [beats],
  );
  return (
    <details className="rounded-xl border border-sky-500/20 bg-sky-950/10 shadow-lg overflow-hidden transition-all duration-300 open:bg-sky-950/20" open>
      <summary className="cursor-pointer select-none px-4 py-3 flex items-center gap-2 hover:bg-sky-900/30 transition-colors">
        <Activity className="h-4 w-4 text-sky-400" />
        <span className="text-[11px] font-semibold uppercase tracking-widest text-sky-300">
          Swarm Operations
        </span>
        <span className="ml-auto text-[10px] font-mono bg-sky-900/50 text-sky-300 px-2 py-0.5 rounded-full">
          {sorted.length} events
        </span>
      </summary>
      <div className="px-4 pb-4 pt-2">
        <div className="relative border-l-2 border-sky-900/50 ml-3 pl-4 space-y-4">
          {sorted.map((b, i) => (
            <SwarmBeatRow key={b.fact_id} beat={b} index={i + 1} />
          ))}
        </div>
      </div>
    </details>
  );
}

function SwarmBeatRow({ beat, index }: { beat: FactDTO; index: number }) {
  const [open, setOpen] = useState(false);
  const c = beat.content || {};
  
  const isTool = c.phase === "tool_call";
  const isSuccess = c.ok === true;
  const isFail = c.ok === false || !!c.error;
  
  const Icon = isTool ? (c.tool === "search_web" ? Globe : Wrench) : 
               isSuccess ? CheckCircle2 : 
               isFail ? XCircle : Activity;
               
  const iconColor = isFail ? "text-red-400 bg-red-950" :
                    isSuccess ? "text-emerald-400 bg-emerald-950" :
                    isTool ? "text-amber-400 bg-amber-950" :
                    "text-sky-400 bg-sky-950";

  return (
    <div className="relative group">
      {/* Timeline Dot */}
      <div className={`absolute -left-[25px] top-1.5 h-5 w-5 rounded-full border-2 border-slate-900 flex items-center justify-center ${iconColor} z-10 transition-transform group-hover:scale-110`}>
        <Icon className="h-2.5 w-2.5" />
      </div>
      
      <div className="rounded-lg border border-white/5 bg-slate-950/60 transition-colors hover:border-white/10">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="w-full flex flex-wrap items-center gap-2 px-3 py-2 text-left"
        >
          <span className="text-[10px] font-mono text-slate-500 uppercase">#{index}</span>
          <span className="text-[12px] font-medium text-slate-200">
            {c.role || "agent"} <span className="text-slate-500">·</span> {c.phase || "step"}
          </span>
          
          {c.tool && (
            <span className="text-[10px] font-mono bg-amber-950/50 text-amber-300 px-1.5 py-0.5 rounded ml-2 border border-amber-500/20">
              {c.tool}
            </span>
          )}
          
          {isFail && (
            <span className="text-[10px] font-mono text-red-400 ml-auto flex items-center gap-1">
              FAILED
            </span>
          )}
          
          {open ? <ChevronDown className="h-3 w-3 text-slate-500 ml-auto" />
                : <ChevronRight className="h-3 w-3 text-slate-500 ml-auto" />}
        </button>
        
        {open && (
          <div className="border-t border-white/5 px-3 py-3 space-y-3 text-[11px] font-mono bg-slate-950/40">
            {c.file && (
              <div className="flex items-center gap-2 text-slate-400">
                <span className="uppercase tracking-wider text-[10px]">Target</span>
                <span className="text-sky-300">{c.file}</span>
              </div>
            )}
            
            {c.args && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Arguments</div>
                <pre className="bg-slate-950/80 border border-slate-800 rounded p-2 text-amber-200 whitespace-pre-wrap break-words">
                  {c.args}
                </pre>
              </div>
            )}
            
            {c.error && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-red-400 mb-1">Error</div>
                <pre className="bg-red-950/30 border border-red-900/50 rounded p-2 text-red-300 whitespace-pre-wrap break-words">
                  {String(c.error)}
                </pre>
              </div>
            )}
            
            {!c.args && !c.error && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Payload</div>
                <pre className="bg-slate-950/80 border border-slate-800 rounded p-2 text-slate-300 whitespace-pre-wrap break-words max-h-48 overflow-auto">
                  {JSON.stringify(c, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
