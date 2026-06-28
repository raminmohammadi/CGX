import { useState } from "react";
import { Check, Send, X } from "lucide-react";
import type { ArtifactDTO, TaskNodeDTO } from "../../lib/api";
import { TextArea } from "../Input";
import { cn } from "../../lib/utils";
import { ApproveForm } from "./ApproveForm";

type DecidePayload = { chosen: Record<string, any>; rationale?: string };

export function AskUserForm({
  task, linked, onDecide, pending,
}: {
  task: TaskNodeDTO;
  linked: ArtifactDTO | null;
  onDecide: (p: DecidePayload) => Promise<void> | void;
  pending: boolean;
}) {
  const expectedKind = String(task.inputs?.expected_kind || "freeform");
  // Dispatch on the contract carried in the task input so each variant
  // stays in its own component instead of one ballooning switch.
  if (expectedKind === "choose_path") {
    return <ChoosePathForm linked={linked} onDecide={onDecide} pending={pending} />;
  }
  if (expectedKind === "choose_recommendation") {
    return <ChooseRecommendationForm linked={linked} onDecide={onDecide} pending={pending} />;
  }
  if (expectedKind === "approve") {
    return <ApproveForm linked={linked} onDecide={onDecide} pending={pending} />;
  }
  if (expectedKind === "clarify_answers") {
    return <ClarifyAnswersForm linked={linked} onDecide={onDecide} pending={pending} />;
  }
  if (expectedKind === "approve_plan") {
    return <ApprovePlanForm linked={linked} onDecide={onDecide} pending={pending} />;
  }
  return <FreeformForm onDecide={onDecide} pending={pending} />;
}

function ChoosePathForm({
  linked, onDecide, pending,
}: { linked: ArtifactDTO | null; onDecide: (p: DecidePayload) => any; pending: boolean }) {
  const options: any[] = linked?.content?.options || [];
  return (
    <div className="rounded-lg border border-fuchsia-500/30 bg-fuchsia-950/15 p-4 space-y-3">
      <p className="text-[10px] uppercase tracking-wider font-mono text-fuchsia-300">
        Pick a direction to investigate
      </p>
      {options.length === 0 && (
        <p className="text-[11px] text-slate-400 italic">No options surfaced.</p>
      )}
      <div className="space-y-2">
        {options.map((o, i) => (
          <button
            key={String(o.chunk_id || i)}
            type="button"
            disabled={pending}
            onClick={() => onDecide({
              chosen: { anchor_chunk_id: String(o.chunk_id || ""), title: String(o.title || "") },
            })}
            className={cn(
              "w-full text-left rounded-md border border-white/10 bg-slate-950/50",
              "px-3 py-2 hover:bg-slate-900 disabled:opacity-40 disabled:cursor-not-allowed",
            )}
          >
            <p className="text-[12px] text-slate-100 font-medium">
              <span className="text-fuchsia-400 font-mono mr-1.5">{i + 1}.</span>
              {o.title || o.chunk_id || "option"}
            </p>
            {o.rationale && (
              <p className="text-[11px] text-slate-400 mt-0.5">{o.rationale}</p>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function ChooseRecommendationForm({
  linked, onDecide, pending,
}: { linked: ArtifactDTO | null; onDecide: (p: DecidePayload) => any; pending: boolean }) {
  const recs: any[] = linked?.content?.recommendations || [];
  const kindBadge: Record<string, string> = {
    investigate_more: "bg-indigo-900/80 text-indigo-300",
    plan_change:      "bg-orange-900/80 text-orange-300",
    ask_followup:     "bg-fuchsia-900/80 text-fuchsia-300",
    done:             "bg-slate-700 text-slate-300",
  };
  return (
    <div className="rounded-lg border border-purple-500/30 bg-purple-950/15 p-4 space-y-3">
      <p className="text-[10px] uppercase tracking-wider font-mono text-purple-300">
        Choose a recommendation
      </p>
      {recs.length === 0 && (
        <p className="text-[11px] text-slate-400 italic">No recommendations available.</p>
      )}
      <div className="space-y-2">
        {recs.map((r, i) => (
          <button
            key={String(r.id || i)}
            type="button"
            disabled={pending}
            onClick={() => onDecide({
              chosen: {
                id: String(r.id || `r${i + 1}`),
                title: String(r.title || ""),
                rationale: String(r.rationale || ""),
                kind: String(r.kind || "done"),
                ...(r.anchor_chunk_id ? { anchor_chunk_id: String(r.anchor_chunk_id) } : {}),
              },
            })}
            className="w-full text-left rounded-md border border-white/10 bg-slate-950/50 px-3 py-2 hover:bg-slate-900 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <div className="flex items-center gap-2 mb-0.5">
              <span className={cn(
                "text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded",
                kindBadge[String(r.kind)] || "bg-slate-700 text-slate-300",
              )}>{r.kind}</span>
              <p className="text-[12px] text-slate-100 font-medium">
                {r.title || r.id || "recommendation"}
              </p>
            </div>
            {r.rationale && (
              <p className="text-[11px] text-slate-400">{r.rationale}</p>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function FreeformForm({
  onDecide, pending,
}: { onDecide: (p: DecidePayload) => any; pending: boolean }) {
  const [text, setText] = useState("");
  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/40 p-4 space-y-3">
      <p className="text-[10px] uppercase tracking-wider font-mono text-slate-400">
        Reply
      </p>
      <TextArea
        rows={3}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type your answer…"
        disabled={pending}
      />
      <div className="flex justify-end">
        <button
          type="button"
          disabled={pending || !text.trim()}
          onClick={() => onDecide({ chosen: { text: text.trim() } })}
          className="av-btn-primary"
        >
          <Send className="h-3 w-3" /> Send
        </button>
      </div>
    </div>
  );
}


type Question = {
  id: string;
  prompt: string;
  hint?: string;
  suggested?: string[];
};

function ClarifyAnswersForm({
  linked, onDecide, pending,
}: { linked: ArtifactDTO | null; onDecide: (p: DecidePayload) => any; pending: boolean }) {
  const questions: Question[] = (linked?.content?.questions || []) as Question[];
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const update = (id: string, v: string) => setAnswers((a) => ({ ...a, [id]: v }));
  const filled = questions.filter((q) => (answers[q.id] || "").trim().length > 0).length;
  const canSubmit = !pending && filled >= Math.min(questions.length, 3);
  return (
    <div className="rounded-lg border border-emerald-500/30 bg-emerald-950/10 p-4 space-y-3">
      <p className="text-[10px] uppercase tracking-wider font-mono text-emerald-300">
        Clarify project requirements
      </p>
      {questions.length === 0 && (
        <p className="text-[11px] text-slate-400 italic">No questions surfaced.</p>
      )}
      <div className="space-y-3">
        {questions.map((q, i) => (
          <div key={q.id || i} className="space-y-1.5">
            <label className="block text-[12px] text-slate-100">
              <span className="text-emerald-400 font-mono mr-1.5">{i + 1}.</span>
              {q.prompt}
            </label>
            {q.hint && (
              <p className="text-[10px] font-mono text-slate-500">{q.hint}</p>
            )}
            <TextArea
              rows={2}
              value={answers[q.id] || ""}
              onChange={(e) => update(q.id, e.target.value)}
              placeholder={q.suggested && q.suggested.length > 0
                ? `e.g. ${q.suggested[0]}` : "Your answer…"}
              disabled={pending}
            />
            {q.suggested && q.suggested.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {q.suggested.map((s) => (
                  <button
                    key={s} type="button" disabled={pending}
                    onClick={() => update(q.id, s)}
                    className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-white/10 bg-slate-950/50 text-slate-300 hover:border-emerald-500/30"
                  >{s}</button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="flex items-center justify-between gap-2 pt-1">
        <p className="text-[10px] font-mono text-slate-500">
          {filled}/{questions.length} answered
        </p>
        <button
          type="button"
          disabled={!canSubmit}
          onClick={() => onDecide({ chosen: { answers } })}
          className="av-btn-primary"
        >
          <Send className="h-3 w-3" /> Submit answers
        </button>
      </div>
    </div>
  );
}

function ApprovePlanForm({
  linked, onDecide, pending,
}: { linked: ArtifactDTO | null; onDecide: (p: DecidePayload) => any; pending: boolean }) {
  const layers: any[] = linked?.content?.layers || [];
  const planMd = String(linked?.content?.plan_md || "");
  const totalFiles = layers.reduce(
    (n, l) => n + (Array.isArray(l?.files) ? l.files.length : 0), 0);
  return (
    <div className="rounded-lg border border-orange-500/30 bg-orange-950/10 p-4 space-y-3">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <p className="text-[10px] uppercase tracking-wider font-mono text-orange-300">
          Approve work plan
        </p>
        <p className="text-[10px] font-mono text-slate-400">
          {layers.length} layer{layers.length === 1 ? "" : "s"} · {totalFiles} file{totalFiles === 1 ? "" : "s"}
        </p>
      </div>
      <div className="rounded-lg border border-white/5 bg-slate-950 p-3 max-h-72 overflow-y-auto space-y-3">
        {layers.length === 0 && !planMd && (
          <p className="text-[11px] text-slate-400 italic">Work plan is empty.</p>
        )}
        {layers.length === 0 && planMd && (
          <pre className="text-[11px] font-mono text-slate-300 whitespace-pre-wrap break-words">{planMd}</pre>
        )}
        {layers.map((layer, li) => (
          <div key={li}>
            <p className="text-[11px] font-mono text-fuchsia-300">{String(layer.name || "layer")}</p>
            <ul className="mt-1 space-y-0.5 pl-3">
              {(layer.files || []).map((f: any, fi: number) => (
                <li key={fi} className="text-[11px] text-slate-300">
                  <span className="font-mono text-slate-100">{String(f.path)}</span>
                  {f.description && <span className="text-slate-500"> — {String(f.description)}</span>}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-2 justify-end pt-1">
        <button
          type="button" disabled={pending}
          onClick={() => onDecide({ chosen: { approved: false } })}
          className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/5 px-3 py-1.5 text-[11px] font-mono text-slate-200 hover:bg-white/10 disabled:opacity-40"
        ><X className="h-3 w-3" /> Reject</button>
        <button
          type="button" disabled={pending}
          onClick={() => onDecide({ chosen: { approved: true } })}
          className="av-btn-primary"
        ><Check className="h-3 w-3" /> Approve &amp; Scaffold</button>
      </div>
    </div>
  );
}
