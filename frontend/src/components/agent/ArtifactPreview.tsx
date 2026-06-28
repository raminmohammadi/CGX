import type { ArtifactDTO } from "../../lib/api";
import { Markdown } from "../Markdown";
import { DiffView } from "../DiffView";

// Render any session artifact in a kind-aware preview. Falls back to a
// JSON dump for kinds we don't yet have a dedicated view for so the user
// still sees something useful.
export function ArtifactPreview({ artifact }: { artifact: ArtifactDTO }) {
  return (
    <section className="rounded-lg border border-white/5 bg-slate-950/40 p-3 space-y-2">
      <header className="flex items-center justify-between gap-2 text-[10px] font-mono text-slate-400 uppercase tracking-wider">
        <span>artifact · {artifact.kind}</span>
        <span className="text-slate-600">{artifact.artifact_id.slice(0, 8)}</span>
      </header>
      <Body artifact={artifact} />
    </section>
  );
}

function Body({ artifact }: { artifact: ArtifactDTO }) {
  const c = artifact.content || {};
  if (artifact.kind === "directions_list") return <DirectionsBody c={c} />;
  if (artifact.kind === "findings_bundle") return <FindingsBody c={c} />;
  if (artifact.kind === "recommendation_list") return <RecommendationBody c={c} />;
  if (artifact.kind === "code_change_plan") return <PlanBody c={c} />;
  if (artifact.kind === "applied_changes") return <AppliedBody c={c} />;
  if (artifact.kind === "verify_report") return <VerifyBody c={c} />;
  if (artifact.kind === "requirements_sheet") return <RequirementsBody c={c} />;
  if (artifact.kind === "work_plan") return <WorkPlanBody c={c} />;
  if (artifact.kind === "scaffold_patches") return <ScaffoldPatchesBody c={c} />;
  return <pre className="text-[11px] font-mono text-slate-300 whitespace-pre-wrap break-words">{JSON.stringify(c, null, 2)}</pre>;
}

function DirectionsBody({ c }: { c: any }) {
  const options: any[] = c.options || [];
  return (
    <div className="space-y-2">
      {c.restatement && (
        <p className="text-[12px] text-slate-300 italic">{String(c.restatement)}</p>
      )}
      {options.map((o, i) => (
        <div key={i} className="rounded border border-white/5 bg-slate-950/60 px-2 py-1.5">
          <p className="text-[12px] text-slate-100">
            <span className="text-fuchsia-400 font-mono mr-1.5">{i + 1}.</span>
            {o.title || o.chunk_id}
          </p>
          {o.rationale && <p className="text-[11px] text-slate-400 mt-0.5">{o.rationale}</p>}
          {o.chunk_id && (
            <p className="text-[10px] font-mono text-slate-500 mt-0.5">{o.chunk_id}</p>
          )}
        </div>
      ))}
    </div>
  );
}

function FindingsBody({ c }: { c: any }) {
  const md = String(c.answer_md || "");
  return (
    <div className="space-y-2">
      {c.question && (
        <p className="text-[11px] text-slate-400 italic">Q: {String(c.question)}</p>
      )}
      {md && (
        <div className="rounded border border-white/5 bg-slate-950 p-3 max-h-96 overflow-y-auto">
          <Markdown text={md} />
        </div>
      )}
    </div>
  );
}

function RecommendationBody({ c }: { c: any }) {
  const recs: any[] = c.recommendations || [];
  return (
    <div className="space-y-2">
      {recs.map((r, i) => (
        <div key={r.id || i} className="rounded border border-white/5 bg-slate-950/60 px-2 py-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
              {r.kind}
            </span>
            <p className="text-[12px] text-slate-100 font-medium">{r.title}</p>
          </div>
          {r.rationale && <p className="text-[11px] text-slate-400 mt-0.5">{r.rationale}</p>}
        </div>
      ))}
    </div>
  );
}

function PlanBody({ c }: { c: any }) {
  const planMd = String(c.plan_md || "");
  const diffs: { file: string; patch: string }[] = c.diffs || [];
  return (
    <div className="space-y-2">
      {planMd && (
        <div className="rounded border border-white/5 bg-slate-950 p-3 max-h-72 overflow-y-auto">
          <Markdown text={planMd} />
        </div>
      )}
      {diffs.length > 0 && (
        <DiffView diff={diffs.map((d) => d.patch).join("\n")} />
      )}
    </div>
  );
}

function AppliedBody({ c }: { c: any }) {
  const applied: string[] = c.applied_files || [];
  const failed: string[] = c.failed_files || [];
  return (
    <div className="space-y-2 text-[11px] font-mono">
      <p className="text-slate-300">
        Applied: <span className="text-emerald-400">{applied.length}</span>
        {" · "}Failed: <span className="text-red-400">{failed.length}</span>
        {c.backup_dir && <span className="text-slate-500"> · backup: {c.backup_dir}</span>}
      </p>
      {applied.length > 0 && (
        <ul className="space-y-0.5 pl-3 text-slate-300 list-disc">
          {applied.map((f) => <li key={f}>{f}</li>)}
        </ul>
      )}
      {failed.length > 0 && (
        <ul className="space-y-0.5 pl-3 text-red-300 list-disc">
          {failed.map((f) => <li key={f}>{f}</li>)}
        </ul>
      )}
    </div>
  );
}

function VerifyBody({ c }: { c: any }) {
  const passed = Boolean(c.tests_passed);
  const ran = Boolean(c.ran);
  return (
    <div className="space-y-2 text-[11px] font-mono">
      <p className="text-slate-300">
        {ran
          ? (passed
              ? <span className="text-emerald-400">tests passed</span>
              : <span className="text-red-400">tests failed (rc={c.returncode})</span>)
          : <span className="text-slate-400">tests skipped: {String(c.skipped_reason || "n/a")}</span>}
        {Array.isArray(c.tests_selected) && c.tests_selected.length > 0 &&
          <span className="text-slate-500"> · {c.tests_selected.length} selected</span>}
      </p>
      {c.stdout && (
        <pre className="bg-slate-950 border border-white/5 rounded p-2 text-slate-300 max-h-60 overflow-y-auto whitespace-pre-wrap">
          {String(c.stdout).slice(-4000)}
        </pre>
      )}
    </div>
  );
}


function RequirementsBody({ c }: { c: any }) {
  const questions: any[] = c.questions || [];
  return (
    <div className="space-y-2">
      {c.goal && (
        <p className="text-[11px] text-slate-400 italic">Goal: {String(c.goal)}</p>
      )}
      <p className="text-[10px] font-mono text-slate-500">
        {questions.length} question{questions.length === 1 ? "" : "s"} · source: {String(c.source || "llm")}
      </p>
      <ul className="space-y-1.5">
        {questions.map((q, i) => (
          <li key={q.id || i} className="rounded border border-white/5 bg-slate-950/60 px-2 py-1.5">
            <p className="text-[12px] text-slate-100">
              <span className="text-emerald-400 font-mono mr-1.5">{i + 1}.</span>
              {String(q.prompt || q.id)}
            </p>
            {q.hint && <p className="text-[10px] font-mono text-slate-500 mt-0.5">{String(q.hint)}</p>}
            {Array.isArray(q.suggested) && q.suggested.length > 0 && (
              <p className="text-[10px] font-mono text-slate-500 mt-0.5">
                suggested: {q.suggested.join(" · ")}
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function WorkPlanBody({ c }: { c: any }) {
  const layers: any[] = c.layers || [];
  const planMd = String(c.plan_md || "");
  const totalFiles = layers.reduce(
    (n: number, l: any) => n + (Array.isArray(l?.files) ? l.files.length : 0), 0);
  return (
    <div className="space-y-2">
      <p className="text-[10px] font-mono text-slate-500">
        {layers.length} layer{layers.length === 1 ? "" : "s"} · {totalFiles} file{totalFiles === 1 ? "" : "s"}
      </p>
      {planMd && (
        <div className="rounded border border-white/5 bg-slate-950 p-3 max-h-72 overflow-y-auto">
          <Markdown text={planMd} />
        </div>
      )}
      <div className="space-y-2">
        {layers.map((layer, li) => (
          <div key={li} className="rounded border border-white/5 bg-slate-950/60 px-2 py-1.5">
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
    </div>
  );
}

function ScaffoldPatchesBody({ c }: { c: any }) {
  const generated: any[] = c.generated || [];
  const failed: any[] = c.failed || [];
  const diffs: { file: string; patch: string }[] = c.diffs || [];
  return (
    <div className="space-y-2 text-[11px] font-mono">
      <p className="text-slate-300">
        Generated: <span className="text-emerald-400">{generated.length}</span>
        {" · "}Failed: <span className="text-red-400">{failed.length}</span>
      </p>
      {generated.length > 0 && (
        <ul className="space-y-0.5 pl-3 text-slate-300 list-disc">
          {generated.map((g, i) => (
            <li key={i}>
              {String(g.file)}
              <span className="text-slate-500"> · {Number(g.bytes || 0)}B</span>
              {g.layer && <span className="text-slate-500"> · {String(g.layer)}</span>}
            </li>
          ))}
        </ul>
      )}
      {failed.length > 0 && (
        <ul className="space-y-0.5 pl-3 text-red-300 list-disc">
          {failed.map((f, i) => (
            <li key={i}>{String(f.file)} — {String(f.error)}</li>
          ))}
        </ul>
      )}
      {diffs.length > 0 && (
        <DiffView diff={diffs.map((d) => d.patch).join("\n")} />
      )}
    </div>
  );
}
