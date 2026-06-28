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
  if (artifact.kind === "build_report") return <BuildReportBody c={c} />;
  if (artifact.kind === "repair_plan") return <RepairPlanBody c={c} />;
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

// Tailwind palette per verify outcome token. Mirrors the backend's
// VERIFY_OUTCOMES tuple so each classification gets a stable colour.
const VERIFY_OUTCOME_META: Record<string, { tone: string; label: string }> = {
  passed:              { tone: "text-emerald-400", label: "tests passed" },
  assertions_failed:   { tone: "text-red-400",     label: "assertions failed" },
  collection_error:    { tone: "text-amber-400",   label: "collection error" },
  no_tests_collected:  { tone: "text-slate-400",   label: "no tests collected" },
  timeout:             { tone: "text-amber-400",   label: "timeout" },
  pytest_missing:      { tone: "text-amber-400",   label: "pytest missing" },
  skipped:             { tone: "text-slate-400",   label: "skipped" },
};

function VerifyBody({ c }: { c: any }) {
  const ran = Boolean(c.ran);
  const outcome = String(c.outcome || (ran ? (c.tests_passed ? "passed" : "assertions_failed") : "skipped"));
  const meta = VERIFY_OUTCOME_META[outcome] ?? { tone: "text-slate-400", label: outcome };
  return (
    <div className="space-y-2 text-[11px] font-mono">
      <p className="text-slate-300">
        <span className={meta.tone}>{meta.label}</span>
        {ran && <span className="text-slate-500"> · rc={c.returncode}</span>}
        {!ran && c.skipped_reason && <span className="text-slate-500"> · {String(c.skipped_reason)}</span>}
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

const BUILD_OUTCOME_META: Record<string, { tone: string; label: string }> = {
  succeeded: { tone: "text-emerald-400", label: "venv ready" },
  failed:    { tone: "text-red-400",     label: "install failures" },
  no_venv:   { tone: "text-amber-400",   label: "host interpreter (no venv)" },
  skipped:   { tone: "text-slate-400",   label: "skipped (non-python)" },
  partial:   { tone: "text-amber-400",   label: "partial" },
};

function BuildReportBody({ c }: { c: any }) {
  const outcome = String(c.outcome || "skipped");
  const meta = BUILD_OUTCOME_META[outcome] ?? { tone: "text-slate-400", label: outcome };
  const installed: string[] = c.installed_packages || [];
  const failed: string[] = c.failed_installs || [];
  const manifests: string[] = c.installed_from || [];
  return (
    <div className="space-y-2 text-[11px] font-mono">
      <p className="text-slate-300">
        <span className={meta.tone}>{meta.label}</span>
        <span className="text-slate-500"> · {String(c.project_type || "unknown")}</span>
        {c.venv_path && <span className="text-slate-500"> · {String(c.venv_path)}</span>}
      </p>
      {manifests.length > 0 && (
        <p className="text-slate-500">manifests: {manifests.join(", ")}</p>
      )}
      {installed.length > 0 && (
        <div>
          <p className="text-slate-400">preflight-installed ({installed.length}):</p>
          <ul className="pl-3 list-disc text-emerald-300">
            {installed.map((p) => <li key={p}>{p}</li>)}
          </ul>
        </div>
      )}
      {failed.length > 0 && (
        <div>
          <p className="text-slate-400">failed installs ({failed.length}):</p>
          <ul className="pl-3 list-disc text-red-300">
            {failed.map((p) => <li key={p}>{p}</li>)}
          </ul>
        </div>
      )}
      {Array.isArray(c.style_issues) && c.style_issues.length > 0 && (
        <div>
          <p className="text-amber-300">test-style issues ({c.style_issues.length}):</p>
          <ul className="pl-3 list-disc text-amber-200/90">
            {c.style_issues.map((i: any, idx: number) => (
              <li key={idx}>
                <span className="text-slate-300">{String(i.file)}</span>
                <span className="text-slate-500">:{String(i.lineno)}</span>{" "}
                <span className="text-slate-400">{String(i.class_name)}</span>
                {i.kind && <span className="text-slate-500"> · {String(i.kind)}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {c.note && (
        <p className="text-slate-500 italic">{String(c.note)}</p>
      )}
      {c.pip_log_tail && (
        <pre className="bg-slate-950 border border-white/5 rounded p-2 text-slate-300 max-h-40 overflow-y-auto whitespace-pre-wrap">
          {String(c.pip_log_tail).slice(-2000)}
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


const REPAIR_CLASSIFICATION_META: Record<string, { tone: string; label: string }> = {
  unittest_pytest_mix:       { tone: "text-amber-400", label: "unittest/pytest mix" },
  missing_module_pythonpath: { tone: "text-sky-400",   label: "missing module · pythonpath" },
  missing_fixture:           { tone: "text-emerald-400", label: "missing fixture · hoist" },
  unknown:                   { tone: "text-slate-400", label: "unclassified" },
};

function RepairPlanBody({ c }: { c: any }) {
  const classification = String(c.classification || "unknown");
  const meta = REPAIR_CLASSIFICATION_META[classification]
    ?? { tone: "text-slate-400", label: classification };
  const diffs: { file: string; patch: string }[] = c.diffs || [];
  const locations: any[] = c.locations || [];
  const attempt = Number(c.repair_attempt || 0);
  const rationale = String(c.rationale || "");
  return (
    <div className="space-y-2 text-[11px] font-mono">
      <p className="text-slate-300">
        <span className={meta.tone}>{meta.label}</span>
        {attempt > 0 && <span className="text-slate-500"> · attempt {attempt}</span>}
        <span className="text-slate-500"> · {diffs.length} diff{diffs.length === 1 ? "" : "s"}</span>
      </p>
      {rationale && (
        <p className="text-slate-400 italic whitespace-pre-wrap">{rationale}</p>
      )}
      {locations.length > 0 && (
        <ul className="space-y-0.5 pl-3 text-slate-300 list-disc">
          {locations.map((loc, i) => {
            const label = loc.class_name ?? loc.fixture_name ?? loc.module_name ?? null;
            return (
              <li key={i}>
                <span className="text-slate-100">{String(loc.file)}</span>
                {label && <span className="text-slate-500"> · {String(label)}</span>}
                {loc.lineno != null && <span className="text-slate-500"> · L{Number(loc.lineno)}</span>}
                {loc.target && <span className="text-slate-500"> → {String(loc.target)}</span>}
                {Array.isArray(loc.helpers) && loc.helpers.length > 0 && (
                  <span className="text-slate-500"> · {loc.helpers.slice(0, 4).join(", ")}{loc.helpers.length > 4 ? "..." : ""}</span>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {diffs.length > 0 && (
        <DiffView diff={diffs.map((d) => d.patch).join("\n")} />
      )}
    </div>
  );
}
