import {
  CheckCircle2, ClipboardList, FileEdit, FlaskConical, GitBranch,
  HelpCircle, ListChecks, Package, Search, TerminalSquare, Wrench, XCircle,
} from "lucide-react";
import type { ArtifactDTO, ArtifactKind } from "../../lib/api";
import { Markdown } from "../Markdown";
import { DiffView } from "../DiffView";
import { ErrorBoundary } from "../ErrorBoundary";

// Icon per artifact kind so each card reads at a glance as "tool call: X",
// mirroring how tool invocations are shown in richer agent-chat UIs.
const ARTIFACT_ICONS: Partial<Record<ArtifactKind, typeof HelpCircle>> = {
  directions_list: GitBranch,
  findings_bundle: Search,
  recommendation_list: ListChecks,
  code_change_plan: FileEdit,
  applied_changes: FileEdit,
  verify_report: FlaskConical,
  requirements_sheet: ClipboardList,
  work_plan: ClipboardList,
  scaffold_patches: FileEdit,
  build_report: Package,
  repair_plan: Wrench,
  smoke_report: TerminalSquare,
  api_check_report: TerminalSquare,
};

// Best-effort ok/fail read per kind, purely for a small header status icon --
// the detailed outcome is still rendered by each kind's own Body below.
function artifactStatus(artifact: ArtifactDTO): "ok" | "fail" | null {
  const c = artifact.content || {};
  switch (artifact.kind) {
    case "verify_report":
      return c.ran ? (c.tests_passed ? "ok" : "fail") : null;
    case "build_report":
      if (c.outcome === "succeeded") return "ok";
      if (c.outcome === "failed") return "fail";
      return null;
    case "applied_changes":
      return Array.isArray(c.failed_files) && c.failed_files.length > 0 ? "fail" : "ok";
    case "scaffold_patches":
      return Array.isArray(c.failed) && c.failed.length > 0 ? "fail" : "ok";
    case "smoke_report":
    case "api_check_report":
      if (c.outcome === "passed") return "ok";
      if (c.outcome === "failed") return "fail";
      return null;
    default:
      return null;
  }
}

// Render any session artifact in a kind-aware preview. Falls back to a
// JSON dump for kinds we don't yet have a dedicated view for so the user
// still sees something useful.
export function ArtifactPreview({ artifact }: { artifact: ArtifactDTO }) {
  const Icon = ARTIFACT_ICONS[artifact.kind] || HelpCircle;
  const status = artifactStatus(artifact);
  return (
    <section className="rounded-lg border border-white/5 bg-slate-950/40 p-3 space-y-2">
      <header className="flex items-center gap-2 text-[10px] font-mono text-slate-400 uppercase tracking-wider">
        <Icon className="h-3 w-3 text-emerald-400 shrink-0" />
        <span>{artifact.kind}</span>
        {status === "ok" && <CheckCircle2 className="h-3 w-3 text-emerald-400" />}
        {status === "fail" && <XCircle className="h-3 w-3 text-red-400" />}
        <span className="text-slate-600 ml-auto">{artifact.artifact_id.slice(0, 8)}</span>
      </header>
      <ErrorBoundary label={`artifact ${artifact.kind}`}>
        <Body artifact={artifact} />
      </ErrorBoundary>
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
  if (artifact.kind === "smoke_report") return <SmokeReportBody c={c} />;
  if (artifact.kind === "api_check_report") return <ApiCheckReportBody c={c} />;
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
  const applied: string[] = Array.isArray(c.applied_files) ? c.applied_files : [];
  // ``disk_apply`` emits failed_files as ``[{file, error}, ...]``; legacy
  // payloads occasionally still carry plain ``string[]``. Normalise both
  // shapes so rendering can't trip React error #31.
  const failedRaw: any[] = Array.isArray(c.failed_files) ? c.failed_files : [];
  const failed: { file: string; error: string }[] = failedRaw.map((f) =>
    typeof f === "string"
      ? { file: f, error: "" }
      : { file: String(f?.file ?? ""), error: String(f?.error ?? "") },
  );
  return (
    <div className="space-y-2 text-[11px] font-mono">
      <p className="text-slate-300">
        Applied: <span className="text-emerald-400">{applied.length}</span>
        {" · "}Failed: <span className="text-red-400">{failed.length}</span>
        {c.backup_dir && <span className="text-slate-500"> · backup: {String(c.backup_dir)}</span>}
      </p>
      {applied.length > 0 && (
        <ul className="space-y-0.5 pl-3 text-slate-300 list-disc">
          {applied.map((f) => <li key={f}>{String(f)}</li>)}
        </ul>
      )}
      {failed.length > 0 && (
        <ul className="space-y-0.5 pl-3 text-red-300 list-disc">
          {failed.map((f, i) => (
            <li key={i}>
              {f.file}
              {f.error && <span className="text-slate-500"> — {f.error}</span>}
            </li>
          ))}
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
      {c.reproduce_cmd && (
        <div>
          <p className="text-slate-500">reproduce:</p>
          <pre className="bg-slate-950 border border-white/5 rounded p-2 text-emerald-300 overflow-x-auto whitespace-pre">
            {String(c.reproduce_cmd)}
          </pre>
        </div>
      )}
      {Array.isArray(c.failures) && c.failures.length > 0 && (
        <div className="space-y-1">
          <p className="text-slate-500">{c.failures.length} failure{c.failures.length === 1 ? "" : "s"}:</p>
          {c.failures.slice(0, 8).map((f: any, i: number) => (
            <details key={i} className="bg-slate-950 border border-red-900/40 rounded p-2">
              <summary className="cursor-pointer text-red-300">
                <span className="text-slate-500">{String(f.kind || "failure")}</span>
                {" · "}
                <span>{String(f.nodeid || "")}</span>
                {f.type && <span className="text-slate-500"> · {String(f.type)}</span>}
              </summary>
              {f.message && (
                <p className="text-slate-300 mt-1 whitespace-pre-wrap break-words">{String(f.message).slice(0, 600)}</p>
              )}
              {f.traceback && (
                <pre className="mt-1 text-slate-400 whitespace-pre-wrap break-words max-h-48 overflow-y-auto">
                  {String(f.traceback).slice(-2000)}
                </pre>
              )}
            </details>
          ))}
        </div>
      )}
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
  const resolved: { name: string; version: string }[] = c.resolved_packages || [];
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
      {resolved.length > 0 && (
        <details className="text-slate-400">
          <summary className="cursor-pointer text-slate-400">
            resolved: {resolved.length} package{resolved.length === 1 ? "" : "s"}
          </summary>
          <ul className="pl-3 list-disc text-slate-300 max-h-40 overflow-y-auto">
            {resolved.map((p) => (
              <li key={p.name}>
                <span>{p.name}</span>
                <span className="text-slate-500">=={p.version}</span>
              </li>
            ))}
          </ul>
        </details>
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


function SmokeReportBody({ c }: { c: any }) {
  const outcome = String(c.outcome || "skipped");
  const modules: any[] = Array.isArray(c.modules) ? c.modules : [];
  const failed = modules.filter((m) => !m?.ok);
  const passed = modules.filter((m) => !!m?.ok);
  const outcomeColor =
    outcome === "passed" ? "text-emerald-300"
      : outcome === "failed" ? "text-red-300"
      : "text-slate-400";
  return (
    <div className="text-[11px] font-mono space-y-1.5">
      <p className={outcomeColor}>
        outcome: {outcome} · tested {modules.length} module{modules.length === 1 ? "" : "s"}
      </p>
      {failed.length > 0 && (
        <div>
          <p className="text-red-300">failed imports ({failed.length}):</p>
          <ul className="pl-3 list-disc text-red-200/90 space-y-1">
            {failed.map((m: any, i: number) => (
              <li key={i}>
                <span className="text-slate-100">{String(m.name)}</span>
                {m.stderr_tail && (
                  <pre className="bg-slate-950 border border-white/5 rounded p-1.5 mt-1 text-slate-300 max-h-32 overflow-y-auto whitespace-pre-wrap">
                    {String(m.stderr_tail).slice(-1200)}
                  </pre>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      {passed.length > 0 && (
        <details className="text-slate-400">
          <summary className="cursor-pointer">
            passed imports ({passed.length})
          </summary>
          <ul className="pl-3 list-disc text-emerald-300">
            {passed.map((m: any, i: number) => (<li key={i}>{String(m.name)}</li>))}
          </ul>
        </details>
      )}
      {outcome === "skipped" && (
        <p className="text-slate-500 italic">
          No third-party imports detected in the applied files (or no
          python interpreter available); SMOKE was a no-op.
        </p>
      )}
    </div>
  );
}


function ApiCheckReportBody({ c }: { c: any }) {
  const outcome = String(c.outcome || "skipped");
  const refs: any[] = Array.isArray(c.references) ? c.references : [];
  const failed = refs.filter((r) => !r?.ok);
  const passed = refs.filter((r) => !!r?.ok);
  const probeError = c.probe_error ? String(c.probe_error) : "";
  const outcomeColor =
    outcome === "passed" ? "text-emerald-300"
      : outcome === "failed" ? "text-red-300"
      : "text-slate-400";
  return (
    <div className="text-[11px] font-mono space-y-1.5">
      <p className={outcomeColor}>
        outcome: {outcome} · checked {refs.length} reference{refs.length === 1 ? "" : "s"}
      </p>
      {probeError && (
        <p className="text-amber-300">probe_error: {probeError}</p>
      )}
      {failed.length > 0 && (
        <div>
          <p className="text-red-300">unresolved references ({failed.length}):</p>
          <ul className="pl-3 list-disc text-red-200/90 space-y-1">
            {failed.map((r: any, i: number) => (
              <li key={i}>
                <span className="text-slate-100">
                  {String(r.module)}.{String(r.name)}
                </span>
                {r.error && (
                  <p className="text-slate-400 mt-0.5">{String(r.error)}</p>
                )}
                {Array.isArray(r.references) && r.references.length > 0 && (
                  <p className="text-slate-500 text-[10px] mt-0.5">
                    referenced from {r.references.map((ref: any) =>
                      `${ref.file}:${ref.lineno}`).join(", ")}
                  </p>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      {passed.length > 0 && (
        <details className="text-slate-400">
          <summary className="cursor-pointer">
            resolved references ({passed.length})
          </summary>
          <ul className="pl-3 list-disc text-emerald-300">
            {passed.map((r: any, i: number) => (
              <li key={i}>{String(r.module)}.{String(r.name)}</li>
            ))}
          </ul>
        </details>
      )}
      {outcome === "skipped" && !probeError && (
        <p className="text-slate-500 italic">
          No third-party API references detected in the applied files (or
          no python interpreter available); API_CHECK was a no-op.
        </p>
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
  const pinAdjustments: any[] = c.pin_adjustments || [];
  return (
    <div className="space-y-2 text-[11px] font-mono">
      <p className="text-slate-300">
        Generated: <span className="text-emerald-400">{generated.length}</span>
        {" · "}Failed: <span className="text-red-400">{failed.length}</span>
        {pinAdjustments.length > 0 && (
          <>{" · "}Pin adjustments: <span className="text-amber-400">{pinAdjustments.length}</span></>
        )}
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
      {pinAdjustments.length > 0 && (
        <details className="text-slate-300">
          <summary className="cursor-pointer text-amber-300">
            Tightened {pinAdjustments.length} fragile peer pin{pinAdjustments.length === 1 ? "" : "s"}
          </summary>
          <ul className="space-y-0.5 pl-3 list-disc">
            {pinAdjustments.map((a, i) => (
              <li key={i}>
                <span className="text-slate-400">{String(a.file || "requirements.txt")}</span>
                {" · "}
                <span className="text-amber-200">{String(a.peer)}</span>
                {a.before && (
                  <>
                    {" "}<span className="text-slate-500 line-through">{String(a.before)}</span>
                  </>
                )}
                {" -> "}
                <span className="text-emerald-300">{String(a.after)}</span>
                <span className="text-slate-500"> (from {String(a.consumer)}=={String(a.consumer_version)})</span>
              </li>
            ))}
          </ul>
        </details>
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
