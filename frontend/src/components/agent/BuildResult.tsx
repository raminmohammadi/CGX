import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CheckCircle2, ChevronDown, ChevronRight, ExternalLink, FileDiff,
  Image as ImageIcon, MonitorPlay, Square, XCircle,
} from "lucide-react";
import { api, type AgentSessionState, type ArtifactDTO } from "../../lib/api";
import { cn, formatRelative } from "../../lib/utils";

// The review half of the feedback loop for a swarm build: after SWARM_VERIFY
// reports, this panel shows the verdict, one screenshot per route the browser
// smoke loaded, a live preview of the real production bundle, and the history
// of revisions (what the user asked -> which files changed, with the per-file
// instruction and diff). The user then types feedback in the composer below;
// SWARM_REVISE localizes it to the exact files and SWARM_VERIFY re-screenshots
// the result, so this panel updates in place as SSE frames land.

export interface BuildResultProps {
  sid: string;
  state: AgentSessionState;
}

interface RouteShot {
  route: string;
  mounted: boolean;
  errors: string[];
  screenshot: string | null;   // bare file name, served by the screenshots endpoint
  text_sample?: string;
}

// The newest SWARM_VERIFY_REPORT drives the verdict + screenshots.
export function latestVerifyReport(artifacts: ArtifactDTO[]): ArtifactDTO | null {
  const reports = artifacts.filter((a) => a.kind === "swarm_verify_report");
  if (reports.length === 0) return null;
  return reports.reduce((best, a) => (a.created_at > best.created_at ? a : best), reports[0]);
}

export function revisionArtifacts(artifacts: ArtifactDTO[]): ArtifactDTO[] {
  return artifacts
    .filter((a) => a.kind === "swarm_revision")
    .sort((a, b) => b.created_at - a.created_at);
}

function routeShots(report: ArtifactDTO | null): RouteShot[] {
  const runtime = report?.content?.runtime;
  const routes: any[] = Array.isArray(runtime?.routes) ? runtime.routes : [];
  return routes.map((r) => {
    const shot = typeof r.screenshot === "string" && r.screenshot
      ? r.screenshot.split(/[\\/]/).pop() || null
      : null;
    return {
      route: String(r.route ?? "/"),
      mounted: Boolean(r.mounted),
      errors: Array.isArray(r.errors) ? r.errors.map(String) : [],
      screenshot: shot,
      text_sample: typeof r.text_sample === "string" ? r.text_sample : undefined,
    };
  });
}

export function BuildResult({ sid, state }: BuildResultProps) {
  const report = useMemo(() => latestVerifyReport(state.artifacts), [state.artifacts]);
  const revisions = useMemo(() => revisionArtifacts(state.artifacts), [state.artifacts]);
  const shots = useMemo(() => routeShots(report), [report]);
  const [lightbox, setLightbox] = useState<RouteShot | null>(null);
  const [preview, setPreview] = useState<{ url: string } | null>(null);
  const [previewInline, setPreviewInline] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);

  // A re-verify after a revision rewrites the bundle in place; drop the stale
  // preview so the next click reflects the new build.
  useEffect(() => { setPreview(null); setPreviewInline(false); }, [report?.artifact_id]);

  const openPreview = useCallback(async (inline: boolean) => {
    setPreviewBusy(true); setPreviewError(null);
    try {
      const p = preview ?? await api.agentSessionPreviewStart(sid);
      setPreview({ url: p.url });
      if (inline) setPreviewInline(true);
      else window.open(p.url, "_blank", "noopener");
    } catch (e) {
      setPreviewError(String((e as Error)?.message || e));
    } finally { setPreviewBusy(false); }
  }, [preview, sid]);

  const stopPreview = useCallback(async () => {
    setPreviewInline(false);
    setPreview(null);
    try { await api.agentSessionPreviewStop(sid); } catch { /* best-effort */ }
  }, [sid]);

  if (!report && revisions.length === 0) return null;
  const c = report?.content ?? {};
  const ok = Boolean(c.verify_ok);
  const summary = String(c.summary || (ok ? "verified" : "partial build"));
  const running = state.tasks.some(
    (t) => t.kind !== "ask_user" && (t.status === "in_progress" || t.status === "ready"));
  const pendingRevision = state.tasks.find(
    (t) => t.kind === "swarm_revise" && (t.status === "in_progress" || t.status === "ready"));

  return (
    <section
      data-testid="build-result"
      className="mb-4 rounded-xl border border-emerald-500/20 bg-slate-950/50 overflow-hidden"
    >
      <header className="flex items-center gap-2 px-4 py-2.5 border-b border-white/5 bg-slate-950/60">
        <MonitorPlay className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
        <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-slate-400">Build result</span>
        {report && (
          <span className={cn(
            "ml-1 inline-flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded border",
            ok ? "border-emerald-500/30 bg-emerald-950/40 text-emerald-300"
               : "border-amber-500/30 bg-amber-950/40 text-amber-300",
          )}>
            {ok ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
            {ok ? "verified" : "partial build"}
          </span>
        )}
        {report && (
          <span className="text-[10px] font-mono text-slate-500 ml-auto">
            {formatRelative(report.created_at)}
          </span>
        )}
      </header>

      <div className="px-4 py-3 space-y-3">
        {report && (
          <p className="text-[12px] text-slate-300 font-mono leading-relaxed">{summary}</p>
        )}

        {pendingRevision && (
          <p className="text-[11px] font-mono text-violet-300">
            Applying your feedback: “{String(pendingRevision.inputs?.feedback || "").slice(0, 160)}”
            {" "}— the screenshots below refresh when the revised build is verified.
          </p>
        )}

        {shots.length > 0 && (
          <div>
            <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500 mb-1.5 flex items-center gap-1">
              <ImageIcon className="h-3 w-3" /> Screens ({shots.length} route{shots.length === 1 ? "" : "s"})
            </p>
            <div className="flex gap-3 overflow-x-auto pb-1">
              {shots.map((s) => (
                <figure key={s.route} className="shrink-0 w-56">
                  <button
                    type="button"
                    onClick={() => s.screenshot && setLightbox(s)}
                    title={s.screenshot ? "Enlarge" : "No screenshot"}
                    className={cn(
                      "block w-56 h-36 rounded-lg border overflow-hidden bg-slate-900",
                      s.mounted && s.errors.length === 0
                        ? "border-white/10 hover:border-emerald-500/40"
                        : "border-red-500/40",
                    )}
                  >
                    {s.screenshot ? (
                      <img
                        src={api.agentSessionScreenshotUrl(sid, s.screenshot, report?.created_at)}
                        alt={`Screenshot of ${s.route}`}
                        className="w-full h-full object-cover object-top"
                        loading="lazy"
                      />
                    ) : (
                      <span className="flex h-full items-center justify-center text-[10px] font-mono text-slate-500">
                        no screenshot
                      </span>
                    )}
                  </button>
                  <figcaption className="mt-1 flex items-center gap-1.5 text-[10px] font-mono">
                    <span className="text-slate-200 truncate">{s.route}</span>
                    {s.mounted && s.errors.length === 0
                      ? <span className="text-emerald-400">renders</span>
                      : <span className="text-red-400" title={s.errors.join("\n")}>
                          {s.mounted ? "errors" : "blank"}
                        </span>}
                  </figcaption>
                </figure>
              ))}
            </div>
          </div>
        )}

        {report && (
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button" onClick={() => void openPreview(false)}
              disabled={previewBusy || running}
              className="av-btn-primary"
              title={running ? "Available when the build finishes" : "Open the built app in a new tab"}
            ><ExternalLink className="h-3 w-3" /> Open live preview</button>
            <button
              type="button" onClick={() => (previewInline ? void stopPreview() : void openPreview(true))}
              disabled={previewBusy || running}
              className="av-btn-ghost"
            >
              {previewInline ? <><Square className="h-3 w-3" /> Close inline preview</>
                             : <><MonitorPlay className="h-3 w-3" /> Preview here</>}
            </button>
            {preview && (
              <a href={preview.url} target="_blank" rel="noopener noreferrer"
                 className="text-[10px] font-mono text-emerald-400 hover:underline truncate max-w-[16rem]">
                {preview.url}
              </a>
            )}
            {previewError && (
              <span className="text-[10px] font-mono text-red-300">{previewError}</span>
            )}
          </div>
        )}

        {previewInline && preview && (
          <iframe
            title="Live preview of the built app"
            src={preview.url}
            className="w-full h-[480px] rounded-lg border border-white/10 bg-white"
          />
        )}

        {revisions.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500 flex items-center gap-1">
              <FileDiff className="h-3 w-3" /> Revisions ({revisions.length})
            </p>
            {revisions.map((r) => <RevisionRow key={r.artifact_id} artifact={r} />)}
          </div>
        )}

        <p className="text-[10px] font-mono text-slate-500">
          Review the screens or the live preview, then describe what to change in the box below
          (“move the Save button below the form”, “remember the API key after a reload”). CGX maps
          the feedback to the exact files, applies it, rebuilds and re-verifies.
        </p>
      </div>

      {lightbox && lightbox.screenshot && (
        <div
          role="dialog" aria-label={`Screenshot of ${lightbox.route}`}
          onClick={() => setLightbox(null)}
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-6 cursor-zoom-out"
        >
          <img
            src={api.agentSessionScreenshotUrl(sid, lightbox.screenshot, report?.created_at)}
            alt={`Screenshot of ${lightbox.route}`}
            className="max-h-full max-w-full rounded-lg border border-white/10 shadow-2xl"
          />
        </div>
      )}
    </section>
  );
}

// One revision: the feedback, what changed where (with why), diffs on demand.
export function RevisionRow({ artifact }: { artifact: ArtifactDTO }) {
  const [open, setOpen] = useState(false);
  const c = artifact.content || {};
  const changed: string[] = Array.isArray(c.changed) ? c.changed.map(String) : [];
  const unchanged: string[] = Array.isArray(c.unchanged) ? c.unchanged.map(String) : [];
  const failed: string[] = Array.isArray(c.failed) ? c.failed.map(String) : [];
  const changes: any[] = Array.isArray(c.changes) ? c.changes : [];
  const status = String(c.status || "");
  const tone = status === "applied" ? "text-emerald-400"
    : status === "unlocalized" ? "text-red-300" : "text-amber-300";
  return (
    <div className="rounded-lg border border-white/5 bg-slate-950/60">
      <button
        type="button" onClick={() => setOpen((v) => !v)}
        className="w-full flex items-start gap-2 px-3 py-2 text-left hover:bg-white/5"
      >
        {open ? <ChevronDown className="h-3 w-3 mt-0.5 text-slate-500 shrink-0" />
              : <ChevronRight className="h-3 w-3 mt-0.5 text-slate-500 shrink-0" />}
        <div className="min-w-0 flex-1">
          <p className="text-[11px] text-slate-200">
            <span className="text-slate-500 font-mono">#{String(c.revision_index ?? "?")} </span>
            “{String(c.feedback || "")}”
          </p>
          <p className={cn("text-[10px] font-mono mt-0.5", tone)}>
            {status === "applied"
              ? `changed ${changed.join(", ")}`
              : String(c.note || c.reason || status)}
            {unchanged.length > 0 && <span className="text-slate-500"> · unchanged: {unchanged.join(", ")}</span>}
            {failed.length > 0 && <span className="text-red-400"> · failed: {failed.join(", ")}</span>}
          </p>
          {c.visual_check && (
            <p
              className={cn("text-[10px] font-mono mt-0.5",
                c.visual_check.satisfied === true ? "text-emerald-400"
                  : c.visual_check.satisfied === false ? "text-amber-300" : "text-slate-500")}
              title={`screenshot review by ${String(c.visual_check.model || "a vision model")}`}
            >
              {c.visual_check.satisfied === true
                ? "screenshot review: visibly addressed"
                : c.visual_check.satisfied === false
                  ? `screenshot review: NOT visibly addressed${c.visual_check.evidence ? ` — ${String(c.visual_check.evidence)}` : ""}`
                  : "screenshot review: could not judge from screenshots"}
            </p>
          )}
          {Number(c.auto_round) > 0 && (
            <p className="text-[10px] font-mono mt-0.5 text-violet-300">
              automatic follow-up #{String(c.auto_round)} (the previous revision did not visibly meet the request)
            </p>
          )}
        </div>
        <span className="text-[9px] font-mono text-slate-500 shrink-0">{formatRelative(artifact.created_at)}</span>
      </button>
      {open && (
        <div className="border-t border-white/5 px-3 py-2 space-y-2">
          {c.note && status === "applied" && (
            <p className="text-[11px] text-slate-300">{String(c.note)}</p>
          )}
          {Array.isArray(c.parts) && c.parts.length > 1 && (
            <ol className="text-[10px] font-mono text-slate-400 list-decimal pl-4 space-y-0.5">
              {c.parts.map((part: string, i: number) => <li key={i}>{part}</li>)}
            </ol>
          )}
          {c.theme && (
            <p className="text-[10px] font-mono text-violet-300">
              theme: {String(c.theme.from)} → {String(c.theme.to)} ({String(c.theme.tokens_path)})
            </p>
          )}
          {changes.map((ch, i) => (
            <div key={i} className="rounded border border-white/5 bg-slate-950 px-2 py-1.5">
              <p className="text-[11px] font-mono text-slate-100 flex items-center gap-2 flex-wrap">
                {ch.path}
                {ch.part != null && Array.isArray(c.parts) && c.parts.length > 1 && (
                  <span className="text-[9px] text-slate-500">part {String(ch.part)}</span>
                )}
                <span className={cn(
                  "text-[9px] uppercase tracking-wider",
                  ch.outcome === "changed" ? "text-emerald-400"
                    : ch.outcome === "unchanged" ? "text-amber-300" : "text-red-400",
                )}>{String(ch.outcome || "")}</span>
                {ch.partial && (
                  <span
                    className="text-[9px] uppercase tracking-wider text-amber-300"
                    title={`the diff did not touch: ${(ch.tokens || []).filter((t: string) => !(ch.covered || []).includes(t)).join(", ")}`}
                  >partial</span>
                )}
              </p>
              {ch.why && <p className="text-[10px] font-mono text-slate-500">why: {String(ch.why)}</p>}
              {ch.instruction && (
                <p className="text-[11px] text-slate-300 mt-0.5">{String(ch.instruction)}</p>
              )}
              {ch.diff && (
                <details className="mt-1">
                  <summary className="cursor-pointer text-[10px] font-mono text-slate-400 hover:text-slate-200">
                    diff
                  </summary>
                  <pre className="mt-1 max-h-64 overflow-auto rounded bg-slate-950/80 p-2 text-[10px] font-mono leading-snug">
                    {String(ch.diff).split("\n").map((line, li) => (
                      <span key={li} className={cn(
                        "block",
                        line.startsWith("+") && !line.startsWith("+++") ? "text-emerald-300"
                          : line.startsWith("-") && !line.startsWith("---") ? "text-red-300"
                          : line.startsWith("@@") ? "text-sky-300" : "text-slate-400",
                      )}>{line}</span>
                    ))}
                  </pre>
                </details>
              )}
            </div>
          ))}
          {Array.isArray(c.candidates) && c.candidates.length > 0 && (
            <details>
              <summary className="cursor-pointer text-[10px] font-mono text-slate-500 hover:text-slate-300">
                how the files were located ({c.candidates.length} candidate{c.candidates.length === 1 ? "" : "s"})
              </summary>
              <ul className="mt-1 space-y-0.5">
                {c.candidates.map((cand: any, i: number) => (
                  <li key={i} className="text-[10px] font-mono text-slate-400">
                    <span className="text-slate-200">{String(cand.path)}</span>
                    <span className="text-slate-600"> ({String(cand.score)})</span>
                    {Array.isArray(cand.reasons) && cand.reasons.length > 0 && (
                      <span> — {cand.reasons.join("; ")}</span>
                    )}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
