import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ExternalLink, Globe, Loader2, Monitor, Plus, RefreshCw,
  Send, Smartphone, Sparkles,
} from "lucide-react";
import {
  api,
  type AgentSessionState, type SiteInfo, type TaskNodeDTO, type TaskProgress,
} from "../lib/api";
import { useWorkspace } from "../store/workspace";
import { applySessionEvent } from "../store/agentEvents";
import { ActiveTaskPanel } from "../components/agent/ActiveTask";
import { TextArea } from "../components/Input";
import { cn } from "../lib/utils";

// Site Studio: a build-on-the-left, live-preview-on-the-right workspace for
// creating a plain static HTML/CSS/JS website end to end. It drives a normal
// greenfield agent session (pinned to the static_site skill) and renders the
// generated files from the sandboxed preview endpoint, reloading on each new
// applied_changes. Feedback is a plain follow-up that re-drives the build.
export default function SiteStudioPage() {
  const { provider, index } = useWorkspace();

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [state, setState] = useState<AgentSessionState | null>(null);
  const [progress, setProgress] = useState<Record<string, TaskProgress>>({});
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [siteName, setSiteName] = useState("");
  const [brief, setBrief] = useState("");
  const [feedback, setFeedback] = useState("");
  const [sites, setSites] = useState<SiteInfo[]>([]);
  const [viewport, setViewport] = useState<"desktop" | "mobile">("desktop");
  const [previewNonce, setPreviewNonce] = useState(0);

  const sseOkRef = useRef(false);

  const refreshSites = useCallback(async () => {
    try { setSites(await api.listSites()); } catch { /* best-effort */ }
  }, []);

  useEffect(() => { void refreshSites(); }, [refreshSites]);

  // Live session stream (mirrors RunTab): apply each frame and, whenever the
  // build writes files, bump the preview nonce so the iframe reloads.
  useEffect(() => {
    if (!sessionId) return;
    setProgress({});
    sseOkRef.current = false;
    const es = new EventSource(api.agentSessionEventsUrl(sessionId));
    es.addEventListener("snapshot", (e: MessageEvent) => {
      sseOkRef.current = true;
      try { setState(JSON.parse(e.data)); } catch { /* ignore */ }
    });
    es.addEventListener("task.output_partial", (e: MessageEvent) => {
      try {
        const ev = JSON.parse(e.data);
        const p = ev?.payload?.progress;
        const tid = ev?.payload?.task_id;
        if (tid && p) setProgress((prev) => ({ ...prev, [tid]: p }));
      } catch { /* ignore */ }
    });
    [
      "session.created", "session.updated", "task.created",
      "task.status_changed", "task.completed", "task.failed",
      "decision.recorded", "fact.added", "fact.stale", "artifact.created",
    ].forEach((name) => es.addEventListener(name, (e: MessageEvent) => {
      sseOkRef.current = true;
      try {
        const ev = JSON.parse(e.data);
        setState((prev) => (prev ? applySessionEvent(prev, name, ev?.payload) : prev));
        if (name === "artifact.created"
            && ev?.payload?.artifact?.kind === "applied_changes") {
          setPreviewNonce((n) => n + 1);
        }
      } catch { /* ignore */ }
    }));
    es.onerror = () => { sseOkRef.current = false; };
    return () => es.close();
  }, [sessionId]);

  const startBuild = useCallback(async () => {
    const name = siteName.trim();
    const objective = brief.trim();
    if (!name || !objective) {
      setError("Give the site a name and describe what it should contain.");
      return;
    }
    setPending(true); setError(null);
    try {
      const site = await api.createSite(name);
      const next = await api.agentSessionCreate({
        objective,
        project_root: site.project_root,
        title: name,
        mode: "greenfield",
        index, provider,
        run_initial_task: true,
        skills: ["static_site"],
        require_plan_approval: true,
      });
      setState(next);
      setSessionId(next.session.session_id);
      void refreshSites();
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, [siteName, brief, index, provider, refreshSites]);

  const openExisting = useCallback(async (site: SiteInfo) => {
    setPending(true); setError(null);
    try {
      const list = await api.agentSessionList(site.project_root);
      if (list.length > 0) {
        const sid = list[0].session_id;
        const st = await api.agentSessionGet(sid, site.project_root);
        setState(st);
        setSessionId(sid);
        setPreviewNonce((n) => n + 1);
      } else {
        setSiteName(site.slug);
        setError("No prior build for this site yet — describe it and build.");
      }
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, []);

  const onDecide = useCallback(async (payload: {
    chosen: Record<string, any>; rationale?: string;
  }) => {
    if (!sessionId || !state) return;
    const task = activeTaskOf(state);
    if (!task) return;
    setPending(true); setError(null);
    try {
      const next = await api.agentSessionDecision(sessionId, {
        task_id: task.task_id,
        chosen: payload.chosen,
        rationale: payload.rationale ?? null,
        index, provider, run_initial_task: true,
      });
      setState(next);
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, [sessionId, state, index, provider]);

  const sendFeedback = useCallback(async () => {
    if (!sessionId || !feedback.trim()) return;
    setPending(true); setError(null);
    try {
      const next = await api.agentSessionMessage(sessionId, {
        message: feedback.trim(), index, provider, run_initial_task: true,
      });
      setState(next); setFeedback("");
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, [sessionId, feedback, index, provider]);

  const cancel = useCallback(async () => {
    if (!sessionId) return;
    try { setState(await api.agentSessionCancel(sessionId)); }
    catch (e) { setError(String((e as Error)?.message || e)); }
  }, [sessionId]);

  const reset = () => {
    setSessionId(null); setState(null); setProgress({});
    setBrief(""); setError(null); void refreshSites();
  };

  const activeTask = state ? activeTaskOf(state) : null;
  const building = !!state?.tasks.some(
    (t) => t.kind !== "ask_user"
      && (t.status === "in_progress" || t.status === "ready"));
  const hasOutput = !!state?.artifacts?.some((a) => a.kind === "applied_changes");
  const previewUrl = sessionId && hasOutput
    ? `${api.agentSessionPreviewUrl(sessionId)}?v=${previewNonce}`
    : "";
  const liveProgress = useMemo(() => {
    const vals = Object.values(progress);
    return vals.length ? vals[vals.length - 1] : null;
  }, [progress]);

  // ---- start screen -------------------------------------------------------
  if (!sessionId || !state) {
    return (
      <div className="h-full overflow-y-auto p-8 max-w-3xl mx-auto space-y-6">
        <div className="flex items-center gap-2">
          <Globe className="h-5 w-5 text-emerald-400" />
          <h1 className="text-lg font-bold text-white">Site Studio</h1>
        </div>
        <p className="text-sm text-slate-400">
          Describe a website in plain English. CGX asks a few questions, plans
          the pages, builds a static HTML/CSS/JS site, and renders it live so
          you can give feedback and iterate — all on your machine.
        </p>

        <div className="rounded-xl border border-white/10 bg-slate-950/40 p-5 space-y-4">
          <label className="block space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
              Site name
            </span>
            <input
              value={siteName}
              onChange={(e) => setSiteName(e.target.value)}
              placeholder="my-portfolio"
              className="av-input w-full text-sm"
              spellCheck={false}
            />
            <span className="text-[10px] font-mono text-slate-600">
              Created under ~/.cgx/sites/&lt;slug&gt;
            </span>
          </label>
          <label className="block space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
              What should it contain?
            </span>
            <TextArea
              rows={5}
              value={brief}
              onChange={(e) => setBrief(e.target.value)}
              placeholder={"e.g. A personal portfolio with a home page, an "
                + "about page, a projects gallery, and a contact form. Clean, "
                + "modern, dark theme, responsive."}
            />
          </label>
          {error && <p className="text-xs text-red-300 font-mono">{error}</p>}
          <div className="flex justify-end">
            <button className="av-btn-primary" onClick={startBuild} disabled={pending}>
              {pending ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                       : <Sparkles className="h-3.5 w-3.5" />}
              {pending ? "Starting…" : "Build my site"}
            </button>
          </div>
        </div>

        {sites.length > 0 && (
          <div className="space-y-2">
            <p className="text-[11px] font-mono uppercase tracking-wider text-slate-500">
              Your sites
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {sites.map((s) => (
                <button
                  key={s.slug}
                  onClick={() => openExisting(s)}
                  disabled={pending}
                  className="text-left rounded-lg border border-white/10 bg-slate-950/40 px-3 py-2 hover:border-emerald-500/30 disabled:opacity-40"
                >
                  <p className="text-sm text-slate-100 font-medium truncate">{s.slug}</p>
                  <p className="text-[10px] font-mono text-slate-500 truncate">
                    {s.has_index ? "built" : "empty"} · {s.project_root}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // ---- studio (split view) ------------------------------------------------
  return (
    <div className="h-full flex overflow-hidden">
      {/* Build column */}
      <div className="w-[42%] min-w-[340px] max-w-[560px] border-r border-muted flex flex-col">
        <div className="px-4 py-3 border-b border-muted flex items-center gap-2">
          <Globe className="h-4 w-4 text-emerald-400 shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-white truncate">
              {state.session.title || "Site"}
            </p>
            <p className="text-[10px] font-mono text-slate-500 truncate">
              {state.session.status}
              {state.session.current_focus ? ` · ${state.session.current_focus}` : ""}
            </p>
          </div>
          {building && (
            <button className="av-btn-ghost py-1 px-2 text-[10px]" onClick={cancel}>
              Stop
            </button>
          )}
          <button className="av-btn-ghost py-1 px-2 text-[10px]" onClick={reset}>
            <Plus className="h-3 w-3" /> New
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {building && liveProgress && (
            <div className="rounded-lg border border-amber-500/20 bg-amber-950/10 px-3 py-2">
              <div className="flex items-center gap-2 text-[11px] font-mono text-amber-300">
                <Loader2 className="h-3 w-3 animate-spin" />
                Building {liveProgress.index}/{liveProgress.total}
                <span className="text-slate-400 truncate">{liveProgress.path}</span>
              </div>
            </div>
          )}
          <ActiveTaskPanel
            task={activeTask}
            artifacts={state.artifacts}
            decisions={state.decisions}
            facts={state.facts}
            onDecide={onDecide}
            pending={pending}
          />
        </div>

        {/* Feedback bar */}
        <div className="border-t border-muted p-3 space-y-2">
          {error && <p className="text-[11px] text-red-300 font-mono">{error}</p>}
          <div className="flex items-end gap-2">
            <TextArea
              rows={2}
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="Feedback: e.g. make the header sticky, use a blue hero, add a pricing page…"
              disabled={pending}
              className="flex-1"
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault(); void sendFeedback();
                }
              }}
            />
            <button
              className="av-btn-primary shrink-0"
              onClick={sendFeedback}
              disabled={pending || !feedback.trim()}
            >
              {pending ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                       : <Send className="h-3.5 w-3.5" />}
              Send
            </button>
          </div>
        </div>
      </div>

      {/* Preview column */}
      <div className="flex-1 flex flex-col bg-slate-900/40 min-w-0">
        <div className="px-4 py-2 border-b border-muted flex items-center gap-2">
          <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500">
            Live preview
          </span>
          <div className="ml-auto flex items-center gap-1">
            <button
              className={cn("av-btn-icon h-6 w-6", viewport === "desktop" && "text-emerald-400")}
              title="Desktop" onClick={() => setViewport("desktop")}
            ><Monitor className="h-3.5 w-3.5" /></button>
            <button
              className={cn("av-btn-icon h-6 w-6", viewport === "mobile" && "text-emerald-400")}
              title="Mobile" onClick={() => setViewport("mobile")}
            ><Smartphone className="h-3.5 w-3.5" /></button>
            <button
              className="av-btn-icon h-6 w-6" title="Reload"
              onClick={() => setPreviewNonce((n) => n + 1)}
              disabled={!previewUrl}
            ><RefreshCw className="h-3.5 w-3.5" /></button>
            <a
              className={cn("av-btn-icon h-6 w-6", !previewUrl && "pointer-events-none opacity-40")}
              title="Open in new tab" href={previewUrl || "#"}
              target="_blank" rel="noopener noreferrer"
            ><ExternalLink className="h-3.5 w-3.5" /></a>
          </div>
        </div>
        <div className="flex-1 overflow-auto flex items-start justify-center p-4">
          {previewUrl ? (
            <iframe
              key={previewNonce}
              title="site-preview"
              src={previewUrl}
              // Untrusted generated content: allow scripts to run but keep it
              // in an opaque origin (no allow-same-origin) so it cannot touch
              // the CGX app's origin, cookies, or API.
              sandbox="allow-scripts allow-forms allow-popups allow-modals"
              className={cn(
                "bg-white rounded-lg border border-white/10 shadow-2xl h-full",
                viewport === "mobile" ? "w-[390px]" : "w-full",
              )}
            />
          ) : (
            <div className="h-full w-full flex flex-col items-center justify-center text-center gap-2 text-slate-500">
              <Loader2 className="h-5 w-5 animate-spin text-emerald-500/70" />
              <p className="text-xs font-mono">
                {building ? "Building your site…" : "Preview appears once the first files are written."}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// The task the user most likely cares about right now: a pending question,
// else whatever is running, else the latest.
function activeTaskOf(state: AgentSessionState): TaskNodeDTO | null {
  const tasks = state.tasks || [];
  const ask = tasks.find((t) => t.kind === "ask_user" && t.status === "in_progress");
  if (ask) return ask;
  const running = tasks.find((t) => t.status === "in_progress" || t.status === "ready");
  if (running) return running;
  return tasks.length ? tasks[tasks.length - 1] : null;
}
