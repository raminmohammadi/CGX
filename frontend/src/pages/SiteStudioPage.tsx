import { useCallback, useEffect, useState } from "react";
import {
  ExternalLink, Globe, Loader2, Monitor, Plus, RefreshCw,
  Send, Smartphone, Sparkles,
} from "lucide-react";
import { api, type SiteInfo } from "../lib/api";
import { useWorkspace } from "../store/workspace";
import { TextArea } from "../components/Input";
import { cn } from "../lib/utils";

// Build flavors -- all NO build step and self-contained, so every one renders
// instantly in the sandboxed preview. The flavor is sent to the backend,
// which styles accordingly (plain CSS / Tailwind CDN / Tailwind + Alpine).
type Flavor = "simple" | "modern" | "interactive";
const FLAVORS: { key: Flavor; label: string; hint: string }[] = [
  { key: "simple", label: "Simple", hint: "Hand-written CSS, works offline" },
  { key: "modern", label: "Modern", hint: "Tailwind + fonts, polished" },
  { key: "interactive", label: "Interactive", hint: "Modern + Alpine.js" },
];

type LogEntry = { kind: "build" | "revise"; text: string };

// Site Studio: describe a site (left) and see it render live (right). It uses
// a focused one-shot generator that emits a complete self-contained
// index.html -- far better than the generic build pipeline for "make me a
// website" -- and revises that file in place from plain-language feedback.
export default function SiteStudioPage() {
  const { provider } = useWorkspace();

  const [slug, setSlug] = useState<string | null>(null);
  const [siteName, setSiteName] = useState("");
  const [brief, setBrief] = useState("");
  const [flavor, setFlavor] = useState<Flavor>("modern");
  const [feedback, setFeedback] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sites, setSites] = useState<SiteInfo[]>([]);
  const [viewport, setViewport] = useState<"desktop" | "mobile">("desktop");
  const [previewNonce, setPreviewNonce] = useState(0);
  const [log, setLog] = useState<LogEntry[]>([]);

  const refreshSites = useCallback(async () => {
    try { setSites(await api.listSites()); } catch { /* best-effort */ }
  }, []);
  useEffect(() => { void refreshSites(); }, [refreshSites]);

  const startBuild = useCallback(async () => {
    const name = siteName.trim();
    const b = brief.trim();
    if (!name || !b) {
      setError("Give the site a name and describe what it should contain.");
      return;
    }
    setBusy(true); setBusyLabel("Designing your site…"); setError(null);
    try {
      const res = await api.generateSite({ name, brief: b, flavor, provider });
      setSlug(res.slug);
      setLog([{ kind: "build", text: b }]);
      setPreviewNonce((n) => n + 1);
      void refreshSites();
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setBusy(false); setBusyLabel(""); }
  }, [siteName, brief, flavor, provider, refreshSites]);

  const openExisting = useCallback((site: SiteInfo) => {
    setSlug(site.slug);
    setLog([]);
    setPreviewNonce((n) => n + 1);
  }, []);

  const sendFeedback = useCallback(async () => {
    const fb = feedback.trim();
    if (!slug || !fb) return;
    setBusy(true); setBusyLabel("Applying your change…"); setError(null);
    try {
      await api.reviseSite({ slug, feedback: fb, flavor, provider });
      setLog((l) => [...l, { kind: "revise", text: fb }]);
      setFeedback("");
      setPreviewNonce((n) => n + 1);
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setBusy(false); setBusyLabel(""); }
  }, [slug, feedback, flavor, provider]);

  const reset = () => {
    setSlug(null); setLog([]); setBrief(""); setFeedback(""); setError(null);
    void refreshSites();
  };

  const previewUrl = slug
    ? `${api.sitePreviewUrl(slug)}?v=${previewNonce}`
    : "";

  // ---- start screen -------------------------------------------------------
  if (!slug) {
    return (
      <div className="h-full overflow-y-auto p-8 max-w-3xl mx-auto space-y-6">
        <div className="flex items-center gap-2">
          <Globe className="h-5 w-5 text-emerald-400" />
          <h1 className="text-lg font-bold text-white">Site Studio</h1>
        </div>
        <p className="text-sm text-slate-400">
          Describe a website in plain English. CGX designs a complete, styled
          page, renders it live, and lets you refine it with feedback — all on
          your machine.
        </p>

        <div className="rounded-xl border border-white/10 bg-slate-950/40 p-5 space-y-4">
          <label className="block space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
              Site name
            </span>
            <input
              value={siteName}
              onChange={(e) => setSiteName(e.target.value)}
              placeholder="ember-and-oak"
              className="av-input w-full text-sm"
              spellCheck={false}
            />
          </label>
          <label className="block space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
              What should it be?
            </span>
            <TextArea
              rows={5}
              value={brief}
              onChange={(e) => setBrief(e.target.value)}
              placeholder={"e.g. A landing page for a cozy neighborhood coffee "
                + "shop called Ember & Oak: hero, featured menu with prices, our "
                + "story, opening hours, and a contact section. Warm, modern, "
                + "responsive."}
            />
          </label>
          <div className="space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
              Style
            </span>
            <div className="grid grid-cols-3 gap-2">
              {FLAVORS.map((f) => (
                <button
                  key={f.key} type="button" onClick={() => setFlavor(f.key)}
                  className={cn(
                    "text-left rounded-lg border px-3 py-2 transition",
                    flavor === f.key
                      ? "border-emerald-500/40 bg-emerald-500/10"
                      : "border-white/10 bg-slate-950/40 hover:border-white/20",
                  )}
                >
                  <p className={cn("text-xs font-medium",
                    flavor === f.key ? "text-emerald-300" : "text-slate-200")}>
                    {f.label}
                  </p>
                  <p className="text-[10px] text-slate-500 leading-snug mt-0.5">{f.hint}</p>
                </button>
              ))}
            </div>
            <p className="text-[10px] font-mono text-slate-600">
              Every flavor is build-less and previews instantly. Full React/Vue
              apps (which need a build) live in the Agent Loop.
            </p>
          </div>
          {error && <p className="text-xs text-red-300 font-mono">{error}</p>}
          <div className="flex justify-end">
            <button className="av-btn-primary" onClick={startBuild} disabled={busy}>
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    : <Sparkles className="h-3.5 w-3.5" />}
              {busy ? busyLabel || "Working…" : "Build my site"}
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
                  key={s.slug} onClick={() => openExisting(s)}
                  className="text-left rounded-lg border border-white/10 bg-slate-950/40 px-3 py-2 hover:border-emerald-500/30"
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
      <div className="w-[38%] min-w-[320px] max-w-[520px] border-r border-muted flex flex-col">
        <div className="px-4 py-3 border-b border-muted flex items-center gap-2">
          <Globe className="h-4 w-4 text-emerald-400 shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-white truncate">{slug}</p>
            <p className="text-[10px] font-mono text-slate-500">
              {busy ? busyLabel : "ready · edit with feedback below"}
            </p>
          </div>
          <button className="av-btn-ghost py-1 px-2 text-[10px]" onClick={reset}>
            <Plus className="h-3 w-3" /> New
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {log.length === 0 && !busy && (
            <p className="text-[12px] text-slate-500">
              Opened an existing site. Describe a change below to revise it.
            </p>
          )}
          {log.map((e, i) => (
            <div key={i} className="rounded-lg border border-white/10 bg-slate-950/40 px-3 py-2">
              <p className="text-[10px] font-mono uppercase tracking-wider text-emerald-400/70">
                {e.kind === "build" ? "Brief" : "Revision"}
              </p>
              <p className="text-[12px] text-slate-200 mt-0.5 whitespace-pre-wrap">{e.text}</p>
            </div>
          ))}
          {busy && (
            <div className="rounded-lg border border-amber-500/20 bg-amber-950/10 px-3 py-2 flex items-center gap-2 text-[11px] font-mono text-amber-300">
              <Loader2 className="h-3 w-3 animate-spin" /> {busyLabel}
            </div>
          )}
        </div>

        <div className="border-t border-muted p-3 space-y-2">
          {error && <p className="text-[11px] text-red-300 font-mono">{error}</p>}
          <div className="flex items-end gap-2">
            <TextArea
              rows={2}
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="Change: e.g. make the header sticky, warmer palette, add a testimonials section…"
              disabled={busy}
              className="flex-1"
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault(); void sendFeedback();
                }
              }}
            />
            <button className="av-btn-primary shrink-0" onClick={sendFeedback}
                    disabled={busy || !feedback.trim()}>
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
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
            <button className={cn("av-btn-icon h-6 w-6", viewport === "desktop" && "text-emerald-400")}
                    title="Desktop" onClick={() => setViewport("desktop")}>
              <Monitor className="h-3.5 w-3.5" /></button>
            <button className={cn("av-btn-icon h-6 w-6", viewport === "mobile" && "text-emerald-400")}
                    title="Mobile" onClick={() => setViewport("mobile")}>
              <Smartphone className="h-3.5 w-3.5" /></button>
            <button className="av-btn-icon h-6 w-6" title="Reload"
                    onClick={() => setPreviewNonce((n) => n + 1)}>
              <RefreshCw className="h-3.5 w-3.5" /></button>
            <a className="av-btn-icon h-6 w-6" title="Open in new tab"
               href={previewUrl} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-3.5 w-3.5" /></a>
          </div>
        </div>
        <div className="flex-1 overflow-auto flex items-start justify-center p-4">
          {busy && log.length === 0 ? (
            <div className="h-full w-full flex flex-col items-center justify-center gap-2 text-slate-500">
              <Loader2 className="h-5 w-5 animate-spin text-emerald-500/70" />
              <p className="text-xs font-mono">{busyLabel || "Designing your site…"}</p>
            </div>
          ) : (
            <iframe
              key={previewNonce}
              title="site-preview"
              src={previewUrl}
              // Untrusted generated content: scripts run but in an opaque
              // origin (no allow-same-origin) so it can't reach the CGX app.
              sandbox="allow-scripts allow-forms allow-popups allow-modals"
              className={cn(
                "bg-white rounded-lg border border-white/10 shadow-2xl h-full",
                viewport === "mobile" ? "w-[390px]" : "w-full",
              )}
            />
          )}
        </div>
      </div>
    </div>
  );
}
