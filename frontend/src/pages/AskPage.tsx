import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot, Brain, ChevronDown, PanelRightClose, PanelRightOpen, Sparkles,
  Square, TriangleAlert, User,
} from "lucide-react";
import { api, type SessionMessage } from "../lib/api";
import { streamSSE } from "../lib/sse";
import { abortConnection, getConnection, setConnection } from "../lib/connections";
import { useTasks, type ChatMsg } from "../store/tasks";
import { useWorkspace } from "../store/workspace";
import { Markdown } from "../components/Markdown";
import { Pill } from "../components/Pill";

const PAGE_KEY = "ask";

// Model families that expose a native reasoning / "thinking" phase. Mirrors
// the backend registry in cgx.answer.model_caps.model_supports_thinking so
// the toggle is only offered when the server would actually honor it.
const THINKING_MODEL_KEYS = [
  "deepseek-r1", "qwq", "qwen3", "gpt-oss", "magistral", "cogito",
  "smallthinker", "phi4-reasoning", "phi-4-reasoning", "granite3.2",
  "granite3.3", "o1", "o3", "o4-mini", "gemini-2.5",
];

function modelSupportsThinking(model: string | undefined): boolean {
  if (!model) return false;
  const m = model.trim().toLowerCase();
  return THINKING_MODEL_KEYS.some((k) => m.includes(k));
}

export default function AskPage() {
  const { provider, index, selectedSessionId, setSelectedSession, setProvider } = useWorkspace();
  const { ask, setAsk, appendAskMessage, resetAsk } = useTasks();
  const { busy, messages, error } = ask;

  const threadRef = useRef<HTMLDivElement | null>(null);
  const draftRef = useRef<HTMLTextAreaElement | null>(null);

  // On mount: if busy but no live connection, stream finished while we were away.
  useEffect(() => {
    if (busy && !getConnection(PAGE_KEY)) {
      setAsk({ busy: false });
      // Mark any still-streaming message as done.
      useTasks.setState((s) => {
        const msgs = s.ask.messages.map((m) =>
          m.streaming ? { ...m, streaming: false } : m
        );
        return { ask: { ...s.ask, messages: msgs } };
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load session messages when the selected session changes to one we haven't
  // loaded yet. The ask store is a module-level singleton that keeps streaming
  // across tab switches, so on remount we must NOT re-fetch and clobber the
  // messages already in the store for this session -- mid-stream the server
  // has nothing persisted yet, so a re-fetch would blank the in-flight answer.
  // Keying on the store's own sessionId also covers the brand-new session that
  // send() creates, so no separate skip-flag is needed.
  useEffect(() => {
    let alive = true;
    (async () => {
      if (!selectedSessionId) { resetAsk(); return; }
      if (useTasks.getState().ask.sessionId === selectedSessionId) return;
      try {
        const items = await api.sessionMessages(selectedSessionId);
        if (!alive) return;
        const conv: ChatMsg[] = items
          .filter((m: SessionMessage) => m.role === "user" || m.role === "assistant")
          .map((m: SessionMessage) => ({
            role: m.role as "user" | "assistant",
            content: m.content,
            sources: (m.meta as any)?.sources,
            intent: (m.meta as any)?.intent,
          }));
        setAsk({ messages: conv, sessionId: selectedSessionId });
      } catch {
        if (alive) resetAsk();
      }
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSessionId]);

  useEffect(() => {
    if (threadRef.current)
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
  }, [messages]);

  const lastAssistant = useMemo(
    () => [...messages].reverse().find((m) => m.role === "assistant"),
    [messages],
  );

  const startSession = useCallback(async () => {
    try {
      const s = await api.createSession();
      setSelectedSession(s.id);
      return s.id;
    } catch {
      return null;
    }
  }, [setSelectedSession]);

  // Patch the last message in the store using a functional updater (safe from closures).
  const patchLast = (patch: Partial<ChatMsg>) => {
    useTasks.setState((s) => {
      const msgs = [...s.ask.messages];
      if (!msgs.length) return s;
      const last = msgs[msgs.length - 1];
      if (last.role !== "assistant") return s;
      msgs[msgs.length - 1] = { ...last, ...patch };
      return { ask: { ...s.ask, messages: msgs } };
    });
  };

  const send = async () => {
    const text = (draftRef.current?.value ?? "").trim();
    if (!text || busy) return;
    if (draftRef.current) draftRef.current.value = "";

    setAsk({ error: null, busy: true });

    let sid = selectedSessionId;
    if (!sid) sid = await startSession();
    // Tag the store with the session these messages belong to so the loader
    // effect won't re-fetch and wipe them when we navigate back to this tab.
    if (sid) setAsk({ sessionId: sid });

    appendAskMessage({ role: "user", content: text });
    appendAskMessage({ role: "assistant", content: "", streaming: true, thought: "" });

    abortConnection(PAGE_KEY);
    const conn = streamSSE(
      "/api/ask",
      { question: text, session_id: sid || null, index, provider },
      (ev, data) => {
        if (ev === "thought" && data?.delta) {
          // Append delta to thought using functional updater.
          useTasks.setState((s) => {
            const msgs = [...s.ask.messages];
            const last = msgs[msgs.length - 1];
            if (!last || last.role !== "assistant") return s;
            msgs[msgs.length - 1] = { ...last, thought: (last.thought ?? "") + String(data.delta) };
            return { ask: { ...s.ask, messages: msgs } };
          });
        } else if (ev === "thought_warning") {
          patchLast({ warning: String(data?.message || "") });
        } else if (ev === "intent") {
          patchLast({
            intent: {
              mode: String(data?.mode || ""),
              scope: data?.scope ? String(data.scope) : undefined,
            },
          });
        } else if (ev === "answer_delta" && data?.delta) {
          useTasks.setState((s) => {
            const msgs = [...s.ask.messages];
            const last = msgs[msgs.length - 1];
            if (!last || last.role !== "assistant") return s;
            msgs[msgs.length - 1] = { ...last, content: (last.content ?? "") + String(data.delta) };
            return { ask: { ...s.ask, messages: msgs } };
          });
        } else if (ev === "answer") {
          patchLast({
            content: String(data?.answer_md || ""),
            sources: Array.isArray(data?.sources) ? data.sources : [],
            streaming: false,
          });
        } else if (ev === "cancelled") {
          patchLast({ streaming: false, content: "_Cancelled._" });
        } else if (ev === "error") {
          patchLast({ content: `**Error:** ${data?.message || "unknown"}`, streaming: false });
          setAsk({ error: String(data?.message || "error") });
        } else if (ev === "done") {
          patchLast({ streaming: false });
        }
      },
      (err) => {
        setAsk({ error: String((err as any)?.message || err), busy: false });
        patchLast({ streaming: false, content: "_Connection closed before answer arrived._" });
      },
    );

    setConnection(PAGE_KEY, conn);
    conn.done.finally(() => {
      setAsk({ busy: false });
      patchLast({ streaming: false });
      abortConnection(PAGE_KEY);
    });
  };

  const cancel = () => {
    abortConnection(PAGE_KEY);
    setAsk({ busy: false });
    patchLast({ streaming: false, content: "_Cancelled._" });
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const sources = lastAssistant?.sources || [];
  const intent = lastAssistant?.intent;
  const sourceCount = sources.length;

  // Thinking toggle: only meaningful for reasoning-capable models. When the
  // active model doesn't support it, the control is disabled and treated as
  // off regardless of the persisted flag.
  const thinkSupported = modelSupportsThinking(provider.model);
  const thinkOn = !!provider.think && thinkSupported;

  // Default-collapsed retrieval panel: most reads are answer-first, so keep
  // the rank list one click away rather than always-on screen real estate.
  const [showSources, setShowSources] = useState(false);

  return (
    <div className="flex h-full w-full p-4 gap-4 overflow-hidden">
      <div className="flex-1 flex flex-col bg-surface rounded-xl border border-muted overflow-hidden min-w-0">
        <ChatHeader
          intent={intent}
          sourceCount={sourceCount}
          showSources={showSources}
          onToggleSources={() => setShowSources((v) => !v)}
        />

        <div ref={threadRef} className="flex-1 px-6 py-5 overflow-y-auto space-y-6">
          {messages.length === 0 && <AskEmptyState />}
          {messages.map((m, i) => (
            <ChatBubble key={i} msg={m} />
          ))}
        </div>

        <AskBar
          draftRef={draftRef}
          busy={busy}
          onSend={send}
          onCancel={cancel}
          onKeyDown={handleKey}
          error={error}
          think={thinkOn}
          thinkSupported={thinkSupported}
          onToggleThink={() => setProvider({ think: !thinkOn })}
          model={provider.model}
        />
      </div>

      {showSources && (
        <RetrievalPanel
          sources={sources}
          intent={intent}
          onClose={() => setShowSources(false)}
        />
      )}
    </div>
  );
}

function ChatHeader({
  intent, sourceCount, showSources, onToggleSources,
}: {
  intent?: { mode?: string; scope?: string };
  sourceCount: number;
  showSources: boolean;
  onToggleSources: () => void;
}) {
  return (
    <div className="flex items-center justify-between px-5 py-2.5 border-b border-muted bg-slate-950/40">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-[10px] uppercase tracking-[0.18em] font-mono text-slate-500">
          Grounded answer
        </span>
        {intent?.mode && <Pill tone="purple">{intent.mode}</Pill>}
        {intent?.scope && <Pill tone="slate">scope: {intent.scope}</Pill>}
      </div>
      <button
        onClick={onToggleSources}
        className={
          "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-mono transition " +
          (showSources
            ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
            : "bg-slate-900 text-slate-400 border border-white/5 hover:text-emerald-300 hover:border-emerald-500/30")
        }
        title={showSources ? "Hide retrieval ranks" : "Show retrieval ranks"}
      >
        {showSources
          ? <PanelRightClose className="h-3.5 w-3.5" />
          : <PanelRightOpen className="h-3.5 w-3.5" />}
        <span>Sources</span>
        {sourceCount > 0 && (
          <span className="px-1.5 py-px rounded-sm bg-emerald-500/20 text-emerald-300 text-[10px] font-semibold">
            {sourceCount}
          </span>
        )}
      </button>
    </div>
  );
}

function AskBar({
  draftRef, busy, onSend, onCancel, onKeyDown, error,
  think, thinkSupported, onToggleThink, model,
}: {
  draftRef: React.RefObject<HTMLTextAreaElement | null>;
  busy: boolean;
  onSend: () => void;
  onCancel: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  error: string | null;
  think: boolean;
  thinkSupported: boolean;
  onToggleThink: () => void;
  model: string;
}) {
  return (
    <div className="px-5 pt-3 pb-4 border-t border-muted bg-slate-950/60">
      <div className="rounded-2xl bg-slate-900/80 border border-white/5
                      focus-within:border-emerald-500/50 focus-within:shadow-neon
                      transition flex items-end gap-3 px-4 py-3">
        <Sparkles className="h-4 w-4 text-emerald-400/80 mt-1.5 shrink-0" />
        <textarea
          ref={draftRef}
          defaultValue=""
          onKeyDown={onKeyDown}
          rows={2}
          placeholder="Ask anything about this codebase — signatures, modules, behavior, change ideas…"
          className="flex-1 bg-transparent outline-none text-sm text-white py-1
                     resize-none placeholder-slate-500 max-h-48 leading-relaxed"
        />
        <button
          type="button"
          onClick={onToggleThink}
          aria-pressed={think}
          title={
            think
              ? (thinkSupported
                  ? "Thinking on — model reasons before answering. Click to turn off."
                  : `Thinking on, but ${model || "this model"} has no reasoning phase, so it won't take effect. Click to turn off.`)
              : "Thinking off — answers directly (faster). Click to turn on."
          }
          className={
            "shrink-0 self-stretch inline-flex items-center gap-1.5 px-3 rounded-xl text-xs font-mono border transition " +
            (think
              ? (thinkSupported
                  ? "bg-purple-500/15 text-purple-300 border-purple-500/40 shadow-[0_0_16px_-6px_rgba(168,85,247,0.7)]"
                  : "bg-amber-500/15 text-amber-300 border-amber-500/40")
              : "bg-slate-900 text-slate-300 border-white/10 hover:text-purple-300 hover:border-purple-500/40")
          }
        >
          <Brain className="h-3.5 w-3.5" />
          <span>Thinking</span>
          <span
            className={
              "px-1.5 py-px rounded-sm text-[10px] font-semibold " +
              (think
                ? (thinkSupported ? "bg-purple-500/25 text-purple-200" : "bg-amber-500/25 text-amber-200")
                : "bg-slate-800 text-slate-400")
            }
          >
            {think ? (thinkSupported ? "on" : "on*") : "off"}
          </span>
        </button>
        {busy ? (
          <button
            onClick={onCancel}
            className="shrink-0 inline-flex items-center gap-1.5 px-4 py-2 rounded-xl
                       bg-slate-800 text-slate-200 border border-white/10
                       hover:bg-slate-700 transition text-sm font-medium"
          >
            <Square className="h-3.5 w-3.5" /> Stop
          </button>
        ) : (
          <button
            onClick={onSend}
            className="shrink-0 inline-flex items-center gap-1.5 px-5 py-2 rounded-xl
                       bg-emerald-500 text-slate-950 hover:bg-emerald-400
                       transition text-sm font-bold shadow-[0_0_24px_-6px_rgba(16,185,129,0.6)]"
          >
            <Sparkles className="h-3.5 w-3.5" /> Ask
          </button>
        )}
      </div>
      <div className="flex items-center justify-between mt-1.5 px-1">
        <div className="flex items-center gap-3 min-w-0">
          {think && !thinkSupported && (
            <span className="text-[10px] text-amber-400/80 font-mono">
              on* — {model || "this model"} has no reasoning phase; Thinking won't take effect
            </span>
          )}
          <p className="text-[10px] text-slate-500 font-mono">
            <kbd className="px-1 py-px rounded bg-slate-900 border border-white/5 text-slate-400">Enter</kbd>
            {" "}to send ·{" "}
            <kbd className="px-1 py-px rounded bg-slate-900 border border-white/5 text-slate-400">Shift</kbd>
            {" + "}
            <kbd className="px-1 py-px rounded bg-slate-900 border border-white/5 text-slate-400">Enter</kbd>
            {" "}for newline
          </p>
        </div>
        {error && <p className="text-[10px] text-red-400 font-mono">{error}</p>}
      </div>
    </div>
  );
}

function ChatBubble({ msg }: { msg: ChatMsg }) {
  if (msg.role === "user") {
    return (
      <div className="flex gap-3">
        <div className="bg-emerald-500/10 text-emerald-400 h-5 w-5 font-bold rounded flex items-center justify-center border border-emerald-500/20 text-[10px]">
          <User className="h-3 w-3" />
        </div>
        <div className="space-y-0.5 min-w-0 flex-1">
          <p className="text-slate-400 text-[10px] font-mono">Grounded Input Query</p>
          <p className="text-slate-200 font-medium text-sm whitespace-pre-wrap">{msg.content}</p>
        </div>
      </div>
    );
  }
  return (
    <div className="flex gap-3">
      <div className="bg-purple-500/10 text-purple-400 h-5 w-5 rounded flex items-center justify-center border border-purple-500/20 text-[10px]">
        <Bot className="h-3 w-3" />
      </div>
      <div className="space-y-3 w-full min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-slate-400 text-[10px] font-mono">CGX Response</p>
          {msg.intent?.mode && <Pill tone="purple">{msg.intent.mode}</Pill>}
          {msg.intent?.scope && <Pill tone="slate">scope: {msg.intent.scope}</Pill>}
        </div>
        {msg.thought && (
          <div className="rounded-lg border border-muted bg-slate-950/60 p-3 text-[11px] text-slate-400 font-mono whitespace-pre-wrap leading-relaxed">
            <div className="flex items-center gap-2 mb-1 text-[10px] uppercase tracking-wider text-slate-500">
              <span className="av-dot" /> thinking
            </div>
            {msg.thought}
          </div>
        )}
        {msg.warning && (
          <p className="text-[10px] text-amber-400/80 bg-amber-500/5 px-2 py-1 rounded border border-amber-500/10 font-mono">
            <TriangleAlert className="inline h-3 w-3 mr-1" />
            {msg.warning}
          </p>
        )}
        {msg.content ? (
          <Markdown text={msg.content} />
        ) : msg.streaming ? (
          <div className="text-[11px] text-slate-500 font-mono flex items-center gap-2">
            <span className="av-dot" /> retrieving sources & drafting…
          </div>
        ) : null}
      </div>
    </div>
  );
}

const PAGE_SIZE = 10;

function RetrievalPanel({
  sources, intent, onClose,
}: {
  sources: any[];
  intent?: { mode?: string; scope?: string };
  onClose?: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  useEffect(() => setShowAll(false), [sources]);

  const visible = showAll ? sources : sources.slice(0, PAGE_SIZE);
  const hiddenCount = sources.length - PAGE_SIZE;

  return (
    <div className="w-72 bg-surface rounded-xl border border-muted flex flex-col flex-shrink-0">
      <div
        className="p-3 border-b bg-slate-950/80 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-slate-400 font-mono"
        style={{ borderColor: "rgba(255,255,255,0.06)" }}
      >
        <span className="flex items-center gap-1.5">
          <Brain className="h-3 w-3 text-emerald-400" /> Retrieval Ranks
        </span>
        <span className="flex items-center gap-2">
          <span className="text-[9px] text-slate-500 lowercase">fused via rrf</span>
          {onClose && (
            <button
              onClick={onClose}
              className="text-slate-500 hover:text-slate-200 transition"
              title="Hide retrieval ranks"
            >
              <PanelRightClose className="h-3.5 w-3.5" />
            </button>
          )}
        </span>
      </div>
      <div className="flex-1 p-3 font-mono text-[10px] space-y-1.5 overflow-y-auto bg-slate-950/20 text-slate-400">
        {(intent?.mode || intent?.scope) && (
          <p className="text-emerald-300/80 text-[10px] mb-2">
            {intent?.mode && (
              <>
                intent: <span className="text-emerald-400">{intent.mode}</span>
              </>
            )}
            {intent?.mode && intent?.scope && <span className="text-slate-600"> · </span>}
            {intent?.scope && (
              <>
                scope: <span className="text-slate-300">{intent.scope}</span>
              </>
            )}
          </p>
        )}
        {sources.length === 0 && (
          <p className="text-slate-500 italic">Sources will populate after the first answer.</p>
        )}
        {visible.map((s, i) => (
          <SourceRow key={i} src={s} rank={i + 1} />
        ))}
        {!showAll && hiddenCount > 0 && (
          <button
            onClick={() => setShowAll(true)}
            className="w-full flex items-center justify-center gap-1 mt-1 py-1.5 text-[9px] text-slate-500 hover:text-slate-300 border border-white/5 rounded transition-colors"
          >
            <ChevronDown className="h-2.5 w-2.5" />
            {hiddenCount} more
          </button>
        )}
      </div>
    </div>
  );
}

function kindShort(kind: string): string {
  if (!kind) return "";
  if (kind === "function") return "fn";
  if (kind === "class") return "cls";
  if (kind === "method") return "mth";
  if (kind === "module") return "mod";
  return kind.slice(0, 3);
}

function SourceRow({ src, rank }: { src: any; rank: number }) {
  const symbol = src?.symbol || src?.name || "";
  const filePath: string = src?.path || src?.file || src?.source || "";
  const basename = filePath ? filePath.split("/").pop() || filePath : "";
  const kind = src?.kind || "";
  const kindLabel = kindShort(kind);
  const startLine: number | undefined = src?.start_line;
  const endLine: number | undefined = src?.end_line;
  const lineRange = startLine
    ? endLine && endLine !== startLine ? `L${startLine}–${endLine}` : `L${startLine}`
    : "";
  const score: number | null =
    typeof src?.score === "number" ? src.score
      : typeof src?.hit_meta?.score === "number" ? src.hit_meta.score
        : typeof src?.hit_meta?.rrf_score === "number" ? src.hit_meta.rrf_score : null;
  const demoted = !!src?.hit_meta?.scope_demoted;
  const label = symbol || basename || `chunk ${rank}`;
  const sub = [symbol ? basename : "", lineRange].filter(Boolean).join("  ");

  return (
    <div
      className={
        "p-1.5 border rounded space-y-0.5 " +
        (demoted
          ? "border-amber-500/15 bg-amber-500/[0.03] opacity-60"
          : "border-white/5 bg-slate-950/40")
      }
      title={demoted ? "Off-scope: score penalized for the detected query scope" : undefined}
    >
      <div className="flex items-center gap-1.5 min-w-0">
        <span className="text-slate-600 shrink-0 text-[8px]">#{rank}</span>
        {kindLabel && (
          <span className="text-[8px] uppercase tracking-wider font-mono text-emerald-500/70 bg-emerald-500/5 border border-emerald-500/10 px-1 rounded shrink-0">
            {kindLabel}
          </span>
        )}
        {demoted && (
          <span className="text-[8px] uppercase tracking-wider font-mono text-amber-400/80 bg-amber-500/10 border border-amber-500/20 px-1 rounded shrink-0">
            off-scope
          </span>
        )}
        <span className="font-medium text-slate-200 truncate flex-1 text-[10px]">{label}</span>
        {score !== null && (
          <span
            className={
              "text-[9px] shrink-0 " + (demoted ? "text-amber-400/70" : "text-emerald-400")
            }
          >
            {score.toFixed(3)}
          </span>
        )}
      </div>
      {sub && <p className="text-[8px] text-slate-500 truncate pl-4">{sub}</p>}
    </div>
  );
}

function AskEmptyState() {
  return (
    <div className="text-center text-slate-500 text-xs font-mono py-10">
      Drop a question below. CGX will fuse semantic + symbolic + graph retrieval before answering.
    </div>
  );
}
