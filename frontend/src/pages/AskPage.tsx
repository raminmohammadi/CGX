import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bot, Brain, ChevronDown, Send, Square, TriangleAlert, User } from "lucide-react";
import { api, type SessionMessage } from "../lib/api";
import { streamSSE } from "../lib/sse";
import { abortConnection, getConnection, setConnection } from "../lib/connections";
import { useTasks, type ChatMsg } from "../store/tasks";
import { useWorkspace } from "../store/workspace";
import { Markdown } from "../components/Markdown";
import { Pill } from "../components/Pill";

const PAGE_KEY = "ask";

export default function AskPage() {
  const { provider, index, selectedSessionId, setSelectedSession } = useWorkspace();
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

  // Load session messages when sidebar picks a session.
  useEffect(() => {
    let alive = true;
    (async () => {
      if (!selectedSessionId) { resetAsk(); return; }
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
        setAsk({ messages: conv });
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
          patchLast({ intent: { mode: String(data?.mode || "") } });
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

  return (
    <div className="flex h-full w-full p-4 gap-4 overflow-hidden">
      <RetrievalPanel sources={lastAssistant?.sources || []} intent={lastAssistant?.intent} />

      <div className="flex-1 flex flex-col bg-surface rounded-xl border border-muted overflow-hidden">
        <div ref={threadRef} className="flex-1 p-5 overflow-y-auto space-y-5 text-xs">
          {messages.length === 0 && <AskEmptyState />}
          {messages.map((m, i) => (
            <ChatBubble key={i} msg={m} />
          ))}
        </div>

        <div className="p-3 border-t border-muted bg-slate-950">
          <div className="rounded-lg bg-slate-900 border border-muted focus-within:border-emerald-500/50 focus-within:shadow-neon transition flex items-end gap-2 p-1.5">
            <textarea
              ref={draftRef}
              defaultValue=""
              onKeyDown={handleKey}
              rows={1}
              placeholder="Query code signatures, modules, or implementations..."
              className="w-full bg-transparent outline-none text-xs text-white px-2 py-1.5 resize-none placeholder-slate-500 max-h-32"
            />
            {busy ? (
              <button onClick={cancel} className="av-btn-ghost shrink-0">
                <Square className="h-3 w-3" /> Stop
              </button>
            ) : (
              <button onClick={send} className="av-btn-primary shrink-0">
                <Send className="h-3 w-3" /> Ask
              </button>
            )}
          </div>
          {error && <p className="text-[10px] text-red-400 mt-2 font-mono">{error}</p>}
        </div>
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
          <p className="text-slate-400 text-[10px] font-mono">Averix Response</p>
          {msg.intent?.mode && <Pill tone="purple">{msg.intent.mode}</Pill>}
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

function RetrievalPanel({ sources, intent }: { sources: any[]; intent?: { mode?: string } }) {
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
        <span className="text-[9px] text-slate-500 lowercase">fused via rrf</span>
      </div>
      <div className="flex-1 p-3 font-mono text-[10px] space-y-1.5 overflow-y-auto bg-slate-950/20 text-slate-400">
        {intent?.mode && (
          <p className="text-emerald-300/80 text-[10px] mb-2">
            intent: <span className="text-emerald-400">{intent.mode}</span>
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
      : typeof src?.hit_meta?.rrf_score === "number" ? src.hit_meta.rrf_score : null;
  const label = symbol || basename || `chunk ${rank}`;
  const sub = [symbol ? basename : "", lineRange].filter(Boolean).join("  ");

  return (
    <div className="p-1.5 border border-white/5 rounded bg-slate-950/40 space-y-0.5">
      <div className="flex items-center gap-1.5 min-w-0">
        <span className="text-slate-600 shrink-0 text-[8px]">#{rank}</span>
        {kindLabel && (
          <span className="text-[8px] uppercase tracking-wider font-mono text-emerald-500/70 bg-emerald-500/5 border border-emerald-500/10 px-1 rounded shrink-0">
            {kindLabel}
          </span>
        )}
        <span className="font-medium text-slate-200 truncate flex-1 text-[10px]">{label}</span>
        {score !== null && (
          <span className="text-[9px] text-emerald-400 shrink-0">{score.toFixed(3)}</span>
        )}
      </div>
      {sub && <p className="text-[8px] text-slate-500 truncate pl-4">{sub}</p>}
    </div>
  );
}

function AskEmptyState() {
  return (
    <div className="text-center text-slate-500 text-xs font-mono py-10">
      Drop a question below. Averix will fuse semantic + symbolic + graph retrieval before answering.
    </div>
  );
}
