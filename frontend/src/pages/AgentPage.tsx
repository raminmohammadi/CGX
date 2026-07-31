import { useCallback, useEffect, useRef, useState } from "react";
import {
  api, ApiError,
  type AgentSessionState, type AgentSessionSummary, type TaskProgress,
} from "../lib/api";
import { useWorkspace } from "../store/workspace";
import { useAgentSession } from "../store/agentSession";
import { SessionLauncher } from "../components/agent/SessionLauncher";
import { LiveView, PriorSessions, activeTask } from "../components/agent/LiveView";

// Stateful, session-shaped Agent page (Phase 4). The legacy batch view
// still lives at ``/agent-legacy`` -- this one drives the new session
// runner under ``/api/agent-session``.
export default function AgentPage() {
  const { provider, index, projectRoot, setProjectRoot } = useWorkspace();
  const {
    activeId, setActiveId, selectedTaskId, setSelectedTaskId,
    runModels, setRunModel,
  } = useAgentSession();

  const [state, setState] = useState<AgentSessionState | null>(null);
  const [sessions, setSessions] = useState<AgentSessionSummary[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reply, setReply] = useState("");
  const [progress, setProgress] = useState<Record<string, TaskProgress>>({});
  const pollRef = useRef<number | null>(null);
  // True while the SSE stream is healthy; the poll below only fires as a
  // fallback when the stream is down so the two never double-fetch.
  const sseOkRef = useRef(false);
  const refetchTimer = useRef<number | null>(null);

  const refreshSessions = useCallback(async () => {
    try {
      const list = await api.agentSessionList(projectRoot || null);
      setSessions(list);
    } catch { /* sidebar is best-effort */ }
  }, [projectRoot]);

  const loadState = useCallback(async (sid: string) => {
    try {
      const s = await api.agentSessionGet(sid, projectRoot || null);
      setState(s);
      setError(null);
    } catch (e) {
      // A 404 on the active id means the persisted localStorage entry
      // points at a session that no longer exists (project root changed,
      // db recreated, session deleted out-of-band). Clear the stale id
      // and refresh the sidebar so the launcher takes over instead of
      // re-firing the same 404 on every reload.
      if (e instanceof ApiError && e.status === 404) {
        setActiveId(null);
        setSelectedTaskId(null);
        setState(null);
        setError(null);
        void refreshSessions();
        return;
      }
      setError(String((e as Error)?.message || e));
    }
  }, [projectRoot, refreshSessions, setActiveId, setSelectedTaskId]);

  useEffect(() => { refreshSessions(); }, [refreshSessions]);
  useEffect(() => {
    if (activeId) loadState(activeId);
    else setState(null);
  }, [activeId, loadState]);

  // Live updates over SSE: the backend drains the pipeline in the
  // background and streams events here. State-changing events trigger a
  // (debounced) authoritative refetch; ``task.output_partial`` carries
  // transient per-file scaffold progress that never lands in a snapshot.
  useEffect(() => {
    if (!activeId) return;
    setProgress({});
    sseOkRef.current = false;
    const es = new EventSource(api.agentSessionEventsUrl(activeId));
    const scheduleRefetch = () => {
      if (refetchTimer.current) return;
      refetchTimer.current = window.setTimeout(() => {
        refetchTimer.current = null;
        loadState(activeId);
      }, 250);
    };
    es.addEventListener("snapshot", (e: MessageEvent) => {
      sseOkRef.current = true;
      try { setState(JSON.parse(e.data)); } catch { /* ignore */ }
    });
    es.addEventListener("task.output_partial", (e: MessageEvent) => {
      sseOkRef.current = true;
      try {
        const ev = JSON.parse(e.data);
        const p = ev?.payload?.progress;
        const tid = ev?.payload?.task_id;
        if (tid && p) setProgress((prev) => ({ ...prev, [tid]: p }));
      } catch { /* ignore */ }
    });
    [
      "session.updated", "task.created", "task.status_changed",
      "task.completed", "task.failed", "decision.recorded",
      "fact.added", "fact.stale", "artifact.created",
    ].forEach((name) => es.addEventListener(name, () => {
      sseOkRef.current = true;
      scheduleRefetch();
    }));
    es.onerror = () => { sseOkRef.current = false; };
    return () => {
      es.close();
      if (refetchTimer.current) {
        window.clearTimeout(refetchTimer.current);
        refetchTimer.current = null;
      }
    };
  }, [activeId, loadState]);

  // Fallback poll: only runs while a task is in-flight *and* the SSE
  // stream is unavailable, so a dropped stream still converges.
  useEffect(() => {
    if (!activeId || !state) return;
    const hasInFlight = state.tasks.some(
      (t) => t.kind !== "ask_user"
        && (t.status === "in_progress" || t.status === "ready"),
    );
    if (!hasInFlight || sseOkRef.current) {
      if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
      return;
    }
    if (pollRef.current) return;
    pollRef.current = window.setInterval(() => { loadState(activeId); }, 2500);
    return () => {
      if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [state, activeId, loadState]);

  const createSession = useCallback(async (opts: {
    objective: string; projectRoot: string;
    mode: "explore" | "greenfield" | null;
  }) => {
    setPending(true); setError(null);
    try {
      if (opts.projectRoot && opts.projectRoot !== projectRoot) setProjectRoot(opts.projectRoot);
      const next = await api.agentSessionCreate({
        objective: opts.objective,
        project_root: opts.projectRoot || null,
        title: opts.objective.slice(0, 80),
        mode: opts.mode,
        index, provider,
        run_initial_task: true,
      });
      setState(next);
      setRunModel(next.session.session_id, provider.model);
      setActiveId(next.session.session_id);
      setSelectedTaskId(next.session.root_task_id);
      await refreshSessions();
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, [index, provider, projectRoot, setProjectRoot, refreshSessions,
      setActiveId, setSelectedTaskId, setRunModel]);

  const deleteSession = useCallback(async (sid: string) => {
    setError(null);
    try {
      await api.agentSessionDelete(sid, projectRoot || null);
      if (activeId === sid) {
        setActiveId(null);
        setSelectedTaskId(null);
        setState(null);
      }
      await refreshSessions();
    } catch (e) {
      setError(String((e as Error)?.message || e));
    }
  }, [activeId, projectRoot, refreshSessions, setActiveId, setSelectedTaskId]);

  const postDecision = useCallback(async (taskId: string,
      payload: { chosen: Record<string, any>; rationale?: string }) => {
    if (!activeId) return;
    setPending(true); setError(null);
    try {
      const next = await api.agentSessionDecision(activeId, {
        task_id: taskId, chosen: payload.chosen,
        rationale: payload.rationale ?? null,
        index, provider, run_initial_task: true,
      });
      setRunModel(activeId, provider.model);
      setState(next);
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, [activeId, index, provider, setRunModel]);

  const postMessage = useCallback(async () => {
    if (!activeId || !reply.trim()) return;
    setPending(true); setError(null);
    try {
      const next = await api.agentSessionMessage(activeId, {
        message: reply.trim(),
        index, provider, run_initial_task: true,
      });
      setRunModel(activeId, provider.model);
      setState(next); setReply("");
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, [activeId, index, provider, reply, setRunModel]);

  // Cooperative cancel (P2.2): ask the backend to stop the drain after
  // the current task. SSE then converges the snapshot to the stopped
  // state, so we only need the request here.
  const cancelSession = useCallback(async () => {
    if (!activeId) return;
    setError(null);
    try {
      const next = await api.agentSessionCancel(activeId);
      setState(next);
    } catch (e) {
      setError(String((e as Error)?.message || e));
    }
  }, [activeId]);

  if (!activeId || !state) {
    return (
      <div className="flex h-full w-full overflow-hidden">
        <div className="flex-1 overflow-y-auto">
          <SessionLauncher
            defaultProjectRoot={projectRoot}
            onCreate={createSession}
            pending={pending}
            error={error}
          />
          <PriorSessions
            sessions={sessions}
            onPick={(sid) => { setActiveId(sid); setSelectedTaskId(null); }}
            onDelete={deleteSession}
          />
        </div>
      </div>
    );
  }
  const running = !!state && state.tasks.some(
    (t) => t.kind !== "ask_user"
      && (t.status === "in_progress" || t.status === "ready"),
  );

  return (
    <LiveView
      state={state} sessions={sessions} pending={pending} error={error}
      progress={progress} running={running}
      runModel={runModels[activeId] ?? null} selectedModel={provider.model}
      reply={reply} setReply={setReply}
      selectedTaskId={selectedTaskId} setSelectedTaskId={setSelectedTaskId}
      onDecide={(payload) => {
        const taskId = selectedTaskId
          ?? activeTask(state)?.task_id ?? "";
        if (taskId) return postDecision(taskId, payload);
      }}
      onSend={postMessage}
      onCancel={cancelSession}
      onSwitch={(sid) => { setActiveId(sid); setSelectedTaskId(null); }}
      onNew={() => { setActiveId(null); setSelectedTaskId(null); setState(null); }}
      onRefresh={() => loadState(activeId)}
      onDelete={deleteSession}
    />
  );
}


