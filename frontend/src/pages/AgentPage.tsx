import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type AgentSessionState, type AgentSessionSummary,
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
  } = useAgentSession();

  const [state, setState] = useState<AgentSessionState | null>(null);
  const [sessions, setSessions] = useState<AgentSessionSummary[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reply, setReply] = useState("");
  const pollRef = useRef<number | null>(null);

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
      setError(String((e as Error)?.message || e));
    }
  }, [projectRoot]);

  useEffect(() => { refreshSessions(); }, [refreshSessions]);
  useEffect(() => {
    if (activeId) loadState(activeId);
    else setState(null);
  }, [activeId, loadState]);

  // Poll while any task is in-flight (running executors, not human asks).
  useEffect(() => {
    if (!activeId || !state) return;
    const hasInFlight = state.tasks.some(
      (t) => t.kind !== "ask_user"
        && (t.status === "in_progress" || t.status === "ready"),
    );
    if (!hasInFlight) {
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
      setActiveId(next.session.session_id);
      setSelectedTaskId(next.session.root_task_id);
      await refreshSessions();
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, [index, provider, projectRoot, setProjectRoot, refreshSessions,
      setActiveId, setSelectedTaskId]);

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
      setState(next);
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, [activeId, index, provider]);

  const postMessage = useCallback(async () => {
    if (!activeId || !reply.trim()) return;
    setPending(true); setError(null);
    try {
      const next = await api.agentSessionMessage(activeId, {
        message: reply.trim(),
        index, provider, run_initial_task: true,
      });
      setState(next); setReply("");
    } catch (e) {
      setError(String((e as Error)?.message || e));
    } finally { setPending(false); }
  }, [activeId, index, provider, reply]);

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
  return (
    <LiveView
      state={state} sessions={sessions} pending={pending} error={error}
      reply={reply} setReply={setReply}
      selectedTaskId={selectedTaskId} setSelectedTaskId={setSelectedTaskId}
      onDecide={(payload) => {
        const taskId = selectedTaskId
          ?? activeTask(state)?.task_id ?? "";
        if (taskId) return postDecision(taskId, payload);
      }}
      onSend={postMessage}
      onSwitch={(sid) => { setActiveId(sid); setSelectedTaskId(null); }}
      onNew={() => { setActiveId(null); setSelectedTaskId(null); setState(null); }}
      onRefresh={() => loadState(activeId)}
      onDelete={deleteSession}
    />
  );
}


