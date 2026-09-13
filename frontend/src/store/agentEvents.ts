import type {
  AgentSessionState, ArtifactDTO, DecisionDTO, FactDTO, TaskNodeDTO,
} from "../lib/api";

// Event-sourced state application for the live agent view.
//
// The SSE stream carries the full changed entity in every frame (the backend
// emits ``task.to_dict()`` / ``fact.to_dict()`` / etc), so the UI can mutate
// exactly the row that changed instead of throwing the payload away and
// refetching the whole session snapshot on every beat. That refetch-per-event
// design is what made the running view feel like it "reloaded" and re-stacked
// its lists; applying the payload in place lets a node animate its own
// transition. A debounced full refetch still runs as a reconciliation fallback
// in case the bounded SSE queue ever drops a frame under a burst.

function upsertById<T extends Record<string, any>>(
  list: T[], item: T, idKey: string,
): T[] {
  const id = item[idKey];
  const idx = list.findIndex((x) => x[idKey] === id);
  if (idx === -1) return [...list, item];
  const next = list.slice();
  next[idx] = item;
  return next;
}

// Apply one named SSE event's payload to the current session state, returning a
// new state object (or the same reference when the event is a no-op) so React's
// referential-equality checks only re-render what actually changed.
export function applySessionEvent(
  prev: AgentSessionState,
  type: string,
  payload: any,
): AgentSessionState {
  if (!payload) return prev;
  switch (type) {
    case "session.created":
    case "session.updated":
      return { ...prev, session: { ...prev.session, ...payload } };

    // TASK_CREATED / TASK_COMPLETED / TASK_FAILED carry the full task dict;
    // TASK_STATUS_CHANGED wraps it under ``task`` alongside from/to.
    case "task.created":
    case "task.completed":
    case "task.failed":
      return { ...prev, tasks: upsertById(prev.tasks, payload as TaskNodeDTO, "task_id") };
    case "task.status_changed": {
      const t = payload.task as TaskNodeDTO | undefined;
      return t ? { ...prev, tasks: upsertById(prev.tasks, t, "task_id") } : prev;
    }

    case "fact.added":
      return { ...prev, facts: upsertById(prev.facts, payload as FactDTO, "fact_id") };
    case "fact.stale": {
      const ids: string[] = payload.fact_ids || [];
      if (!ids.length) return prev;
      const set = new Set(ids);
      return {
        ...prev,
        facts: prev.facts.map((f) => (set.has(f.fact_id) ? { ...f, stale: true } : f)),
      };
    }

    case "artifact.created":
      return { ...prev, artifacts: upsertById(prev.artifacts, payload as ArtifactDTO, "artifact_id") };
    case "decision.recorded":
      return { ...prev, decisions: upsertById(prev.decisions, payload as DecisionDTO, "decision_id") };

    default:
      return prev;
  }
}
