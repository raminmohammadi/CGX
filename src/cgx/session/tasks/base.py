

"""Executor protocol + registry for session task kinds.

An *executor* is a pure function that takes a :class:`TaskNode` plus a
shared :class:`ExecutorDeps` bag and returns an :class:`ExecutorResult`
describing what it produced. Executors do not write to the
:class:`SessionStore` directly -- the :class:`SessionRunner` persists
their outputs, facts, and artifacts after the call. This keeps
executors easy to unit-test without a database and gives the runner a
single place to enforce ordering / event emission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cgx.session.models import Artifact, Fact, TaskKind, TaskNode
from cgx.trace import traced


@dataclass
class ExecutorDeps:
    """Shared runtime dependencies passed to every executor.

    The fields are deliberately ``Optional`` so unit tests can build a
    minimal :class:`ExecutorDeps` without spinning up a real LLM /
    index. Executors are responsible for validating the fields they
    need and raising a clear error if a required dep is missing.
    """
    project_root: Optional[str] = None
    index_dir: Optional[str] = None
    records_path: Optional[str] = None
    embed_model: Optional[str] = None
    provider: Any = None  # cgx.providers.LLMProvider; typed Any to avoid import
    store: Any = None  # cgx.session.store.SessionStore; typed Any to avoid cycle
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutorResult:
    """What an executor produces.

    The runner persists each piece in turn:

    * ``outputs`` -> ``TaskNode.outputs``
    * ``facts`` -> :meth:`SessionStore.add_fact` (each gets a
      ``surfaced_in_task_id`` matching the task).
    * ``artifact`` -> :meth:`SessionStore.save_artifact` and
      :attr:`TaskNode.produced_artifact_id`.
    * ``failure`` non-empty -> task transitions to ``FAILED`` instead
      of ``DONE``; the runner still persists any facts the executor
      surfaced before erroring.
    * ``retryable`` True (only meaningful with ``failure``) -> the
      failure is a *plan-quality* problem an LLM retry could fix (e.g.
      a manifest that failed validation), not a hard environment/crash
      error. The router may re-queue the task once with the failure
      message folded into its goal as a constraint instead of ending
      the session terminally.
    """
    outputs: Dict[str, Any] = field(default_factory=dict)
    facts: List[Fact] = field(default_factory=list)
    artifact: Optional[Artifact] = None
    failure: Optional[str] = None
    retryable: bool = False


Executor = Callable[[TaskNode, ExecutorDeps], ExecutorResult]


# --------------------- registry ---------------------

_REGISTRY: Dict[TaskKind, Executor] = {}


def register_executor(kind: TaskKind) -> Callable[[Executor], Executor]:
    """Decorator that wires ``fn`` as the executor for ``kind``.

    Re-registering replaces the prior executor; tests rely on that to
    swap in stubs without monkey-patching internal dicts. The registered
    function is wrapped in :func:`cgx.trace.traced` so every executor
    invocation surfaces an ``enter``/``exit`` record in the project
    agent log when the global trace toggle is on.
    """
    def _wrap(fn: Executor) -> Executor:
        wrapped = traced("executor")(fn)
        _REGISTRY[kind] = wrapped
        return wrapped
    return _wrap


def get_executor(kind: TaskKind) -> Optional[Executor]:
    return _REGISTRY.get(kind)


def dispatch(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Look up the executor for ``task.kind`` and run it.

    Raises :class:`LookupError` when no executor is registered -- the
    runner catches that and transitions the task to ``FAILED`` with a
    helpful message rather than crashing the request.
    """
    fn = _REGISTRY.get(task.kind)
    if fn is None:
        raise LookupError(
            f"no executor registered for task kind {task.kind.value!r}")
    return fn(task, deps)


def session_skills(task: TaskNode, deps: ExecutorDeps) -> Optional[List[str]]:
    """Return the owning session's explicit skill list, or ``None``.

    ``None`` (not ``[]``) is returned when the session has no skills set
    so callers can pass it straight through to
    ``cgx.answer.engine``'s ``skills=`` kwarg and get its existing
    auto-detect-from-goal-text fallback for free.
    """
    if deps.store is None:
        return None
    try:
        session = deps.store.get_session(task.session_id)
    except Exception:
        return None
    if session is None or not session.skills:
        return None
    return list(session.skills)
