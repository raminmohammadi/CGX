

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
    """
    outputs: Dict[str, Any] = field(default_factory=dict)
    facts: List[Fact] = field(default_factory=list)
    artifact: Optional[Artifact] = None
    failure: Optional[str] = None


Executor = Callable[[TaskNode, ExecutorDeps], ExecutorResult]


# --------------------- registry ---------------------

_REGISTRY: Dict[TaskKind, Executor] = {}


def register_executor(kind: TaskKind) -> Callable[[Executor], Executor]:
    """Decorator that wires ``fn`` as the executor for ``kind``.

    Re-registering replaces the prior executor; tests rely on that to
    swap in stubs without monkey-patching internal dicts.
    """
    def _wrap(fn: Executor) -> Executor:
        _REGISTRY[kind] = fn
        return fn
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
