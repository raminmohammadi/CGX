"""Averix multi-agent orchestration.

Three logical roles cooperating on a user goal:

* :class:`~cgx.agents.planner.Planner` — decomposes a natural-language
  request into an ordered list of atomic :class:`~cgx.agents.types.Task`
  objects, each assigned a ``kind`` from
  ``{ask, plan, scaffold, search, summarize, apply, verify}``.
* :class:`~cgx.agents.tracker.Tracker` — state machine that executes
  each task by dispatching to a capability callable and emits
  :class:`~cgx.agents.types.AgentEvent` updates.
* :class:`~cgx.agents.judge.Judge` — validates produced artifacts against
  task-level criteria before declaring a step complete.

Task kinds:

* ``ask``      — answer a question grounded in the indexed code.
* ``plan``     — produce a unified-diff change plan for an *existing* codebase.
* ``scaffold`` — generate a **complete new project** from a plain-language idea
  (no existing index required).
* ``search``   — retrieve relevant code chunks from the index.
* ``summarize``— condense the outputs of prior tasks into a brief summary.
* ``apply``    — write the diffs produced by ``plan``/``scaffold`` to disk.
* ``verify``   — run impacted pytest tests against the modified tree.

The :func:`~cgx.agents.loop.run_agent` helper wires the three together so
the UI / CLI can stream a live "thought process" of plan creation,
execution, and validation.
"""

from cgx.agents.types import AgentEvent, Plan, Task, TaskStatus  # noqa: F401
from cgx.agents.planner import Planner  # noqa: F401
from cgx.agents.judge import Judge  # noqa: F401
from cgx.agents.tracker import Tracker  # noqa: F401
from cgx.agents.loop import run_agent  # noqa: F401

__all__ = [
    "AgentEvent",
    "Plan",
    "Task",
    "TaskStatus",
    "Planner",
    "Judge",
    "Tracker",
    "run_agent",
]
