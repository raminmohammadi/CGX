

"""Task executors for the session-shaped agent.

Each :class:`~cgx.session.models.TaskKind` has at most one executor
registered here. The :mod:`base` module defines the executor protocol
plus the shared dependency bag; concrete executors live in sibling
modules (``explore.py``, ``ask.py``, ...).

Importing this package side-effect-registers the Phase 1 executors so
:func:`~cgx.session.tasks.base.dispatch` finds them at runtime.
"""

from __future__ import annotations

from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    dispatch,
    get_executor,
    register_executor,
)

# Side-effect imports -- each module decorates its executor with
# :func:`register_executor` at import time. Listed explicitly so
# linters don't strip the imports.
from cgx.session.tasks import explore as _explore  # noqa: F401
from cgx.session.tasks import ask as _ask  # noqa: F401
from cgx.session.tasks import investigate as _investigate  # noqa: F401
from cgx.session.tasks import recommend as _recommend  # noqa: F401
from cgx.session.tasks import plan_change as _plan_change  # noqa: F401
from cgx.session.tasks import apply as _apply  # noqa: F401
from cgx.session.tasks import verify as _verify  # noqa: F401
from cgx.session.tasks import clarify_requirements as _clarify_requirements  # noqa: F401
from cgx.session.tasks import decompose as _decompose  # noqa: F401
from cgx.session.tasks import scaffold as _scaffold  # noqa: F401

__all__ = [
    "ExecutorDeps",
    "ExecutorResult",
    "dispatch",
    "get_executor",
    "register_executor",
]
