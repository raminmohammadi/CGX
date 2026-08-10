

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
from cgx.session.tasks import runtime_verify as _runtime_verify  # noqa: F401
from cgx.session.tasks import re_verify as _re_verify  # noqa: F401
from cgx.session.tasks import clarify_requirements as _clarify_requirements  # noqa: F401
from cgx.session.tasks import decompose as _decompose  # noqa: F401
from cgx.session.tasks import scaffold as _scaffold  # noqa: F401
from cgx.session.tasks import bootstrap_env as _bootstrap_env  # noqa: F401
from cgx.session.tasks import api_check as _api_check  # noqa: F401
from cgx.session.tasks import smoke as _smoke  # noqa: F401
from cgx.session.tasks import repair as _repair  # noqa: F401
from cgx.session.tasks import diagnose as _diagnose  # noqa: F401
from cgx.session.tasks import ast_scaffold as _ast_scaffold  # noqa: F401
from cgx.session.tasks import swarm_tech_lead as _swarm_tech_lead  # noqa: F401
from cgx.session.tasks import swarm_developer as _swarm_developer  # noqa: F401
from cgx.session.tasks import swarm_verify as _swarm_verify  # noqa: F401

__all__ = [
    "ExecutorDeps",
    "ExecutorResult",
    "dispatch",
    "get_executor",
    "register_executor",
]
