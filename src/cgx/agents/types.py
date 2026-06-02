# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""Shared types for the agent orchestration layer.

All data types here are plain dataclasses so they JSON-serialise cleanly
for persistence (session history) and for the UI event stream.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class TaskStatus(str, enum.Enum):
    """Lifecycle states for a :class:`Task`.

    Using ``str`` as a mixin keeps the value JSON-serialisable without a
    custom encoder.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskKind(str, enum.Enum):
    """Capability dispatched to satisfy a task.

    The kinds intentionally mirror the existing high-level entry points
    so the agent layer doesn't reinvent retrieval or codegen.
    """

    ASK = "ask"                         # routes to cgx.answer.engine.answer_with_llm
    PLAN = "plan"                       # routes to cgx.answer.engine.generate_code_plan
    SCAFFOLD = "scaffold"               # routes to cgx.answer.engine.generate_project_scaffold (new project from scratch)
    SCAFFOLD_MANIFEST = "scaffold_manifest"  # lightweight call: returns file list only, injects SCAFFOLD_FILE tasks
    SCAFFOLD_FILE = "scaffold_file"     # generates exactly one file given spec + prior context
    SEARCH = "search"                   # routes to cgx.pipeline.auto.run_query_auto
    SUMMARIZE = "summarize"             # short, LLM-driven summarisation of prior outputs
    APPLY = "apply"                     # write a prior PLAN/SCAFFOLD's files to disk (+ smoke test)
    VERIFY = "verify"                   # run impacted pytest tests against the modified tree
    FILL_LOGIC = "fill_logic"           # fill empty function bodies in a skeleton file (skeleton-and-fill phase 2)


@dataclass
class Task:
    """A single, atomic unit of work in an agent :class:`Plan`."""

    description: str
    kind: TaskKind = TaskKind.ASK
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""  # short human title; falls back to description when empty
    inputs: Dict[str, Any] = field(default_factory=dict)
    criteria: List[str] = field(default_factory=list)  # plain-English checks for the Judge
    status: TaskStatus = TaskStatus.PENDING
    output: Optional[Dict[str, Any]] = None
    judge: Optional[Dict[str, Any]] = None  # filled by Judge after execution
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)  # task IDs this task depends on (for DAG)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        return d


@dataclass
class Plan:
    """An ordered sequence of tasks plus the original request."""

    goal: str
    tasks: List[Task] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    #: Planner-supplied free-text explanation of *why* the plan is
    #: structured this way. Surfaced in the SSE ``plan`` event payload
    #: so the UI can render a "Plan Rationale" card above the DAG.
    rationale: str = ""
    #: File manifest populated by the Tracker after each APPLY task.
    #: Maps project-relative path → status: "applied" | "failed".
    #: Lets the retry loop identify which files are already on disk and
    #: which ones still need to be fixed, so it can issue targeted prompts.
    owned_files: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "created_at": self.created_at,
            "tasks": [t.to_dict() for t in self.tasks],
            "rationale": self.rationale,
            "owned_files": dict(self.owned_files),
        }

    def summary_lines(self) -> List[str]:
        """Compact one-line-per-task view, used by Tracker.render_plan()."""
        lines = []
        for i, t in enumerate(self.tasks, 1):
            marker = {
                TaskStatus.PENDING: "•",
                TaskStatus.RUNNING: "▶",
                TaskStatus.DONE: "✔",
                TaskStatus.FAILED: "✖",
                TaskStatus.SKIPPED: "⊝",
            }[t.status]
            lines.append(f"{marker} **[{t.kind.value}]** {t.description}")
        return lines


@dataclass
class AgentEvent:
    """Streaming event emitted by the Tracker as it executes a plan.

    The UI subscribes to these to render a live execution graph and a
    thought-process panel without polling the full ``Plan`` state.
    """

    type: str   # "plan", "task_start", "task_progress", "task_done", "task_failed", "judge", "summary"
    payload: Dict[str, Any]
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "payload": self.payload, "at": self.at}
