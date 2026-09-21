"""SWARM_ASSESS executor: is an existing repo the right place to build?

When the Swarm is pointed at a NON-empty ``project_root`` this runs first and
judges whether the tree on disk is relevant to the user's objective:

* **relevant** -> proceed to the Tech Lead, which plans against (and the
  Developer modifies) the existing code.
* **not relevant** -> the router raises an ASK_USER(RELOCATE) so the user can
  point the build at a fresh folder instead of scaffolding an unrelated project
  on top of real code.

The check is deliberately conservative: it only diverts to the relocation
prompt on a confident "unrelated" verdict, and any model/parse failure defaults
to *relevant* (proceed) so a flaky judgment never blocks a legitimate build.
The relevance judgment is grounded on a bounded snapshot of the tree (top-level
layout + a sample of source file paths), never the whole repo.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from cgx.session.mode import _IGNORE_DIRS
from cgx.session.models import TaskKind
from cgx.session.tasks.base import (
    ExecutorDeps, ExecutorResult, TaskNode, register_executor)
from cgx.session.tasks.swarm_log import swarm_beat

_MAX_LISTED_FILES = 60
_SRC_EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
            ".rb", ".php", ".vue", ".md", ".toml", ".cfg", ".json", ".txt")


def snapshot_repo(root: str, *, limit: int = _MAX_LISTED_FILES) -> List[str]:
    """A bounded, relative file listing of a repo for the relevance judge.

    Walks ``root`` skipping the standard ignored dirs (``.git``, ``node_modules``,
    ``.venv``, ``.cgx``/``.cgx-backups`` …), collects source/config/doc files,
    and caps the count so a huge tree can't blow the prompt.
    """
    out: List[str] = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _IGNORE_DIRS and not d.startswith(".")]
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            if fn.lower().endswith(_SRC_EXT):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                out.append(rel)
                if len(out) >= limit:
                    return out
    return out


_SYSTEM = (
    "You judge whether an EXISTING code repository is the right place to carry "
    "out a user's build objective. Answer with a single JSON object and nothing "
    "else: {\"relevant\": true|false, \"reason\": \"one sentence\"}.\n"
    "relevant=true means the repo's existing code is on the same topic as the "
    "objective, so the work should MODIFY/extend it. relevant=false means the "
    "repo is about something unrelated, so building here would scaffold an "
    "unrelated project on top of real code -- the user should pick a fresh "
    "folder. When genuinely unsure, answer relevant=true."
)


@register_executor(TaskKind.SWARM_ASSESS)
def swarm_assess(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Judge existing-repo relevance; default to 'relevant' on any uncertainty."""
    goal = str(task.inputs.get("goal") or "")
    project_root = (task.inputs.get("project_root")
                    or deps.project_root or ".")
    project_root = os.path.abspath(project_root)

    files = snapshot_repo(project_root)
    # An empty snapshot shouldn't happen (router only routes here for a
    # non-empty tree) but if it does, there's nothing to be irrelevant to.
    if not files:
        return ExecutorResult(outputs={"relevant": True,
                                       "reason": "empty tree",
                                       "project_root": project_root,
                                       "goal": goal})

    summary = "\n".join(f"- {p}" for p in files)
    swarm_beat(project_root, "assess", "scan", files=len(files), goal=goal[:200])

    relevant, reason = True, "assessment unavailable; proceeding"
    if deps.provider is not None:
        user = (f"OBJECTIVE:\n{goal}\n\nEXISTING REPO FILES:\n{summary}\n\n"
                "Is this repo relevant to the objective?")
        try:
            res = deps.provider.chat(
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": user}],
                force_json=True)
            parsed = _parse_verdict(str(res.get("content") or ""))
            if parsed is not None:
                relevant = bool(parsed.get("relevant", True))
                reason = str(parsed.get("reason") or "").strip() or reason
        except Exception:  # pragma: no cover - judge is best-effort
            relevant = True

    swarm_beat(project_root, "assess", "verdict", relevant=relevant,
               reason=reason[:200])
    return ExecutorResult(outputs={
        "relevant": relevant,
        "reason": reason,
        "repo_summary": summary,
        "project_root": project_root,
        "goal": goal,
        # Threaded so the downstream Tech Lead keeps the session's approval mode.
        "require_plan_approval": bool(task.inputs.get("require_plan_approval")),
    })


def _parse_verdict(text: str) -> Dict[str, Any] | None:
    """Extract the ``{relevant, reason}`` object from a possibly-fenced reply."""
    t = (text or "").strip()
    if not t:
        return None
    if "```" in t:
        # Strip a ```json fence if present.
        import re
        m = re.search(r"```[a-zA-Z]*\s*(.*?)\s*```", t, re.DOTALL)
        if m:
            t = m.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        # Last resort: a bare "relevant": false anywhere in the text.
        low = t.lower()
        if '"relevant"' in low:
            return {"relevant": "false" not in low.split('"relevant"', 1)[1][:12]}
        return None
