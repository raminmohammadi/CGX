

"""DECOMPOSE executor: turn clarified requirements into a work plan.

Wraps :func:`cgx.answer.engine.plan_scaffold_manifest` so the
greenfield loop produces a typed :class:`Artifact` of kind
``WORK_PLAN`` carrying the file manifest (``plan_md`` + ``layers``) the
downstream ``SCAFFOLD`` executor iterates.

The clarify answers (collected via ASK_USER(CLARIFY_ANSWERS)) are
folded into the goal string so the manifest planner sees the user's
tech-stack / scope decisions in its prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)


@register_executor(TaskKind.DECOMPOSE)
def run_decompose(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Produce a ``WORK_PLAN`` artifact from a clarified objective."""
    if deps.provider is None:
        return ExecutorResult(failure="DECOMPOSE requires an LLM provider")
    if deps.store is None:
        return ExecutorResult(
            failure="DECOMPOSE requires a session store in deps")

    prior_goal = str(task.inputs.get("prior_goal") or "").strip()
    answers = task.inputs.get("answers") or {}
    if not isinstance(answers, dict):
        answers = {}
    questions = _load_questions(task, deps)

    composed_goal = _compose_goal(prior_goal, questions, answers)
    if not composed_goal:
        return ExecutorResult(failure="DECOMPOSE: empty composed goal")

    # Lazy import: the answer engine drags retrieval + prompt builders.
    from cgx.answer.engine import plan_scaffold_manifest

    try:
        result = plan_scaffold_manifest(
            composed_goal, deps.provider, goal=composed_goal)
    except Exception as exc:
        logger.exception("DECOMPOSE: plan_scaffold_manifest crashed")
        return ExecutorResult(
            failure=f"decompose failed: {type(exc).__name__}: {exc}")

    plan_md = str((result or {}).get("plan_md") or "")
    layers = _coerce_layers((result or {}).get("layers"))
    contracts = _coerce_contracts((result or {}).get("contracts"))
    if not _layer_file_count(layers):
        return ExecutorResult(
            failure="DECOMPOSE: planner returned an empty manifest")

    # Deterministic coherence gate: fail early (with an actionable message
    # the router folds into a retry constraint) when the manifest is
    # logically broken, then topologically order files by dependency hints
    # so SCAFFOLD generates dependencies before their consumers.
    coherence_error = _validate_manifest_coherence(layers)
    if coherence_error:
        return ExecutorResult(failure=coherence_error)
    layers = _order_manifest_layers(layers)

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.WORK_PLAN,
        content={
            "prior_goal": prior_goal,
            "composed_goal": composed_goal,
            "answers": dict(answers),
            "plan_md": plan_md,
            "layers": layers,
            "contracts": contracts,
        },
    )
    return ExecutorResult(
        outputs={
            "work_plan_artifact_id": artifact.artifact_id,
            "file_count": _layer_file_count(layers),
            "layer_count": len(layers),
            "contract_count": _contract_entry_count(contracts),
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _load_questions(task: TaskNode,
                    deps: ExecutorDeps) -> List[Dict[str, Any]]:
    """Pull the question list off the upstream REQUIREMENTS_SHEET."""
    artifact_id = str(
        task.inputs.get("requirements_artifact_id") or "").strip()
    if not artifact_id:
        return []
    artifact = deps.store.get_artifact(artifact_id)
    if artifact is None or artifact.kind is not ArtifactKind.REQUIREMENTS_SHEET:
        return []
    qs = (artifact.content or {}).get("questions") or []
    if not isinstance(qs, list):
        return []
    return [q for q in qs if isinstance(q, dict)]


def _compose_goal(prior_goal: str,
                  questions: List[Dict[str, Any]],
                  answers: Dict[str, Any]) -> str:
    """Render a single goal string that bakes the clarify answers in."""
    parts: List[str] = []
    if prior_goal:
        parts.append(prior_goal)
    qa_lines: List[str] = []
    for q in questions:
        qid = str(q.get("id") or "").strip()
        prompt = str(q.get("prompt") or "").strip()
        answer = str(answers.get(qid) or "").strip()
        if not (qid and prompt and answer):
            continue
        qa_lines.append(f"- {prompt} -> {answer}")
    # Surface any free-form answers the user supplied for question ids
    # not seen in the requirements sheet (defensive: tests / older UIs).
    seen_ids = {str(q.get("id") or "") for q in questions}
    for qid, answer in answers.items():
        if str(qid) in seen_ids:
            continue
        ans = str(answer or "").strip()
        if ans:
            qa_lines.append(f"- {qid}: {ans}")
    if qa_lines:
        parts.append("User clarifications:\n" + "\n".join(qa_lines))
    return "\n\n".join(parts).strip()


def _coerce_layers(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for layer in raw:
        if not isinstance(layer, dict):
            continue
        name = str(layer.get("name") or "project").strip()
        files: List[Dict[str, Any]] = []
        for f in (layer.get("files") or []):
            if not isinstance(f, dict):
                continue
            path = str(f.get("path") or "").strip()
            desc = str(f.get("description") or path).strip()
            if not path:
                continue
            entry: Dict[str, Any] = {"path": path, "description": desc}
            deps = _coerce_depends_on(f.get("depends_on"))
            if deps:
                entry["depends_on"] = deps
            files.append(entry)
        out.append({"name": name, "files": files})
    return out


_CONTRACT_KEYS = ("endpoints", "schemas", "functions", "constants")


def _coerce_contracts(raw: Any) -> Dict[str, Any]:
    """Normalize the planner ``contracts`` block for storage on the WORK_PLAN.

    Mirrors :func:`cgx.answer.engine._normalize_contracts` defensively so a
    monkeypatched/legacy planner that returns a raw (or absent) contracts
    block still yields a clean, bounded dict: only the four recognised
    interface categories survive, each a list of small string-keyed dicts
    with empty/malformed entries dropped. Absent categories are omitted so
    an empty or missing block stores as ``{}``.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in _CONTRACT_KEYS:
        items = raw.get(key)
        if not isinstance(items, list):
            continue
        cleaned: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = {
                str(k): v for k, v in item.items()
                if isinstance(k, str) and str(k).strip()
                and isinstance(v, (str, int, float, bool, list, dict))
            }
            if entry:
                cleaned.append(entry)
        if cleaned:
            out[key] = cleaned
    return out


def _contract_entry_count(contracts: Dict[str, Any]) -> int:
    """Total number of declared contract entries across all categories."""
    return sum(len(v) for v in contracts.values() if isinstance(v, list))


def _coerce_depends_on(raw: Any) -> List[str]:
    """Normalize a per-file ``depends_on`` hint to a de-duplicated str list."""
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: set = set()
    for d in raw:
        s = str(d or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _layer_file_count(layers: List[Dict[str, Any]]) -> int:
    return sum(len(layer.get("files") or []) for layer in layers)


# File extensions that count as runnable/source (an "entry point" or a
# module a test can target). Docs, lockfiles, and pure config are absent
# on purpose so a manifest that is all README/config/tests fails early.
_SOURCE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue",
    ".go", ".rs", ".java", ".rb", ".php", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".cs", ".swift", ".kt", ".scala", ".sh",
    ".html", ".css", ".scss", ".sql",
}


def _file_ext(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    dot = base.rfind(".")
    return base[dot:].lower() if dot > 0 else ""


def _is_source_file(path: str) -> bool:
    return _file_ext(path) in _SOURCE_EXTS


def _is_test_file(path: str) -> bool:
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    if (low.startswith("tests/") or low.startswith("test/")
            or "/tests/" in low or "/test/" in low):
        return True
    if base.startswith("test_") or base.endswith("_test.py"):
        return True
    return ".test." in base or ".spec." in base


def _manifest_files(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [f for lay in layers if isinstance(lay, dict)
            for f in (lay.get("files") or [])
            if isinstance(f, dict) and f.get("path")]


def _find_dependency_cycle(
        files: List[Dict[str, Any]]) -> Optional[List[str]]:
    """Return a cyclic path (``a -> b -> a``) among intra-manifest deps."""
    path_set = {f["path"] for f in files}
    adj: Dict[str, List[str]] = {}
    for f in files:
        adj.setdefault(f["path"], [])
        for dep in f.get("depends_on") or []:
            if dep in path_set and dep != f["path"]:
                adj[f["path"]].append(dep)
    white, gray, black = 0, 1, 2
    color = {p: white for p in adj}
    stack: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        color[node] = gray
        stack.append(node)
        for nxt in adj.get(node, []):
            if color.get(nxt) == gray:
                return stack[stack.index(nxt):] + [nxt]
            if color.get(nxt) == white:
                found = dfs(nxt)
                if found:
                    return found
        stack.pop()
        color[node] = black
        return None

    for p in adj:
        if color[p] == white:
            found = dfs(p)
            if found:
                return found
    return None


def _validate_manifest_coherence(
        layers: List[Dict[str, Any]]) -> Optional[str]:
    """Deterministic manifest sanity check.

    Fails DECOMPOSE early (with an actionable message the router folds
    into a retry constraint) when the plan is logically broken:
    ``depends_on`` naming a file absent from the manifest, a dependency
    cycle, or a manifest carrying no runnable source file (only
    docs/config/tests -- nothing to build or to test against).
    """
    files = _manifest_files(layers)
    path_set = {f["path"] for f in files}

    dangling: List[str] = []
    for f in files:
        for dep in f.get("depends_on") or []:
            if dep not in path_set:
                dangling.append(f"{dep!r} (needed by {f['path']!r})")
    if dangling:
        return ("DECOMPOSE: manifest has dangling dependency reference(s): "
                + ", ".join(dangling[:6])
                + ". Every depends_on must name a file in the manifest.")

    cycle = _find_dependency_cycle(files)
    if cycle:
        return ("DECOMPOSE: manifest has a circular dependency: "
                + " -> ".join(cycle)
                + ". Break the cycle so files generate dependency-first.")

    non_test_source = [f["path"] for f in files
                       if _is_source_file(f["path"])
                       and not _is_test_file(f["path"])]
    if not non_test_source:
        return ("DECOMPOSE: manifest has no runnable source file (only "
                "docs/config/tests). Add at least one entry-point module.")
    return None


def _order_manifest_layers(
        layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Topologically sort files *within* each layer by intra-layer
    ``depends_on`` hints so a file is generated after the siblings it
    imports. Declared order is preserved for independent files (stable);
    layer order is left untouched -- cross-layer dependencies are assumed
    satisfied by the planner's layering.
    """
    out: List[Dict[str, Any]] = []
    for lay in layers:
        if not isinstance(lay, dict):
            out.append(lay)
            continue
        files = [f for f in (lay.get("files") or [])
                 if isinstance(f, dict) and f.get("path")]
        out.append({**lay, "files": _toposort_files(files)})
    return out


def _toposort_files(
        files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(files) < 2:
        return list(files)
    paths = [f["path"] for f in files]
    order_index = {p: i for i, p in enumerate(paths)}
    path_set = set(paths)
    by_path = {f["path"]: f for f in files}
    indeg = {p: 0 for p in paths}
    adj: Dict[str, List[str]] = {p: [] for p in paths}
    for f in files:
        for dep in f.get("depends_on") or []:
            if dep in path_set and dep != f["path"]:
                adj[dep].append(f["path"])
                indeg[f["path"]] += 1
    ready = sorted((p for p in paths if indeg[p] == 0),
                   key=order_index.get)
    ordered: List[str] = []
    while ready:
        p = ready.pop(0)
        ordered.append(p)
        newly: List[str] = []
        for nxt in adj[p]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                newly.append(nxt)
        if newly:
            ready.extend(newly)
            ready.sort(key=order_index.get)
    if len(ordered) != len(paths):
        # A cycle slipped past validation -- keep declared order.
        return list(files)
    return [by_path[p] for p in ordered]
