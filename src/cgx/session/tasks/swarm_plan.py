"""Typed plan schema for the swarm agent (mirrors the greenfield WORK_PLAN).

The Tech Lead authors a draft plan; this module coerces that draft into a
stable, validated shape before the Developer executes it one file at a time.
Keeping the schema identical to the greenfield ``WORK_PLAN`` (``layers`` of
``{path, description, depends_on}`` plus ``contracts``) lets the swarm reuse
the DECOMPOSE coherence + ordering machinery instead of a parallel copy.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - py<3.8 fallback, unused in practice
    from typing_extensions import TypedDict  # type: ignore[assignment]


class FileSpec(TypedDict, total=False):
    """One planned file: a path, a purpose, and intra-plan dependencies."""

    path: str
    description: str
    depends_on: List[str]


class LayerSpec(TypedDict, total=False):
    """A named group of files (models / core / api / tests, etc.)."""

    name: str
    files: List[FileSpec]


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

# File extensions the AST/verification ladder treats as runnable source.
_SOURCE_EXT = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java")


def parse_plan_reply(reply: str) -> Optional[Dict[str, Any]]:
    """Extract a plan JSON object from an LLM reply (fenced or bare braces)."""
    raw = reply or ""
    m = _JSON_FENCE.search(raw)
    candidate = m.group(1) if m else None
    if candidate is None:
        start, end = raw.find("{"), raw.rfind("}")
        candidate = raw[start:end + 1] if start != -1 and end > start else None
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _coerce_file(raw: Any) -> Optional[FileSpec]:
    """Coerce one raw file entry into a :class:`FileSpec`, or ``None``."""
    if not isinstance(raw, dict):
        return None
    path = str(raw.get("path") or "").strip()
    if not path:
        return None
    deps_raw = raw.get("depends_on") or []
    deps = ([str(d).strip() for d in deps_raw if str(d).strip()]
            if isinstance(deps_raw, list) else [])
    return {"path": path,
            "description": str(raw.get("description") or "").strip(),
            "depends_on": deps}


def normalize_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a draft plan into the validated schema.

    Drops path-less entries, dedupes by path (first wins), and prunes
    ``depends_on`` edges naming a path absent from the manifest (a dangling
    hint would only mislead ordering and grounding). Dependency ordering
    itself is applied later by the shared toposort (Phase R).
    """
    layers_in = raw.get("layers")
    if not isinstance(layers_in, list):
        layers_in = []
    seen: set = set()
    layers: List[LayerSpec] = []
    for lay in layers_in:
        if not isinstance(lay, dict):
            continue
        files: List[FileSpec] = []
        for f in (lay.get("files") or []):
            spec = _coerce_file(f)
            if spec and spec["path"] not in seen:
                seen.add(spec["path"])
                files.append(spec)
        if files:
            layers.append({"name": str(lay.get("name") or "layer"),
                           "files": files})
    for lay in layers:
        for f in lay["files"]:
            f["depends_on"] = [d for d in f["depends_on"] if d in seen]
    contracts = raw.get("contracts")
    return {
        "goal": str(raw.get("goal") or "").strip(),
        "layers": layers,
        "contracts": contracts if isinstance(contracts, dict) else {},
    }


def _flatten_files(plan: Dict[str, Any]) -> List[FileSpec]:
    """Flatten plan files in declared layer order (no reordering)."""
    out: List[FileSpec] = []
    for lay in plan.get("layers") or []:
        for f in (lay.get("files") or []):
            out.append(f)
    return out


def iter_plan_files(plan: Dict[str, Any]) -> List[FileSpec]:
    """Flatten plan files in global dependency order for the Developer.

    Reuses the shared manifest toposort so a file is only handed to the
    Developer after every file it ``depends_on`` -- the whole point of the
    one-file-at-a-time discipline is that dependencies already exist on disk
    (and can be grounded) before their consumers are written.
    """
    from cgx.session.tasks.decompose import toposort_manifest_files
    return toposort_manifest_files(_flatten_files(plan))


def ordered_paths(plan: Dict[str, Any]) -> List[str]:
    """The plan's file paths in dependency-first execution order."""
    return [f["path"] for f in iter_plan_files(plan)]


def plan_specs(plan: Dict[str, Any]) -> Dict[str, FileSpec]:
    """A ``{path: FileSpec}`` map for O(1) per-file lookup by the Developer."""
    return {f["path"]: f for f in _flatten_files(plan)}


def plan_is_buildable(plan: Dict[str, Any]) -> bool:
    """True when at least one non-test runnable source file is planned."""
    for f in _flatten_files(plan):
        p = f.get("path", "")
        if p.endswith(_SOURCE_EXT) and "test" not in p.lower():
            return True
    return False


def _source_paths(plan: Dict[str, Any]) -> List[str]:
    """Runnable, non-test source paths in the plan (normalised slashes)."""
    out: List[str] = []
    for f in _flatten_files(plan):
        p = (f.get("path") or "").replace("\\", "/")
        if p.endswith(_SOURCE_EXT) and "test" not in p.lower():
            out.append(p)
    return out


def _has_dependency_cycle(files: List[FileSpec]) -> bool:
    """True when the ``depends_on`` graph over planned paths has a cycle."""
    path_set = {f["path"] for f in files}
    adj = {f["path"]: [d for d in (f.get("depends_on") or [])
                       if d in path_set and d != f["path"]]
           for f in files}
    color: Dict[str, int] = {p: 0 for p in path_set}  # 0=white 1=grey 2=black

    def visit(node: str) -> bool:
        color[node] = 1
        for nxt in adj.get(node, []):
            if color[nxt] == 1 or (color[nxt] == 0 and visit(nxt)):
                return True
        color[node] = 2
        return False

    return any(color[p] == 0 and visit(p) for p in path_set)


def verify_plan(plan: Dict[str, Any]) -> List[str]:
    """Return concrete, actionable problems with a normalised plan.

    A plan that survives this gate is *coherent enough to build*: its paths
    are safe and relative, it commits to a single import rooting, its
    dependency graph is acyclic, and every test has a module to exercise. An
    empty list means the plan is fit for the Developer chain; otherwise the
    Tech Lead re-asks the model with the exact problems appended.
    """
    files = _flatten_files(plan)
    problems: List[str] = []
    if not files:
        return ["the plan lists no files"]
    for f in files:
        p = (f.get("path") or "").replace("\\", "/")
        if p.startswith("/") or p.startswith("~"):
            problems.append(f"path {p!r} is absolute; use a relative path")
        if ".." in [seg for seg in p.split("/") if seg]:
            problems.append(f"path {p!r} escapes the project root")
    if not plan_is_buildable(plan):
        problems.append("no runnable non-test source file is planned")
    if _has_dependency_cycle(files):
        problems.append("the depends_on graph has a cycle")
    src = _source_paths(plan)
    under_src = [p for p in src if p.startswith("src/")]
    top_level = [p for p in src if "/" not in p]
    if under_src and top_level:
        problems.append(
            "inconsistent layout: modules are split between 'src/' "
            f"({', '.join(under_src)}) and the top level "
            f"({', '.join(top_level)}); commit to one rooting so the "
            "Developer emits consistent imports")
    if src:
        for f in files:
            p = (f.get("path") or "").replace("\\", "/")
            if (p.endswith(".py") and "test" in p.lower()
                    and not (f.get("depends_on") or [])):
                problems.append(
                    f"test file {p!r} declares no depends_on, so it has no "
                    "target module to import")
    return list(dict.fromkeys(problems))
