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


def _scaffolding_problems(plan: Dict[str, Any]) -> List[str]:
    """Missing project-scaffolding files a complete project must ship.

    Structural completeness is a plan-level invariant, not a per-run patch: a
    project the user can read, install, and test needs a ``README.md`` and a
    dependency manifest, and a ``src/`` layout additionally needs a root
    ``conftest.py`` so pytest inserts the project root onto ``sys.path`` and
    ``import src.pkg`` resolves. Each absence is returned as a concrete,
    re-askable problem rather than being silently accepted.
    """
    files = _flatten_files(plan)
    paths = [(f.get("path") or "").replace("\\", "/") for f in files]
    bases = {p.rsplit("/", 1)[-1].lower() for p in paths}
    problems: List[str] = []
    if "readme.md" not in bases:
        problems.append(
            "no README.md is planned; add a top-level README.md describing "
            "the project, its install steps, and its usage")
    if not ({"requirements.txt", "pyproject.toml"} & bases):
        problems.append(
            "no dependency manifest is planned; add a top-level "
            "requirements.txt (or pyproject.toml) listing the runtime and "
            "test dependencies")
    under_src = any(p.startswith("src/") for p in _source_paths(plan))
    if under_src and "conftest.py" not in paths:
        problems.append(
            "a 'src/' layout needs a root conftest.py so pytest can import "
            "the package; add a conftest.py at the project root")
    return problems


def ensure_scaffolding(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Inject any missing README / dependency manifest / conftest into the plan.

    Structural completeness is a deterministic guarantee, not a request the
    model may decline. The verifier requires a ``README.md``, a dependency
    manifest, and -- for a ``src/`` layout -- a root ``conftest.py``; rather
    than reject the plan and re-ask a weak model that routinely re-omits
    boilerplate, the missing entries are appended directly. The Developer
    already knows how to emit each one (source-derived templates for
    ``requirements.txt``/``conftest.py``, a grounded free-form call for
    ``README.md``), so the plan only has to declare them. ``README.md`` and
    ``requirements.txt`` depend on every planned ``.py`` file so they are
    generated last, after the sources they describe and scan exist on disk.
    """
    files = _flatten_files(plan)
    raw_paths = [f.get("path") or "" for f in files]
    bases = {p.replace("\\", "/").rsplit("/", 1)[-1].lower()
             for p in raw_paths}
    py_paths = [p for p in raw_paths if p.endswith(".py")]
    under_src = any(p.startswith("src/") for p in _source_paths(plan))

    injected: List[FileSpec] = []
    if not ({"requirements.txt", "pyproject.toml"} & bases):
        injected.append({
            "path": "requirements.txt",
            "description": ("Runtime and test dependencies for the project, "
                            "one pip requirement per line."),
            "depends_on": list(py_paths)})
    if under_src and "conftest.py" not in bases:
        injected.append({
            "path": "conftest.py",
            "description": ("Pytest bootstrap that puts the project root and "
                            "src/ on sys.path so tests import the package."),
            "depends_on": []})
    if "readme.md" not in bases:
        injected.append({
            "path": "README.md",
            "description": ("Top-level project README: summary, install steps "
                            "(pip install -r requirements.txt), usage, and how "
                            "to run the tests (pytest)."),
            "depends_on": list(py_paths)})

    if injected:
        layers = list(plan.get("layers") or [])
        layers = layers + [{"name": "scaffolding", "files": injected}]
        plan = {**plan, "layers": layers}
    return plan


def ensure_test_coverage(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Inject a pytest module for every source module no planned test covers.

    A plan that ships zero tests -- or tests only some of its modules --
    leaves the verifier with nothing to run for the uncovered code (the
    "no tests ran" failure). Coverage is therefore a deterministic plan
    invariant, not a request the weak Tech Lead may drop: every non-test
    ``.py`` source module (excluding package ``__init__.py`` markers) that no
    planned ``test_*`` / ``*_test.py`` file already ``depends_on`` gets a
    ``tests/test_<module>.py`` entry injected, depending on that module so it
    is generated after -- and grounded against -- the code it exercises. The
    description carries the test-authoring discipline (construct inputs
    in-test, assert invariants/round-trips) so the Developer writes a real
    suite rather than a placeholder.
    """
    src_py = [p for p in _source_paths(plan) if p.endswith(".py")]
    if not src_py:
        return plan
    files = _flatten_files(plan)
    covered: set = set()
    existing_paths: set = set()
    for f in files:
        p = (f.get("path") or "").replace("\\", "/")
        existing_paths.add(p)
        base = p.rsplit("/", 1)[-1].lower()
        if base.startswith("test_") or base.endswith("_test.py"):
            for d in (f.get("depends_on") or []):
                covered.add(d.replace("\\", "/"))

    injected: List[FileSpec] = []
    chosen: set = set()
    for src in src_py:
        if src in covered:
            continue
        base = src.rsplit("/", 1)[-1]
        if base.lower() == "__init__.py":
            continue
        mod = base[:-3]  # strip the '.py' suffix
        test_path = f"tests/test_{mod}.py"
        if test_path in existing_paths or test_path in chosen:
            parent = src.rsplit("/", 1)[0].rsplit("/", 1)[-1] if "/" in src \
                else ""
            test_path = f"tests/test_{parent}_{mod}.py"
        if test_path in existing_paths or test_path in chosen:
            continue
        chosen.add(test_path)
        injected.append({
            "path": test_path,
            "description": (
                f"Pytest tests for {src!r}. Import its public functions and "
                "classes and exercise them with inputs constructed inside the "
                "test (use the tmp_path fixture for any files); assert "
                "documented invariants and round-trips rather than fabricated "
                "literal values."),
            "depends_on": [src]})

    if injected:
        layers = list(plan.get("layers") or [])
        layers = layers + [{"name": "tests", "files": injected}]
        plan = {**plan, "layers": layers}
    return plan


def verify_plan(plan: Dict[str, Any]) -> List[str]:
    """Return concrete, actionable problems with a normalised plan.

    A plan that survives this gate is *coherent enough to build*: its paths
    are safe and relative, it commits to a single import rooting, its
    dependency graph is acyclic, every test has a module to exercise, and it
    ships the project scaffolding (README, dependency manifest, and -- for a
    ``src/`` layout -- a root conftest.py) that makes it a complete, runnable
    project. An empty list means the plan is fit for the Developer chain;
    otherwise the Tech Lead re-asks the model with the exact problems appended.
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
            base = p.rsplit("/", 1)[-1].lower()
            # Only pytest *test modules* need a target; conftest.py carries
            # fixtures/path setup and legitimately declares no depends_on even
            # though its name contains the substring "test".
            is_test_module = (base.startswith("test_")
                              or base.endswith("_test.py"))
            if (p.endswith(".py") and is_test_module
                    and not (f.get("depends_on") or [])):
                problems.append(
                    f"test file {p!r} declares no depends_on, so it has no "
                    "target module to import")
    problems.extend(_scaffolding_problems(plan))
    return list(dict.fromkeys(problems))
