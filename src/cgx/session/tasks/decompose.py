

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
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cgx.session.budget import LoopBudget
from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.scaffold_validate import (missing_stack_entry_files,
                                           stack_entry_description)
from cgx.session.scope import estimate_scope
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
    session_skills,
)
from cgx.trace import emit_trace

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

    # Deterministic scope calibration (P1.1): derive a complexity tier and
    # a "minimal viable stack" ceiling from the composed goal so the planner
    # cannot over-scope a simple objective (a calculator dragging in a DB,
    # migrations, auth, a frontend framework, and Selenium E2E tests).
    scope = estimate_scope(composed_goal)
    emit_trace("scope_estimate", stage="decompose", **scope.as_dict())

    skills = session_skills(task, deps)
    try:
        result = plan_scaffold_manifest(
            composed_goal, deps.provider, goal=composed_goal, skills=skills,
            scope_constraint=scope.constraint)
    except Exception as exc:
        logger.exception("DECOMPOSE: plan_scaffold_manifest crashed")
        return ExecutorResult(
            failure=f"decompose failed: {type(exc).__name__}: {exc}")

    plan_md = str((result or {}).get("plan_md") or "")
    layers = _coerce_layers((result or {}).get("layers"))
    contracts = _coerce_contracts((result or {}).get("contracts"))
    if not _layer_file_count(layers):
        return ExecutorResult(
            failure="DECOMPOSE: planner returned an empty manifest",
            retryable=True)

    # Plan self-critique (P1.2): one bounded LLM pass reviews the manifest
    # against the scope ceiling and drops speculative files (a DB layer,
    # auth module, or E2E suite the goal never asked for). Advisory and
    # deterministic-safe -- a provider miss leaves the manifest untouched
    # (today's behaviour) -- and the removals are gated by guardrails that
    # refuse to drop an entry point, the last source/test file, or a
    # depends_on target.
    layers, critique_removed = _apply_plan_critique(
        layers, composed_goal, scope, deps)

    # Deterministic coherence gate: repair what can be repaired in place
    # (missing stack entry points, dangling depends_on, dependency cycles
    # -- the latter two only ordering hints), fail early only when the
    # manifest is logically unbuildable, then topologically order files by
    # dependency hints so SCAFFOLD generates dependencies before their
    # consumers.
    report = CoherenceReport()
    _inject_stack_entry_files(layers, report)
    coherence_error = _validate_manifest_coherence(layers, report)
    if coherence_error:
        return ExecutorResult(failure=coherence_error, retryable=True)

    # Coherence surgery as a plan-quality signal (P1.3). The gate above
    # repairs an incoherent manifest in place, but a plan that needed
    # *heavy* structural surgery (a misfiled entry point, cross-language
    # depends_on no import can express, a dependency cycle) is one the
    # planner got wrong at the source -- scaffolding it drags that churn
    # downstream. When the surgery score clears the threshold and a
    # DECOMPOSE_RETRY_BUDGET re-ask is still available, fold the surgery
    # summary into the goal and re-plan once. On the retry (budget spent)
    # we proceed with the repaired manifest, so this is never worse than
    # today's behaviour.
    emit_trace("plan_coherence", stage="decompose", **report.as_dict())
    budget = LoopBudget.from_inputs(task.inputs)
    if (report.surgery_score >= COHERENCE_MUTATION_THRESHOLD
            and not budget.decompose_retry_exhausted):
        logger.warning(
            "DECOMPOSE: coherence gate performed heavy surgery "
            "(score=%d, %s); re-asking the planner for a coherent manifest",
            report.surgery_score, report.as_dict())
        return ExecutorResult(
            failure=_coherence_reask_constraint(report),
            retryable=True)

    layers = _order_manifest_layers(layers)

    # Mandatory endpoint contracts for a cross-language client/server manifest
    # (P0a). A JSX/TSX/Vue client beside a Python backend route talks over
    # HTTP, so a request-key rename (client sends ``operator`` while the
    # handler reads ``operation``) is invisible to every Python-only gate --
    # exactly how ses_4cbf963cdc67435a shipped green with a broken seam. When
    # the planner omitted the ``endpoints`` contract for such a manifest, run
    # one bounded extract pass; fail-closed if it still cannot be produced so
    # the seam is never scaffolded contract-free.
    contract_error = _ensure_cross_seam_endpoints(
        contracts, layers, composed_goal, skills, deps)
    if contract_error:
        return ExecutorResult(failure=contract_error)

    # Mandatory skeleton pass (P0b): generate the full project skeleton
    # before SCAFFOLD starts.
    from cgx.answer.engine import generate_project_skeleton
    manifest_paths = []
    for lay in layers:
        if isinstance(lay, dict):
            for f in (lay.get("files") or []):
                if isinstance(f, dict) and f.get("path"):
                    manifest_paths.append(str(f["path"]))
                    
    skeleton = generate_project_skeleton(manifest_paths, deps.provider, composed_goal)
    if not isinstance(contracts, dict):
        contracts = {}
    if skeleton:
        contracts["project_skeleton"] = skeleton

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
            "project_complexity": scope.complexity,
            "scope": scope.as_dict(),
        },
    )
    return ExecutorResult(
        outputs={
            "work_plan_artifact_id": artifact.artifact_id,
            "file_count": _layer_file_count(layers),
            "layer_count": len(layers),
            "contract_count": _contract_entry_count(contracts),
            "project_complexity": scope.complexity,
            "scope_max_files": scope.max_files,
            "critique_removed": len(critique_removed),
            "coherence_surgery": report.surgery_score,
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _apply_plan_critique(layers: List[Dict[str, Any]], goal: str, scope: Any,
                         deps: ExecutorDeps) -> tuple:
    """Run the bounded plan self-critique (P1.2) and apply safe removals.

    Returns ``(layers, removed_paths)``. The critique is advisory: the model
    proposes speculative files, but the deterministic :func:`_safe_removals`
    guardrails decide what is actually dropped. Any provider/parse failure
    yields no removals so the manifest is unchanged (today's behaviour). A
    ``plan_critique`` trace record is emitted when tracing is on.
    """
    from cgx.answer.engine import critique_scaffold_manifest
    try:
        flagged = critique_scaffold_manifest(
            goal, layers, deps.provider,
            scope_constraint=getattr(scope, "constraint", None))
    except Exception:  # pragma: no cover - defensive: critique is best-effort
        logger.exception("DECOMPOSE: plan self-critique crashed")
        flagged = []
    removed = _safe_removals(layers, flagged)
    if removed:
        layers = _drop_files(layers, removed)
    emit_trace("plan_critique", stage="decompose",
               flagged=list(flagged), removed=removed,
               removed_count=len(removed))
    return layers, removed


def _safe_removals(layers: List[Dict[str, Any]],
                   flagged: List[str]) -> List[str]:
    """Filter the critique's flagged paths down to what is safe to drop.

    Refuses to drop a file another kept file ``depends_on`` (an import a
    sibling needs) and preserves at least one source file and one test file:
    a critique that would gut every source file is rejected wholesale, and
    test files are never dropped when they are the only tests. Order is
    preserved, duplicates removed.
    """
    if not flagged:
        return []
    files = _manifest_files(layers)
    all_paths = [str(f.get("path") or "") for f in files]
    depended: set = set()
    for f in files:
        for d in (f.get("depends_on") or []):
            depended.add(str(d))
    candidates = [p for p in flagged if p and p not in depended]
    if not candidates:
        return []
    had_src = any(_is_source_file(p) and not _is_test_file(p)
                  for p in all_paths)
    had_test = any(_is_test_file(p) for p in all_paths)
    remove_set = set(candidates)
    src_left = any(_is_source_file(p) and not _is_test_file(p)
                   for p in all_paths if p not in remove_set)
    if had_src and not src_left:
        return []  # would gut every source file -- refuse the critique
    test_left = any(_is_test_file(p) for p in all_paths
                    if p not in remove_set)
    if had_test and not test_left:
        candidates = [p for p in candidates if not _is_test_file(p)]
    return candidates


def _drop_files(layers: List[Dict[str, Any]],
                removed: List[str]) -> List[Dict[str, Any]]:
    """Return ``layers`` with ``removed`` paths (and any dangling
    ``depends_on`` references to them) scrubbed out."""
    remove_set = set(removed)
    out: List[Dict[str, Any]] = []
    for layer in layers:
        kept: List[Dict[str, Any]] = []
        for f in (layer.get("files") or []):
            if str(f.get("path") or "").strip() in remove_set:
                continue
            deps_list = f.get("depends_on")
            if deps_list:
                pruned = [d for d in deps_list if str(d) not in remove_set]
                if pruned:
                    f["depends_on"] = pruned
                else:
                    f.pop("depends_on", None)
            kept.append(f)
        out.append({"name": layer.get("name", "project"), "files": kept})
    return out


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


# Cross-seam detection (P0a): a client/server manifest is a JSX/TSX/Vue
# frontend beside a Python backend route -- the two halves talk over HTTP,
# so their shared request/response shape MUST be pinned by an endpoints
# contract or a key rename slips through every Python-only gate.
_FRONTEND_EXTS = (".jsx", ".tsx", ".vue")
_SERVER_BASENAMES = frozenset(
    {"app.py", "main.py", "server.py", "wsgi.py", "asgi.py", "api.py"})
_BACKEND_FRAMEWORKS = ("flask", "fastapi", "django", "express")
_SERVER_SIGNAL_RE = re.compile(
    r"flask|fastapi|django|express|@app\.route|@router|endpoint|"
    r"\bapi\b|\broute", re.IGNORECASE)


def _has_frontend_caller(layers: List[Dict[str, Any]]) -> bool:
    """True when the manifest declares a JS/TS/Vue frontend source file."""
    for layer in (layers or []):
        for f in (layer.get("files") or []):
            path = str(f.get("path") or "").strip().lower()
            if path.endswith(_FRONTEND_EXTS):
                return True
    return False


def _has_backend_route(layers: List[Dict[str, Any]],
                       skills: Optional[List[str]], goal: str) -> bool:
    """True when a Python file looks like a web-framework route handler.

    A canonical entry basename (``app.py`` / ``main.py`` / ...), a
    server-framework signal in the file's description, or a backend skill /
    goal keyword paired with any generated ``.py`` file each qualify.
    """
    text = (" ".join(str(s) for s in (skills or []))
            + " " + (goal or "")).lower()
    skill_fw = any(fw in text for fw in _BACKEND_FRAMEWORKS)
    for layer in (layers or []):
        for f in (layer.get("files") or []):
            path = str(f.get("path") or "").strip()
            if not path.endswith(".py"):
                continue
            base = path.rsplit("/", 1)[-1].lower()
            if base in _SERVER_BASENAMES or skill_fw:
                return True
            if _SERVER_SIGNAL_RE.search(str(f.get("description") or "")):
                return True
    return False


def _is_client_server_manifest(layers: List[Dict[str, Any]],
                               skills: Optional[List[str]],
                               goal: str) -> bool:
    """True for a cross-language frontend<->backend manifest (P0a)."""
    return (_has_frontend_caller(layers)
            and _has_backend_route(layers, skills, goal))


def _ensure_cross_seam_endpoints(
        contracts: Dict[str, Any],
        layers: List[Dict[str, Any]],
        goal: str,
        skills: Optional[List[str]],
        deps: ExecutorDeps) -> Optional[str]:
    """Guarantee a cross-seam manifest carries an ``endpoints`` contract.

    Mutates ``contracts`` in place, adding an ``endpoints`` list recovered by
    a bounded extract pass when the planner omitted it for a client/server
    manifest. Returns ``None`` when the contract is present (or the manifest
    is not cross-seam), or a fail-closed error string when the seam exists
    but no endpoints could be produced -- the caller turns that into a
    terminal DECOMPOSE failure so a contract-free seam is never scaffolded.
    """
    if contracts.get("endpoints"):
        return None
    if not _is_client_server_manifest(layers, skills, goal):
        return None
    try:
        from cgx.answer.engine import extract_endpoint_contracts
        endpoints = extract_endpoint_contracts(goal, layers, deps.provider)
    except Exception:  # pragma: no cover - defensive: extractor is best-effort
        logger.exception("DECOMPOSE: endpoint extraction pass crashed")
        endpoints = []
    if endpoints:
        contracts["endpoints"] = endpoints
        logger.info("DECOMPOSE: recovered %d endpoint contract(s) via the "
                    "extract pass for a cross-seam manifest", len(endpoints))
        return None
    return ("DECOMPOSE: cross-language client/server project requires an "
            "endpoints contract (frontend fetch <-> backend route), but the "
            "planner omitted it and the bounded extraction pass produced "
            "none -- refusing to scaffold the seam contract-free")


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


# Number of *heavyweight* structural repairs (relocated entry points,
# cut cross-language edges, broken cycles) above which DECOMPOSE stops
# trusting its own plan and re-asks the planner once (P1.3). Three was
# chosen from the calculator run that motivated this work: that manifest
# tripped exactly one of each defect, so the third repair is the point
# where "a slip" becomes "an incoherent plan". Kept deliberately high so
# a single routine repair never burns the DECOMPOSE_RETRY_BUDGET.
COHERENCE_MUTATION_THRESHOLD = 3


@dataclass
class CoherenceReport:
    """Tally of the in-place repairs the coherence gate performed.

    Populated by :func:`_inject_stack_entry_files` and
    :func:`_validate_manifest_coherence` when a report is threaded in, and
    read back in :func:`run_decompose` as a plan-quality signal. Only the
    three heavyweight structural defects feed :attr:`surgery_score` (the
    re-ask trigger); ``injected`` and ``dangling_pruned`` are recorded for
    observability but are routine and cheap, so they never inflate it.
    """
    relocated: List[str] = field(default_factory=list)
    injected: List[str] = field(default_factory=list)
    dangling_pruned: List[str] = field(default_factory=list)
    cross_language_cut: List[str] = field(default_factory=list)
    broken_cycles: List[str] = field(default_factory=list)

    @property
    def surgery_score(self) -> int:
        return (len(self.relocated)
                + len(self.cross_language_cut)
                + len(self.broken_cycles))

    def as_dict(self) -> Dict[str, int]:
        return {
            "relocated": len(self.relocated),
            "injected": len(self.injected),
            "dangling_pruned": len(self.dangling_pruned),
            "cross_language_cut": len(self.cross_language_cut),
            "broken_cycles": len(self.broken_cycles),
            "surgery_score": self.surgery_score,
        }


def _coherence_reask_constraint(report: "CoherenceReport") -> str:
    """Explain, for the DECOMPOSE re-ask, why the prior plan was rejected.

    Folded into ``prior_goal`` by :func:`_fold_failure_into_goal` so the
    planner LLM sees the concrete defects (not a bare "try again") and can
    fix them at the source.
    """
    parts: List[str] = []
    if report.relocated:
        parts.append("misfiled required entry file(s): "
                     + ", ".join(report.relocated))
    if report.cross_language_cut:
        parts.append("declared cross-language depends_on no import can "
                     "express on: " + ", ".join(report.cross_language_cut))
    if report.broken_cycles:
        parts.append("introduced dependency cycle(s): "
                     + ", ".join(report.broken_cycles))
    detail = "; ".join(parts) or "required heavy structural repair"
    return ("DECOMPOSE: the prior manifest needed heavy structural repair "
            f"({detail}). Re-plan a coherent manifest -- place entry files "
            "where the toolchain resolves them, only declare depends_on "
            "between files of the same language, and avoid dependency "
            "cycles.")


# Source extensions grouped by the runtime that imports them. Only
# families whose members can import each other are listed: a file's
# ``depends_on`` is an import hint, so an edge between two families is
# not a build-order slip but a statement no language can express.
# Markup, styling, config and data files are deliberately absent -- they
# are runtime-agnostic (index.html legitimately references src/main.jsx,
# requirements.txt legitimately follows backend/app.py) and must never
# be pruned.
_LANG_FAMILIES = {
    "py": {".py"},
    "js": {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue"},
    "go": {".go"},
    "rs": {".rs"},
    "rb": {".rb"},
    "php": {".php"},
    "java": {".java"},
}
_EXT_LANG = {ext: fam for fam, exts in _LANG_FAMILIES.items() for ext in exts}


def _lang_family(path: str) -> Optional[str]:
    """Return the importing runtime for ``path``, or None if agnostic."""
    return _EXT_LANG.get(_file_ext(path))


def _repair_cross_language_deps(
        files: List[Dict[str, Any]],
        layers: List[Dict[str, Any]]) -> List[str]:
    """Drop ``depends_on`` edges no import statement could express.

    A planner that lays out a React frontend beside a Python backend
    routinely writes a pytest file covering the JSX components --
    ``tests/test_main.py depends_on ['src/main.jsx', 'src/App.jsx']``,
    described as "unit tests for main React components". Python cannot
    import JSX, so the node is unsatisfiable: SCAFFOLD invents a module
    name to import, the phantom-import gate rejects it, and every
    regenerate invents a different name for the same missing module.
    The manifest is the only place this is fixable.

    Edges are only cut between two *known, differing* runtimes; anything
    agnostic (HTML, CSS, config, data) keeps its edges, so the injected
    ``index.html -> src/main.jsx`` and ``requirements.txt ->
    backend/app.py`` orderings survive untouched.

    A test file whose dependencies were *entirely* cross-language is
    dropped rather than kept with an empty hint list: the planner's own
    description ties it to sources it cannot reach, and generating it
    anyway just relocates the hallucination from the import list to the
    test bodies. Non-test files are never dropped -- a mislinked module
    is still buildable, and losing one silently could gut the project.

    Returns the paths whose cross-language edges were cut, so the caller
    can tally the repair as a plan-quality signal (P1.3).
    """
    touched: List[str] = []
    doomed: set = set()
    for f in files:
        deps = [str(d) for d in (f.get("depends_on") or [])]
        if not deps:
            continue
        own = _lang_family(f["path"])
        if own is None:
            continue
        alien = [d for d in deps
                 if (_lang_family(d) or own) != own]
        if not alien:
            continue
        kept = [d for d in deps if d not in alien]
        f["depends_on"] = kept
        touched.append(f["path"])
        logger.warning(
            "DECOMPOSE: dropping cross-language depends_on %s from %r "
            "(%s cannot import them)", alien, f["path"], own)
        if not kept and _is_test_file(f["path"]):
            doomed.add(f["path"])
    if not doomed:
        return touched
    logger.warning(
        "DECOMPOSE: dropping unsatisfiable test file(s) %s -- every file "
        "they were planned to cover is in another language",
        sorted(doomed))
    for lay in layers:
        if not isinstance(lay, dict):
            continue
        lay["files"] = [f for f in (lay.get("files") or [])
                        if not (isinstance(f, dict)
                                and f.get("path") in doomed)]
    for f in files:
        surviving = f.get("depends_on")
        if isinstance(surviving, list) and surviving:
            f["depends_on"] = [d for d in surviving if d not in doomed]
    return touched


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


def _relocate_misplaced_stack_entries(
        files: List[Dict[str, Any]],
        missing: List[Dict[str, str]]) -> List[str]:
    """Move an entry file the planner put in the wrong directory.

    A planner that declares ``public/index.html`` has not forgotten the
    Vite entry, it has misfiled it: ``public/`` is copied verbatim as a
    static asset, so the bundler still cannot resolve an entry. Injecting
    a second node makes it worse -- both paths get the same boilerplate
    and SCAFFOLD's duplicate-content gate then drops one of them, which
    is exactly the root entry that has to exist. Rewriting the existing
    node's path keeps one file, in the only place the toolchain looks.

    Mutates the manifest nodes in place, repoints any ``depends_on``
    naming the old path, and returns the paths that were moved.
    """
    moved: List[str] = []
    for entry in list(missing):
        want = entry["path"]
        base = want.rsplit("/", 1)[-1]
        candidates = sorted(
            (f for f in files
             if f["path"] != want and f["path"].rsplit("/", 1)[-1] == base),
            key=lambda f: (f["path"].count("/"), f["path"]))
        if not candidates:
            continue
        node = candidates[0]
        old = node["path"]
        node["path"] = want
        node["description"] = stack_entry_description(want)
        for other in files:
            deps = other.get("depends_on")
            if not isinstance(deps, list) or old not in deps:
                continue
            other["depends_on"] = [want if d == old else d for d in deps]
        missing.remove(entry)
        moved.append(f"{old} -> {want}")
    return moved


def _inject_stack_entry_files(
        layers: List[Dict[str, Any]],
        report: Optional["CoherenceReport"] = None) -> List[str]:
    """Add toolchain-mandated entry files the planner left out.

    A manifest that declares ``vite.config.js`` but no root
    ``index.html`` is unbuildable, and the regenerate loop cannot recover
    from it: SCAFFOLD only ever generates the paths the manifest names,
    so the missing entry point stays missing however many times the tree
    is re-authored. Appending the entry to the layer that carries its
    trigger turns a guaranteed dead end into a normally-generated file.
    Mutates ``layers`` in place and returns the paths that were added.

    When ``report`` is supplied, relocated and injected entry paths are
    tallied on it as a plan-quality signal (P1.3).
    """
    files = _manifest_files(layers)
    paths = [f["path"] for f in files]
    missing = missing_stack_entry_files(paths)
    if not missing:
        return []
    moved = _relocate_misplaced_stack_entries(files, missing)
    if moved:
        if report is not None:
            report.relocated.extend(moved)
        logger.warning(
            "DECOMPOSE: manifest misfiled required entry file(s) %s; "
            "moving them to where the toolchain resolves them", moved)
    if not missing:
        return []
    # Append to the last layer that has files so the generator sees the
    # whole tree (notably the script entry point the HTML must reference)
    # as context, and declare the dependency so the toposort keeps that
    # ordering within the layer.
    target = next((lay for lay in reversed(layers)
                   if isinstance(lay, dict) and lay.get("files")), None)
    if target is None:
        return []
    script = next((p for p in (f["path"] for f in files)
                   if p.rsplit("/", 1)[-1].split(".")[0] in ("main", "index")
                   and p.rsplit(".", 1)[-1] in ("jsx", "tsx", "js", "ts")),
                  None)
    added: List[str] = []
    for entry in missing:
        node: Dict[str, Any] = {"path": entry["path"],
                                "description": entry["description"]}
        if script:
            node["depends_on"] = [script]
        target["files"].append(node)
        added.append(entry["path"])
    if added:
        if report is not None:
            report.injected.extend(added)
        logger.warning(
            "DECOMPOSE: manifest omitted required entry file(s) %s; "
            "injecting them so the project can build", added)
    return added


def _validate_manifest_coherence(
        layers: List[Dict[str, Any]],
        report: Optional["CoherenceReport"] = None) -> Optional[str]:
    """Deterministic manifest sanity check.

    Fails DECOMPOSE early only when the plan is logically unbuildable:
    a manifest carrying no runnable source file (only docs/config/tests
    -- nothing to build or to test against).

    ``depends_on`` problems are *not* fatal: it is only a topological /
    context-scoping hint, so a common planner slip -- a dangling entry
    (a phantom path or a glob like ``src/components/*.jsx``), a
    cross-language edge (a pytest file covering JSX components), or a
    dependency cycle (``a -> b -> a``, routine for small local models)
    -- is repaired in place with a warning rather than sinking an
    otherwise-buildable manifest and terminally failing the session.
    A cycle is broken by dropping the back-edge that closes it; the
    remaining edges still give the toposort a usable generation order.
    """
    files = _manifest_files(layers)
    path_set = {f["path"] for f in files}

    for f in files:
        deps = f.get("depends_on") or []
        kept = [d for d in deps if d in path_set]
        if len(kept) != len(deps):
            dropped = [d for d in deps if d not in path_set]
            if report is not None:
                report.dangling_pruned.append(f["path"])
            logger.warning(
                "DECOMPOSE: pruning dangling depends_on %s from %r",
                dropped, f["path"])
            f["depends_on"] = kept

    # Runs after the dangling prune so a phantom path is reported as
    # phantom rather than as a foreign language, and before the cycle
    # search so a dropped node cannot leave a half-lit cycle behind.
    cut = _repair_cross_language_deps(files, layers)
    if report is not None:
        report.cross_language_cut.extend(cut)
    files = _manifest_files(layers)
    by_path = {f["path"]: f for f in files}

    # Each pass removes exactly one edge, so this terminates.
    cycle = _find_dependency_cycle(files)
    while cycle:
        src, dst = cycle[-2], cycle[-1]
        entry = by_path[src]
        entry["depends_on"] = [d for d in (entry.get("depends_on") or [])
                               if d != dst]
        if report is not None:
            report.broken_cycles.append(f"{src} -> {dst}")
        logger.warning(
            "DECOMPOSE: breaking dependency cycle %s by dropping edge "
            "%r -> %r", " -> ".join(cycle), src, dst)
        cycle = _find_dependency_cycle(files)

    non_test_source = [f["path"] for f in files
                       if _is_source_file(f["path"])
                       and not _is_test_file(f["path"])]
    if not non_test_source:
        return ("DECOMPOSE: manifest has no runnable source file (only "
                "docs/config/tests). Add at least one entry-point module.")
    return None


def _order_manifest_layers(
        layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Topologically sort files by hard-coding the pipeline into 4 strict layers
    (Models -> Core -> API -> Tests) and sorting within each layer.
    """
    out: List[Dict[str, Any]] = []
    all_files: List[Dict[str, Any]] = []
    
    for lay in layers:
        if isinstance(lay, dict):
            for f in (lay.get("files") or []):
                if isinstance(f, dict) and f.get("path"):
                    all_files.append(f)
                    
    layer1, layer2, layer3, layer4 = [], [], [], []
    for f in all_files:
        path = str(f.get("path", "")).lower()
        if "test" in path:
            layer4.append(f)
        elif any(kw in path for kw in ["model", "config", "util", "schema"]):
            layer1.append(f)
        elif any(kw in path for kw in ["main", "app", "route", "api", "server"]):
            layer3.append(f)
        else:
            layer2.append(f)
            
    buckets = [
        {"name": "models_configs_utils", "files": layer1},
        {"name": "core_logic_auth", "files": layer2},
        {"name": "api_routes_main", "files": layer3},
        {"name": "tests", "files": layer4},
    ]
    
    for lay in buckets:
        if lay["files"]:
            out.append({**lay, "files": _toposort_files(lay["files"])})
            
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
                   key=lambda p: order_index[p])
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
            ready.sort(key=lambda p: order_index[p])
    if len(ordered) != len(paths):
        # A cycle slipped past validation -- keep declared order.
        return list(files)
    return [by_path[p] for p in ordered]
