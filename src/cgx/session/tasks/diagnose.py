"""DIAGNOSE -- the reasoning rung between mechanical patch and regenerate.

Spawned only for the ambiguous ``_DIAGNOSE_CLASSES`` failures (the ones
where today's ladder jumps straight to a whole-tree regenerate), this
executor is **deterministic-first**: it reuses the existing
:mod:`cgx.session.repair.classify` verdict + traceback localization and,
when that already names the files a fix must touch, emits a
``minimal_action`` with *no model call at all*. Only a genuinely
ambiguous failure (``unknown`` / no localization) falls back to a
bounded, read-only ReAct loop over the failure + repo, capped at
:data:`DIAGNOSE_STEPS` tool calls, that emits exactly one typed
``DIAGNOSIS``. The loop is provider-agnostic: a small local model that
returns terse/garbled output degrades cleanly to ``escalate`` -- i.e.
today's regenerate path -- so behavior is never *worse* than the ladder.

The router (pure) later reads only ``DIAGNOSIS.minimal_action`` and maps
it to an existing successor; this executor proposes, it never mutates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from cgx.session.budget import LoopBudget
from cgx.session.models import (
    Artifact,
    ArtifactKind,
    Fact,
    FactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.repair.context import (
    GATE_API_CHECK,
    GATE_RUNTIME,
    GATE_SMOKE,
    GATE_VERIFY,
    FailureContext,
)
from cgx.session.repair.ledger import RepairLedger
from cgx.session.tasks.base import ExecutorDeps, ExecutorResult, register_executor
from cgx.trace import emit_trace

logger = logging.getLogger(__name__)

# Default bound on read-only ReAct tool calls per DIAGNOSE round (design
# §12.3). Kept small so a single round stays cheap; the outer REPAIR_BUDGET
# still bounds how many rounds run.
DIAGNOSE_STEPS = 3

# The closed verdict enum the pure router dispatches (design §5).
MINIMAL_ACTIONS = frozenset({
    "patch_files", "add_dependency", "remove_dependency",
    "adjust_manifest", "regenerate_files", "escalate",
})

# The "needs reasoning" gate (design §12.4): a subset of REPAIR's
# regenerate classes. Mechanical tokens keep their fast path to REPAIR;
# only these ambiguous ones route to DIAGNOSE.
_DIAGNOSE_CLASSES = frozenset(
    {"assertion_drift", "collection_error", "unknown", "runtime_failure"})

# Deterministic-first (design §12.1): an import/boot failure whose
# traceback already localizes to first-party files needs no model -- a
# *targeted* regenerate of exactly those files is strictly better than
# today's whole-tree regenerate. ``assertion_drift`` / ``collection_error``
# / ``unknown`` are genuine reasoning cases (which file? what contract?) and
# always take the bounded ReAct path, degrading to ``escalate``.
_DETERMINISTIC_ACTION_BY_CLASS = {
    "runtime_failure": "regenerate_files",
}

# Read-only tool budgets for the bounded ReAct loop.
_TOOL_READ_LIMIT = 4000
_TOOL_GREP_MAX_HITS = 20
_TOOL_GREP_MAX_FILES = 400

# Directories never worth scanning in the repo tools, and the first-party
# source suffixes the grep tool walks.
_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "dist", "build",
    "__pycache__", ".cgx", ".pytest_cache", ".mypy_cache",
})
_SOURCE_SUFFIXES = frozenset({".py", ".js", ".jsx", ".ts", ".tsx"})

# Bound the manifest list shown to the model so a large tree does not blow
# a small model's context window.
_MANIFEST_SHOWN = 60

# (input key, gate token, expected artifact kind) in router-preference
# order -- the runtime/api/smoke/verify report that drove the repair.
_SOURCE_REPORTS: Tuple[Tuple[str, str, ArtifactKind], ...] = (
    ("runtime_artifact_id", GATE_RUNTIME, ArtifactKind.RUNTIME_REPORT),
    ("api_check_artifact_id", GATE_API_CHECK, ArtifactKind.API_CHECK_REPORT),
    ("smoke_artifact_id", GATE_SMOKE, ArtifactKind.SMOKE_REPORT),
    ("verify_artifact_id", GATE_VERIFY, ArtifactKind.VERIFY_REPORT),
)


@register_executor(TaskKind.DIAGNOSE)
def run_diagnose(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Emit one typed ``DIAGNOSIS`` for the upstream gate failure.

    Deterministic-first, then a bounded read-only ReAct fallback; on a
    missing/crashed provider it degrades to ``escalate``. Pure: the
    runner persists the artifact + outputs; nothing is written to disk.
    """
    if not deps.project_root:
        return ExecutorResult(failure="DIAGNOSE requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(failure="DIAGNOSE requires a session store in deps")

    loaded = _load_failure_context(task, deps)
    if isinstance(loaded, ExecutorResult):
        return loaded
    fc, source_key, source_id, scaffold_artifact_id = loaded
    budget = LoopBudget.from_inputs(task.inputs)
    attempt = budget.repair_attempt or 1
    root = Path(deps.project_root)

    # Working memory: what earlier rounds on this chain already tried. The
    # trailing proposal's outcome is resolved now the current signature is
    # known, so a fix that left the identical failure standing is a proven
    # dead end DIAGNOSE will not re-propose (design §7).
    ledger, prior_ledger_id = _load_ledger(task, deps, budget)
    ledger = ledger.finalize_pending(fc.failure_signature)

    diagnosis = _deterministic_diagnosis(fc, root)
    used_model = False
    if diagnosis is None:
        diagnosis, used_model = _react_diagnose(fc, deps, root, task, ledger)
    diagnosis = _guard_repeat(diagnosis, ledger, fc)

    diagnosis.setdefault("failure_signature", fc.failure_signature)
    action = str(diagnosis.get("minimal_action") or "escalate")
    emit_trace(
        "diagnose_verdict", minimal_action=action,
        confidence=diagnosis.get("confidence"),
        signature=fc.failure_signature, used_model=used_model,
        gate=fc.gate, classification=fc.classification,
        prior_attempts=len(ledger.attempts))

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.DIAGNOSIS,
        content=diagnosis,
    )
    ledger_fact = _append_ledger(
        task, deps, ledger, prior_ledger_id, diagnosis, fc, action)
    return ExecutorResult(
        outputs={
            "diagnosis_artifact_id": artifact.artifact_id,
            "minimal_action": action,
            "failure_signature": fc.failure_signature,
            "repair_attempt": attempt,
            "confidence": diagnosis.get("confidence"),
            "target_files": list(diagnosis.get("target_files") or []),
            "used_model": used_model,
            "repair_ledger_fact_id": ledger_fact.fact_id,
            source_key: source_id,
            "scaffold_artifact_id": scaffold_artifact_id,
        },
        facts=[ledger_fact],
        artifact=artifact,
    )


# --------------------- context loading ---------------------

def _load_failure_context(
        task: TaskNode, deps: ExecutorDeps,
) -> Union[ExecutorResult, Tuple[FailureContext, str, str, Optional[str]]]:
    """Resolve the driving report and fold it into a ``FailureContext``.

    Returns ``(fc, source_key, source_id, scaffold_artifact_id)`` on
    success, or an :class:`ExecutorResult` carrying a hard failure when the
    task carries no source report id or the referenced artifact is missing
    / the wrong kind. The reports are probed in router-preference order
    (runtime -> api_check -> smoke -> verify), matching REPAIR.
    """
    inputs = task.inputs or {}
    for source_key, gate, expected_kind in _SOURCE_REPORTS:
        source_id = str(inputs.get(source_key) or "").strip()
        if not source_id:
            continue
        artifact = deps.store.get_artifact(source_id)
        if artifact is None or artifact.kind is not expected_kind:
            return ExecutorResult(
                failure=f"DIAGNOSE: artifact {source_id!r} missing or wrong "
                        f"kind (need {expected_kind.value})")
        content = dict(artifact.content or {})
        classification = (str(inputs.get("classification") or "").strip()
                          or None)
        scaffold_artifact_id = (content.get("scaffold_artifact_id")
                                or inputs.get("scaffold_artifact_id"))
        fc = FailureContext.from_report(
            gate, content,
            goal=_session_goal(task, deps),
            manifest_files=_manifest_files(content, deps),
            installed_packages=_installed_packages(content, deps),
            classification=classification,
        )
        return fc, source_key, source_id, scaffold_artifact_id
    return ExecutorResult(
        failure="DIAGNOSE missing a source report id in inputs (need one of "
                "runtime/api_check/smoke/verify_artifact_id)")


def _session_goal(task: TaskNode, deps: ExecutorDeps) -> str:
    """Best-effort read of the owning session's original objective."""
    try:
        session = deps.store.get_session(task.session_id)
    except Exception:  # pragma: no cover - defensive: store hiccup
        return ""
    if session is None:
        return ""
    return str(getattr(session, "original_objective", "") or "")


def _manifest_files(content: Dict[str, Any], deps: ExecutorDeps) -> List[str]:
    """The project's file list: the report's ``applied_files``, else the
    nearest SCAFFOLD_PATCHES ``generated`` / ``diffs`` entries."""
    files = [str(f) for f in (content.get("applied_files") or []) if f]
    if files:
        return files
    scaffold_id = str(content.get("scaffold_artifact_id") or "").strip()
    if not scaffold_id:
        return files
    art = deps.store.get_artifact(scaffold_id)
    if art is None or art.kind is not ArtifactKind.SCAFFOLD_PATCHES:
        return files
    entries = ((art.content or {}).get("generated")
               or (art.content or {}).get("diffs") or [])
    for entry in entries:
        if isinstance(entry, dict) and entry.get("file"):
            rel = str(entry["file"])
            if rel not in files:
                files.append(rel)
    return files


def _installed_packages(
        content: Dict[str, Any], deps: ExecutorDeps) -> List[str]:
    """Package names from the upstream BUILD_REPORT (names + resolved)."""
    build_id = str(content.get("build_artifact_id") or "").strip()
    if not build_id:
        return []
    art = deps.store.get_artifact(build_id)
    if art is None or art.kind is not ArtifactKind.BUILD_REPORT:
        return []
    bc = art.content or {}
    names = [str(n) for n in (bc.get("installed_packages") or []) if n]
    for entry in bc.get("resolved_packages") or []:
        if isinstance(entry, dict) and entry.get("name"):
            nm = str(entry["name"])
            if nm not in names:
                names.append(nm)
    return names


# --------------------- repair ledger (working memory) ---------------------

def _load_ledger(
        task: TaskNode, deps: ExecutorDeps, budget: LoopBudget,
) -> Tuple[RepairLedger, Optional[str]]:
    """Load the repair chain's ledger, returning ``(ledger, prior_id)``.

    Prefers the fact id threaded on the chain
    (``LoopBudget.repair_ledger_fact_id``); on a fresh chain -- or a
    persisted-then-resumed session where the id was lost -- it falls back to
    the newest non-stale ``REPAIR_LEDGER`` fact. Returns an empty ledger
    (and ``None`` id) when none exists yet, so the first round starts clean.
    """
    if deps.store is None:
        return RepairLedger(), None
    try:
        kb = deps.store.load_kb(task.session_id)
    except Exception:  # pragma: no cover - defensive: store hiccup
        logger.exception("DIAGNOSE: load_kb failed resolving repair ledger")
        return RepairLedger(), budget.repair_ledger_fact_id
    fid = budget.repair_ledger_fact_id
    if fid and fid in kb.facts and kb.facts[fid].kind is FactKind.REPAIR_LEDGER:
        return RepairLedger.from_content(kb.facts[fid].content), fid
    ledgers = [f for f in kb.of_kind(FactKind.REPAIR_LEDGER) if not f.stale]
    if ledgers:
        latest = max(ledgers, key=lambda f: f.created_at)
        return RepairLedger.from_content(latest.content), latest.fact_id
    return RepairLedger(), None


def _attempt_targets(diagnosis: Dict[str, Any], action: str) -> List[str]:
    """The concrete targets a verdict acts on, keyed by ``minimal_action``."""
    if action == "add_dependency":
        return list(diagnosis.get("add_dependencies") or [])
    if action == "remove_dependency":
        return list(diagnosis.get("remove_dependencies") or [])
    return list(diagnosis.get("target_files") or [])


def _guard_repeat(diagnosis: Dict[str, Any], ledger: RepairLedger,
                  fc: FailureContext) -> Dict[str, Any]:
    """Degrade a verdict that repeats a proven dead end to ``escalate``.

    The single biggest win of the ledger: if this exact
    ``(action, targets)`` already left the identical failure standing,
    proposing it again would just churn the loop, so hand off to the
    existing whole-tree regenerate / re-plan path instead.
    """
    action = str(diagnosis.get("minimal_action") or "")
    if action in ("", "escalate"):
        return diagnosis
    if ledger.has_attempted(action, _attempt_targets(diagnosis, action)):
        escalated = _escalate(fc)
        escalated["root_cause"] = (
            f"{fc.classification or 'unknown'}: prior {action} on the same "
            "targets already left this failure standing")
        escalated["rationale"] = (
            "The repair ledger shows this action + targets was already tried "
            "and did not change the failure; escalating rather than repeating "
            "a proven dead end.")
        return escalated
    return diagnosis


def _append_ledger(
        task: TaskNode, deps: ExecutorDeps, ledger: RepairLedger,
        prior_ledger_id: Optional[str], diagnosis: Dict[str, Any],
        fc: FailureContext, action: str) -> Fact:
    """Append this round's proposal and return the fresh ledger fact.

    Append-only: a new ``REPAIR_LEDGER`` fact carries the whole (finalized
    prior + newly-proposed) chain, and the superseded fact is marked stale
    so the facts view shows one live ledger per chain. The runner persists
    the returned fact; the new id is threaded forward via outputs.
    """
    updated = ledger.append(
        action, _attempt_targets(diagnosis, action), fc.failure_signature,
        rationale=str(diagnosis.get("rationale") or ""))
    fact = Fact.new(
        session_id=task.session_id, kind=FactKind.REPAIR_LEDGER,
        content=updated.to_content(), surfaced_in_task_id=task.task_id)
    if prior_ledger_id and prior_ledger_id != fact.fact_id \
            and deps.store is not None:
        try:
            deps.store.mark_facts_stale(task.session_id, [prior_ledger_id])
        except Exception:  # pragma: no cover - defensive: store hiccup
            logger.exception("DIAGNOSE: superseding prior ledger fact failed")
    return fact


# --------------------- deterministic-first ---------------------

def _deterministic_diagnosis(
        fc: FailureContext, root: Path) -> Optional[Dict[str, Any]]:
    """Return a model-free DIAGNOSIS when the fix location is unambiguous.

    Only the import/boot classes in :data:`_DETERMINISTIC_ACTION_BY_CLASS`
    whose traceback already names existing first-party files qualify; the
    verdict is a *targeted* regenerate of exactly those files. Everything
    else returns ``None`` so the bounded ReAct loop runs.
    """
    action = _DETERMINISTIC_ACTION_BY_CLASS.get(fc.classification)
    if action is None:
        return None
    targets = _regen_targets(fc.traceback_files, root)
    if not targets:
        return None
    return {
        "root_cause": (f"{fc.classification}: traceback localizes to "
                       + ", ".join(targets)),
        "minimal_action": action,
        "target_files": targets,
        "rationale": ("Deterministic classifier + traceback localization "
                      f"named the file(s) for a {fc.classification} failure, "
                      "so a targeted regenerate is proposed with no model "
                      "call."),
        "confidence": 0.8,
    }


def _regen_targets(paths: Tuple[str, ...], root: Path) -> List[str]:
    """Existing first-party ``.py`` traceback files, implementation first.

    A regenerate should re-author the module that broke, not the test that
    surfaced it; test files are used only when the traceback named nothing
    else.
    """
    impl: List[str] = []
    tests: List[str] = []
    for rel in paths:
        rel = str(rel).strip()
        if not rel or not rel.endswith(".py") or not (root / rel).is_file():
            continue
        (tests if _looks_like_test(rel) else impl).append(rel)
    return impl or tests


def _looks_like_test(rel: str) -> bool:
    base = rel.rsplit("/", 1)[-1]
    return (base.startswith("test_") or base.endswith("_test.py")
            or rel.startswith("tests/") or "/tests/" in rel)


# --------------------- bounded ReAct loop ---------------------

_SYSTEM_PROMPT = (
    "You are a senior engineer diagnosing why an automatically-generated "
    "project fails its build/test gate. Reason from the failure output, the "
    "file manifest, and the installed packages. You MAY gather evidence with "
    "a few read-only tools, then you MUST emit one final verdict.\n\n"
    "Return a SINGLE JSON object each turn -- never prose.\n"
    "To use a tool: {\"tool\": \"read_file\", \"path\": \"pkg/mod.py\"} or "
    "{\"tool\": \"grep_files\", \"pattern\": \"some_symbol\"} or "
    "{\"tool\": \"inspect_packages\"}.\n"
    "To finish: {\"minimal_action\": <one of patch_files|add_dependency|"
    "remove_dependency|adjust_manifest|regenerate_files|escalate>, "
    "\"root_cause\": <one line>, \"target_files\": [..], "
    "\"add_dependencies\": [..], \"remove_dependencies\": [..], "
    "\"rationale\": <one line>, \"confidence\": <0..1>}.\n"
    "Choose the SMALLEST action that fixes the root cause: add_dependency "
    "for a genuinely missing package, remove_dependency for an unrunnable "
    "one, adjust_manifest/regenerate_files for authoring bugs, patch_files "
    "for a localized logic fix. When you are not confident, return "
    "minimal_action \"escalate\"."
)


def _react_diagnose(
        fc: FailureContext, deps: ExecutorDeps, root: Path,
        task: TaskNode, ledger: RepairLedger) -> Tuple[Dict[str, Any], bool]:
    """Run the bounded read-only ReAct loop; return ``(diagnosis, used_model)``.

    Degrades to an ``escalate`` verdict on a missing provider, a provider
    crash, unparseable output, or an exhausted tool budget -- so behavior is
    never worse than today's regenerate ladder.
    """
    if deps.provider is None:
        return _escalate(fc), False
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _render_context(fc, ledger)},
    ]
    for tool_calls in range(DIAGNOSE_STEPS + 1):
        parsed = _diagnose_call(deps.provider, messages)
        if not isinstance(parsed, dict):
            return _escalate(fc), True
        if parsed.get("minimal_action"):
            return _normalize_verdict(parsed, fc), True
        tool = str(parsed.get("tool") or "").strip()
        if not tool or tool_calls >= DIAGNOSE_STEPS:
            return _escalate(fc), True
        observation = _run_tool(tool, parsed, fc, root)
        emit_trace("diagnose_step", step=tool_calls + 1, tool=tool,
                   signature=fc.failure_signature, gate=fc.gate)
        messages.append({"role": "assistant",
                         "content": json.dumps(parsed)[:1500]})
        messages.append({"role": "user",
                         "content": ("OBSERVATION:\n" + observation)[:3000]})
    return _escalate(fc), True


def _render_context(fc: FailureContext, ledger: RepairLedger) -> str:
    manifest = ", ".join(fc.manifest_files[:_MANIFEST_SHOWN]) or "(unknown)"
    pkgs = ", ".join(fc.installed_packages) or "(none recorded)"
    tb = ", ".join(fc.traceback_files) or "(none localized)"
    return (
        f"GOAL: {fc.goal or '(unspecified)'}\n"
        f"GATE: {fc.gate}\n"
        f"CLASSIFICATION: {fc.classification}\n"
        f"FAILURE SIGNATURE: {fc.failure_signature}\n"
        f"TRACEBACK FILES: {tb}\n"
        f"INSTALLED PACKAGES: {pkgs}\n"
        f"FILE MANIFEST: {manifest}\n"
        f"PRIOR REPAIR ATTEMPTS (do NOT repeat a still_failing action):\n"
        f"{ledger.render()}\n\n"
        f"FAILURE OUTPUT:\n{fc.failure_text}"
    )


def _diagnose_call(
        provider: Any, messages: List[Dict[str, str]]
) -> Optional[Dict[str, Any]]:
    """One schema-constrained provider turn; ``None`` on crash/unparseable."""
    try:
        resp = provider.chat(messages=messages, temperature=0.0,
                             max_tokens=800, force_json=True)
    except Exception:  # pragma: no cover - defensive: provider hiccup
        logger.exception("DIAGNOSE: provider call crashed")
        return None
    raw = (resp or {}).get("content", "") if isinstance(resp, dict) else ""
    from cgx.answer.engine import _extract_json_object
    parsed = _extract_json_object(raw or "")
    return parsed if isinstance(parsed, dict) else None


# --------------------- read-only tools ---------------------

def _run_tool(tool: str, parsed: Dict[str, Any], fc: FailureContext,
              root: Path) -> str:
    if tool == "inspect_packages":
        return "installed packages: " + (
            ", ".join(fc.installed_packages) or "(none recorded)")
    if tool == "read_file":
        return _tool_read_file(str(parsed.get("path") or ""), root)
    if tool == "grep_files":
        return _tool_grep(str(parsed.get("pattern") or ""), root)
    return (f"unknown tool {tool!r}; available: read_file, grep_files, "
            "inspect_packages")


def _tool_read_file(rel: str, root: Path) -> str:
    rel = rel.strip()
    if not rel:
        return "read_file requires a 'path'"
    target = (root / rel).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return f"refused: {rel!r} is outside the project"
    if not target.is_file():
        return f"no such file: {rel}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        return f"could not read {rel}"
    return text[:_TOOL_READ_LIMIT]


def _tool_grep(pattern: str, root: Path) -> str:
    pattern = pattern.strip()
    if not pattern:
        return "grep_files requires a 'pattern'"
    hits: List[str] = []
    for path in _iter_source_files(root):
        try:
            lines = path.read_text(
                encoding="utf-8", errors="replace").splitlines()
        except Exception:  # pragma: no cover - defensive
            continue
        for i, line in enumerate(lines, 1):
            if pattern in line:
                rel = path.relative_to(root).as_posix()
                hits.append(f"{rel}:{i}: {line.strip()[:160]}")
                if len(hits) >= _TOOL_GREP_MAX_HITS:
                    return "\n".join(hits)
    return "\n".join(hits) if hits else f"no matches for {pattern!r}"


def _iter_source_files(root: Path):
    count = 0
    for path in sorted(root.rglob("*")):
        if count >= _TOOL_GREP_MAX_FILES:
            return
        if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        count += 1
        yield path


# --------------------- verdict shaping ---------------------

def _normalize_verdict(
        parsed: Dict[str, Any], fc: FailureContext) -> Dict[str, Any]:
    """Coerce a raw model object into the DIAGNOSIS schema (design §5).

    An out-of-enum ``minimal_action`` (a small model hallucinating) is
    treated as no verdict and degrades to ``escalate``.
    """
    action = str(parsed.get("minimal_action") or "").strip()
    if action not in MINIMAL_ACTIONS:
        return _escalate(fc)
    edits = parsed.get("manifest_edits")
    return {
        "root_cause": (str(parsed.get("root_cause") or "").strip()
                       or fc.classification),
        "minimal_action": action,
        "target_files": _clean_str_list(parsed.get("target_files")),
        "add_dependencies": _clean_str_list(parsed.get("add_dependencies")),
        "remove_dependencies":
            _clean_str_list(parsed.get("remove_dependencies")),
        "remove_tests": _clean_str_list(parsed.get("remove_tests")),
        "manifest_edits": edits if isinstance(edits, list) else [],
        "rationale": str(parsed.get("rationale") or "").strip(),
        "confidence": _clean_confidence(parsed.get("confidence")),
    }


def _escalate(fc: FailureContext) -> Dict[str, Any]:
    """The strictly-additive fallback: today's regenerate / re-plan path."""
    return {
        "root_cause": fc.classification or "unknown",
        "minimal_action": "escalate",
        "target_files": list(fc.traceback_files),
        "rationale": ("No deterministic fix and no bounded diagnosis was "
                      "produced; handing off to the existing regenerate / "
                      "re-plan path."),
        "confidence": 0.0,
    }


def _clean_str_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[str] = []
    for item in value:
        s = str(item).strip()
        if s and s not in out:
            out.append(s)
    return out


def _clean_confidence(value: Any) -> Optional[float]:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, conf))
