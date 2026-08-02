"""Greenfield failure-recovery edges for the session router.

Every helper here backs one recovery edge of the greenfield write loop:
the REPAIR verdict splices (install-deps / regenerate / terminal), the
dropped-file and contract regenerates for SCAFFOLD/APPLY, the crashed
SCAFFOLD resume, the bounded DECOMPOSE retry/re-plan escalations, and
the lesson emission for a repairing VERIFY-pass. They are consulted by
:class:`cgx.session.router.Router` -- the completion-time overrides in
priority order via ``router._COMPLETION_GUARDS``, the failure-time ones
directly from :meth:`Router.on_task_failed` -- and each returns ``[]``
to decline so the caller falls through to the normal table-driven edge.

All helpers are pure: they read the completed/failed task plus the
session's task list and return typed :mod:`cgx.session.actions` values;
nothing here touches the store or an LLM.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from cgx.session.actions import (
    CreateTask,
    RecordLesson,
    RouterAction,
    UpdateSessionStatus,
    UpdateTaskStatus,
)
from cgx.session.budget import LoopBudget
from cgx.session.models import (
    DecisionKind,
    SessionMode,
    SessionStatus,
    TaskKind,
    TaskNode,
    TaskNodeStatus,
)


def _make_budget_ask(over_task: TaskNode, reason: str) -> TaskNode:
    """Build the ASK_USER that surfaces a paused-on-budget session."""
    prior_goal = (over_task.inputs.get("prior_goal")
                  or over_task.inputs.get("goal"))
    return TaskNode.new(
        session_id=over_task.session_id,
        kind=TaskKind.ASK_USER,
        name="Session budget exhausted",
        description=(f"The session hit its {reason} before finishing. "
                     "Autonomous work is paused -- review the progress so "
                     "far and decide whether to continue or stop."),
        parent_task_id=over_task.parent_task_id,
        inputs={
            "expected_kind": DecisionKind.FREEFORM.value,
            "reason": reason,
            "prior_goal": prior_goal,
            "over_task_id": over_task.task_id,
        },
    )


def _repair_install_deps_actions(
        completed: TaskNode) -> List[RouterAction]:
    """Return the router actions that execute an install-deps verdict.

    An ``install_deps`` verdict (set by the REPAIR executor for an
    API_CHECK ``missing_dependency`` failure) tells the router to
    re-provision the environment rather than rewrite code: it re-queues
    a BOOTSTRAP_ENV whose preflight installs the absent third-party
    imports and syncs requirements.txt. BOOTSTRAP_ENV's own successor
    (:func:`cgx.session.router._bootstrap_to_api_check`) then re-probes
    the same symbols, so a successful install flows straight back into
    SMOKE/VERIFY while the shared ``repair_attempt`` +
    ``prior_failure_signatures`` budget on API_CHECK prevents an
    install loop. Returns an empty list for any other strategy so the
    dispatcher falls through to the regenerate / patch / ASK_USER
    paths.
    """
    outputs = completed.outputs or {}
    strategy = str(outputs.get("strategy") or "").strip()
    if strategy != "install_deps":
        return []
    inputs = completed.inputs or {}
    budget = LoopBudget.from_inputs(inputs)
    budget = budget.with_repair_attempt(
        int(outputs.get("repair_attempt") or budget.repair_attempt or 1))
    missing = [str(m) for m in outputs.get("missing_modules") or []]
    boot = TaskNode.new(
        session_id=completed.session_id,
        kind=TaskKind.BOOTSTRAP_ENV,
        name="Install missing dependencies",
        description=("Re-provision the project venv to install the "
                     "third-party package(s) the applied files import "
                     "but that are absent from the environment, then "
                     "re-probe via API_CHECK."),
        parent_task_id=completed.task_id,
        inputs={
            "apply_artifact_id": inputs.get("apply_artifact_id"),
            "plan_artifact_id": inputs.get("plan_artifact_id"),
            "scaffold_artifact_id": inputs.get("scaffold_artifact_id"),
            "prior_goal": inputs.get("prior_goal"),
            "mode": inputs.get("mode") or SessionMode.GREENFIELD.value,
            "missing_modules": missing,
            **budget.repair_chain_inputs(),
        },
    )
    return [CreateTask(boot)]


def _repair_regenerate_actions(completed: TaskNode,
                               tasks: List[TaskNode]) -> List[RouterAction]:
    """Return the router actions that execute a regenerate verdict.

    A regenerate verdict (set by the REPAIR executor when patching is
    impossible or too large to be safe) tells the router to abandon
    the failing subtree under the nearest SCAFFOLD ancestor and
    re-queue a fresh SCAFFOLD with the constraint payload folded into
    its inputs. The dispatcher in
    :meth:`cgx.session.router.Router.on_task_completed` falls back to
    the regular patch / ASK_USER table-driven path when
    this function returns an empty list, so the four early-exit cases
    below (wrong strategy, no SCAFFOLD ancestor, budget exhausted, or
    nothing to abandon) degrade gracefully.

    The budget gate reads a dedicated ``repair_regenerate_attempt``
    counter -- **not** the syntax-churn ``regenerate_attempt`` spent by
    :func:`_scaffold_failed_files_actions` / :func:`_apply_failed_files_actions`
    making the tree parse. A scaffold that burned its whole syntax budget
    converging to a clean, applied tree must still afford the correctness
    loop its full :data:`~cgx.session.budget.REPAIR_REGENERATE_BUDGET`;
    the regenerated
    SCAFFOLD carries ``repair_regenerate_attempt + 1`` so this loop stays
    finite even though ``repair_attempt`` does not survive the regenerate.
    ``prior_failure_signatures`` *do* survive: they are folded into the
    new SCAFFOLD's inputs (and threaded down its APPLY -> ... -> VERIFY
    chain) so a regenerated tree that fails on the identical signature
    is stopped by the flap detector instead of looping until the
    regenerate budget is spent.
    """
    from cgx.session.repair.propose import propose_regenerate  # local import: dep direction

    outputs = completed.outputs or {}
    strategy = str(outputs.get("strategy") or "").strip()
    if strategy != "regenerate":
        return []
    extra_constraints = outputs.get("extra_constraints")
    if not isinstance(extra_constraints, dict):
        extra_constraints = {}
    scaffold = _find_scaffold_ancestor(completed, tasks)
    if scaffold is None:
        return []
    scaffold_budget = LoopBudget.from_inputs(scaffold.inputs)
    if scaffold_budget.repair_regenerate_exhausted:
        return []
    abandon_targets = _collect_descendants(scaffold.task_id, tasks)
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in abandon_targets:
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    # A classification whose fix is a *missing* file (a bundler entry
    # module that was never generated) names it under ``missing_files``;
    # thread it through so the regenerated SCAFFOLD extends the manifest
    # instead of re-authoring the same unbuildable tree.
    missing_files = extra_constraints.get("missing_files")
    if not isinstance(missing_files, list):
        missing_files = None
    new_scaffold = propose_regenerate(
        scaffold, extra_constraints,
        additional_files=missing_files,
        prior_failure_signatures=LoopBudget.from_inputs(
            completed.inputs).prior_failure_signatures)
    new_scaffold.inputs["repair_regenerate_attempt"] = (
        scaffold_budget.spend_repair_regenerate().repair_regenerate_attempt)
    actions.append(CreateTask(new_scaffold))
    return actions


def _repair_terminal_failure_actions(
        completed: TaskNode) -> List[RouterAction]:
    """Fail the session when REPAIR has no automated recovery left.

    Reached from :meth:`cgx.session.router.Router.on_task_completed`
    only after the
    install-deps and regenerate branches have both declined. A REPAIR
    that produced no applicable patch (``can_apply`` False) means every
    automated path -- patch, regenerate, dependency install -- is
    exhausted. Asking the user to hand-edit AI-generated code is never a
    valid recovery, so the loop terminates: the REPAIR node goes
    ``FAILED`` (carrying the classification for the UI) and the whole
    session flips to ``FAILED``. Returns an empty list when the patch is
    applicable so the caller falls through to the APPLY successor.
    """
    outputs = completed.outputs or {}
    if bool(outputs.get("can_apply")):
        return []
    classification = str(outputs.get("classification") or "unknown")
    error = ("Automated repair could not produce a patch "
             f"(classification={classification}); no regenerate or "
             "dependency-install path remained.")
    # An escalated verdict (e.g. an unrecognized collection_error) carries
    # a human-readable rationale explaining why no automated path applies;
    # surface it so the terminal failure is actionable rather than opaque.
    rationale = str(outputs.get("rationale") or "").strip()
    if rationale:
        error = f"{error} {rationale}"
    return [
        UpdateTaskStatus(task_id=completed.task_id,
                         status=TaskNodeStatus.FAILED, error=error),
        UpdateSessionStatus(session_id=completed.session_id,
                            status=SessionStatus.FAILED),
    ]


def _apply_failed_files_actions(completed: TaskNode,
                                tasks: List[TaskNode]) -> List[RouterAction]:
    """Regenerate (or terminally fail) a greenfield APPLY that dropped files.

    The APPLY executor refuses to write a file whose source does not
    parse as valid Python, recording it under ``failed_files`` and
    surfacing a non-zero ``failed_count`` while still applying the rest.
    Continuing to BOOTSTRAP_ENV / VERIFY with a core module silently
    missing guarantees a downstream collection error, so any greenfield
    APPLY that dropped a file re-scaffolds within
    :data:`~cgx.session.budget.REGENERATE_BUDGET` instead of limping
    forward -- unless :func:`_scaffold_failure_signature` shows the
    previous attempt already produced this exact failure, in which case
    the remaining attempts are skipped for the escalation below. When the
    regenerate budget is spent the router escalates once to a revised
    manifest via :func:`_replan_or_fail` (a fresh DECOMPOSE); when the
    re-plan budget is also spent that helper proceeds with the survivors
    rather than discarding the run, and only fails terminally when nothing
    usable was generated. When no SCAFFOLD ancestor exists the session
    fails terminally -- it cannot re-scaffold a tree it cannot find.
    Returns an empty list for explore mode or a clean apply so the
    dispatcher takes the normal APPLY -> VERIFY edge.
    """
    from cgx.session.repair.propose import propose_regenerate  # dep direction

    outputs = completed.outputs or {}
    mode = str(completed.inputs.get("mode") or "").strip()
    if mode != SessionMode.GREENFIELD.value:
        return []
    failed_count = int(outputs.get("failed_count") or 0)
    if failed_count <= 0:
        return []
    scaffold = _find_scaffold_ancestor(completed, tasks)
    if scaffold is None:
        return [UpdateSessionStatus(
            session_id=completed.session_id, status=SessionStatus.FAILED)]
    scaffold_outputs = scaffold.outputs or {}
    constraint = _invalid_scaffold_constraint(
        failed_count,
        apply_failed=outputs.get("failed_files"),
        scaffold_failed=scaffold_outputs.get("failed"))
    # Foundational-file guard: a dropped environment manifest is never
    # fixable by the per-file regenerate loop (re-asking the same model
    # reproduces the drop), so skip that budget and escalate straight to a
    # re-plan that can restructure the manifest.
    foundational = _dropped_foundational_files(
        scaffold_outputs.get("failed"), outputs.get("failed_files"))
    if foundational:
        return _replan_or_fail(
            completed, tasks, scaffold=scaffold,
            failure_note=_foundational_failure_note(foundational))
    budget = LoopBudget.from_inputs(scaffold.inputs)
    signature = _scaffold_failure_signature(
        scaffold_outputs.get("failed"), outputs.get("failed_files"))
    if budget.regenerate_exhausted or (signature and budget.seen(signature)):
        return _replan_or_fail(
            completed, tasks, scaffold=scaffold,
            failure_note=str(constraint.get("rationale") or ""))
    regen_files = _failed_scaffold_paths(
        scaffold_outputs.get("failed"), outputs.get("failed_files"))
    prior_id = str(
        scaffold_outputs.get("scaffold_artifact_id") or "").strip()
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in _collect_descendants(scaffold.task_id, tasks):
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    actions.append(CreateTask(propose_regenerate(
        scaffold, constraint,
        regenerate_files=regen_files,
        prior_scaffold_artifact_id=prior_id,
        prior_failure_signatures=_appended_signature(budget, signature))))
    return actions


def _fold_failure_into_goal(prior_goal: str, failure_note: str) -> str:
    """Append a re-plan failure note to a goal so DECOMPOSE can react.

    The revised goal keeps the original objective verbatim and adds a
    short, explicit note describing why the prior manifest could not be
    scaffolded so the planner restructures the plan (drop the offending
    file, split a layer, pick a simpler stack) instead of re-emitting the
    same broken manifest.
    """
    goal = (prior_goal or "").strip()
    note = (failure_note or "").strip()
    if not note:
        return goal
    banner = ("The previous plan could not be scaffolded. Revise the file "
              "manifest to avoid this failure: " + note)
    return f"{goal}\n\n{banner}" if goal else banner


def _decompose_retry_actions(failed: TaskNode) -> List[RouterAction]:
    """Re-queue a DECOMPOSE whose executor marked the failure retryable.

    The retry copies the failed task's requirements/answers wiring and
    folds ``failed.error`` into ``prior_goal`` via
    :func:`_fold_failure_into_goal` so the planner LLM sees exactly what
    invalidated the prior manifest. Bounded by
    :data:`~cgx.session.budget.DECOMPOSE_RETRY_BUDGET` (counter carried in
    ``inputs["decompose_retry"]``); returns an empty list for any other
    task kind or a spent budget so
    :meth:`cgx.session.router.Router.on_task_failed` falls
    through to the terminal ``FAILED``.
    """
    if failed.kind is not TaskKind.DECOMPOSE:
        return []
    budget = LoopBudget.from_inputs(failed.inputs)
    if budget.decompose_retry_exhausted:
        return []
    prior_goal = str(failed.inputs.get("prior_goal") or "").strip()
    answers = failed.inputs.get("answers")
    retry = TaskNode.new(
        session_id=failed.session_id,
        kind=TaskKind.DECOMPOSE,
        name="Revise the work plan",
        description=("Re-plan the file manifest after the prior plan "
                     "failed validation."),
        parent_task_id=failed.task_id,
        inputs={
            "prior_goal": _fold_failure_into_goal(
                prior_goal, str(failed.error or "")),
            "requirements_artifact_id":
                failed.inputs.get("requirements_artifact_id"),
            "answers": dict(answers) if isinstance(answers, dict) else {},
            "decompose_retry": budget.spend_decompose_retry().decompose_retry,
            "replan_attempt": budget.replan_attempt,
            "prior_failure_signatures":
                list(budget.prior_failure_signatures),
        },
    )
    return [CreateTask(retry)]


def _merged_failure_signatures(*nodes: Optional[TaskNode]) -> List[str]:
    """Union the flap ledgers of ``nodes``, order-preserving.

    A re-plan hops out of the repair chain (SCAFFOLD/APPLY -> DECOMPOSE)
    and back into a fresh one, so the ledger has to be carried by hand
    or the new chain is amnesiac: observed live, where a re-planned tree
    reproduced the identical build failure and spent a second full
    repair budget on it before the flap detector -- which had never seen
    the signature on the new chain -- could stop the loop.
    """
    out: List[str] = []
    for node in nodes:
        if node is None:
            continue
        for sig in LoopBudget.from_inputs(node.inputs).prior_failure_signatures:
            if sig and sig not in out:
                out.append(sig)
    return out


def _replan_or_fail(completed: TaskNode, tasks: List[TaskNode], *,
                    scaffold: Optional[TaskNode],
                    failure_note: str) -> List[RouterAction]:
    """Escalate an exhausted regenerate budget to a fresh DECOMPOSE.

    When a SCAFFOLD/APPLY has spent its per-manifest
    :data:`~cgx.session.budget.REGENERATE_BUDGET` the manifest itself is
    the suspect, not the generation of any single file. Before failing
    the session terminally the router escalates *once* (capped by
    :data:`~cgx.session.budget.REPLAN_BUDGET`) to a revised plan: it
    abandons the live
    subtree under the failing SCAFFOLD and spawns a fresh DECOMPOSE whose
    goal folds in ``failure_note`` so the planner can restructure the
    manifest. The ``replan_attempt`` counter threads DECOMPOSE ->
    ASK_USER(APPROVE_PLAN) -> SCAFFOLD (and copies verbatim across
    :func:`propose_regenerate` retries), so a second exhaustion on the
    revised manifest falls through to terminal ``FAILED``. Returns the
    terminal-fail action when no SCAFFOLD lineage exists or the re-plan
    budget is already spent.

    ``prior_failure_signatures`` ride the same path: a revised manifest
    that reproduces a failure the previous plan already hit is not
    progress, and the gate that sees it again must stop the run rather
    than spend a fresh repair budget re-deriving the same dead end.
    """
    fail: List[RouterAction] = [UpdateSessionStatus(
        session_id=completed.session_id, status=SessionStatus.FAILED)]
    if scaffold is None:
        return fail
    budget = LoopBudget.from_inputs(scaffold.inputs)
    if budget.replan_exhausted:
        # Budgets spent. Rather than discard every file that generated
        # cleanly, proceed with the survivors on the normal edge (the
        # empty return lets the dispatcher take SCAFFOLD -> APPLY /
        # APPLY -> VERIFY) whenever the partial scaffold produced output;
        # the dropped files are already surfaced to the UI via the
        # scaffold ``failed_count`` progress beats. Only a scaffold that
        # produced nothing usable is a terminal dead end.
        survivors = int((scaffold.outputs or {}).get("generated_count") or 0)
        return [] if survivors > 0 else fail
    prior_goal = str(scaffold.inputs.get("prior_goal") or "").strip()
    decompose = _find_ancestor_by_kind(scaffold, tasks, TaskKind.DECOMPOSE)
    answers: Dict[str, object] = {}
    if decompose is not None:
        prior_answers = decompose.inputs.get("answers")
        if isinstance(prior_answers, dict):
            answers = dict(prior_answers)
    new_decompose = TaskNode.new(
        session_id=completed.session_id,
        kind=TaskKind.DECOMPOSE,
        name="Revise the work plan",
        description=("Re-plan the file manifest after the prior plan's "
                     "scaffold could not be generated cleanly."),
        parent_task_id=scaffold.task_id,
        inputs={
            "prior_goal": _fold_failure_into_goal(prior_goal, failure_note),
            "requirements_artifact_id":
                scaffold.inputs.get("requirements_artifact_id"),
            "answers": answers,
            "replan_attempt": budget.spend_replan().replan_attempt,
            # Carry the *spent* syntax-churn regenerate count into the
            # revised manifest's chain. Left to reset, a fresh
            # DECOMPOSE -> SCAFFOLD would be born at ``regenerate_attempt=0``
            # and hand the re-planned manifest a whole second
            # ``REGENERATE_BUDGET``, so the total syntax-churn budget would
            # multiply by the number of re-plans. Threading it (like
            # ``replan_attempt`` and the flap ledger below) makes the
            # regenerate budget a per-session ceiling, not a per-manifest one.
            "regenerate_attempt": budget.regenerate_attempt,
            "prior_failure_signatures":
                _merged_failure_signatures(scaffold, completed),
        },
    )
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in _collect_descendants(scaffold.task_id, tasks):
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    actions.append(CreateTask(new_decompose))
    return actions


def _scaffold_failed_files_actions(completed: TaskNode,
                                   tasks: List[TaskNode]) -> List[RouterAction]:
    """Regenerate (or terminally fail) a SCAFFOLD that dropped files.

    The SCAFFOLD executor records every file whose generation crashed
    (e.g. an LLM read timeout) or returned an empty patch under
    ``outputs.failed`` while still emitting the survivors -- so a partial
    generation returns *success* with ``failed_count > 0``. Proceeding to
    APPLY / VERIFY with a required module never written guarantees a
    downstream collection error the test-repair loop cannot fix: there is
    nothing to patch because the file is simply absent. Mirroring
    :func:`_apply_failed_files_actions`, any SCAFFOLD that dropped a file
    re-scaffolds within :data:`~cgx.session.budget.REGENERATE_BUDGET`,
    folding the concrete
    per-file errors into the regenerate constraint so the retry has
    actionable feedback. A retry that reproduces the identical dropped
    file set with the identical error is not feedback the generator can
    act on, so :func:`_scaffold_failure_signature` short-circuits the
    remaining attempts onto the same escalation the spent budget takes.
    When that budget is spent the router escalates
    once to a revised manifest via :func:`_replan_or_fail` (a fresh
    DECOMPOSE); when the re-plan budget is also spent that helper proceeds
    with the survivors on the normal SCAFFOLD -> APPLY edge rather than
    discarding them, failing terminally only when nothing usable was
    generated. Returns an empty list for a clean scaffold so the
    dispatcher takes the normal SCAFFOLD -> APPLY edge.
    """
    from cgx.session.repair.propose import propose_regenerate  # dep direction

    outputs = completed.outputs or {}
    failed_count = int(outputs.get("failed_count") or 0)
    if failed_count <= 0:
        return []
    constraint = _invalid_scaffold_constraint(
        failed_count, apply_failed=None,
        scaffold_failed=outputs.get("failed"))
    # Foundational-file guard (mirrors _apply_failed_files_actions): a
    # dropped environment manifest escalates straight to a re-plan rather
    # than burning the per-file regenerate budget on a file the same model
    # just failed to produce.
    foundational = _dropped_foundational_files(outputs.get("failed"), None)
    if foundational:
        return _replan_or_fail(
            completed, tasks, scaffold=completed,
            failure_note=_foundational_failure_note(foundational))
    budget = LoopBudget.from_inputs(completed.inputs)
    signature = _scaffold_failure_signature(outputs.get("failed"))
    if budget.regenerate_exhausted or (signature and budget.seen(signature)):
        return _replan_or_fail(
            completed, tasks, scaffold=completed,
            failure_note=str(constraint.get("rationale") or ""))
    regen_files = _failed_scaffold_paths(outputs.get("failed"), None)
    prior_id = str(outputs.get("scaffold_artifact_id") or "").strip()
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in _collect_descendants(completed.task_id, tasks):
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    actions.append(CreateTask(propose_regenerate(
        completed, constraint,
        regenerate_files=regen_files,
        prior_scaffold_artifact_id=prior_id,
        prior_failure_signatures=_appended_signature(budget, signature))))
    return actions


# Bracketed lists, quoted literals and digits inside a scaffold gate's
# error carry the *instance* of the fault (which module was hallucinated,
# which file the content duplicated); the fault is the prose around them.
# Stripping them collapses "imports unknown module(s) ['app']" and the
# next attempt's ``['api']`` onto one signature, so a retry that trades
# one hallucination for another is recognised as no progress.
_SIGNATURE_NOISE = re.compile(r"\[[^\]]*\]|'[^']*'|\"[^\"]*\"|[0-9]+")

# Cap on the per-file error prose folded into a signature. Long enough to
# separate two gates that reject the same file, short enough that a tail
# carrying a path or a count cannot make two identical faults look
# distinct.
_SIGNATURE_ERROR_CHARS = 80


def _scaffold_failure_signature(scaffold_failed: object,
                                apply_failed: object = None) -> str:
    """Return a flap signature for a dropped-file set.

    A regenerate is only worth its budget slot if the retry can plausibly
    differ. Observed live: the same file failed the same gate on all
    three ``regenerate_attempt``s, each round paying a full generation
    pass to re-derive the identical rejection. The signature keys on the
    dropped paths plus the normalized error prose, so
    :meth:`~cgx.session.budget.LoopBudget.seen` can route the second
    occurrence to the escalation a spent budget takes -- a re-plan, where
    the manifest, the actual suspect, is rewritten.

    Draws from the same two ``{"file", "error"}`` sources as
    :func:`_invalid_scaffold_constraint` and :func:`_failed_scaffold_paths`
    so a SCAFFOLD-dropped file and an APPLY-dropped one are keyed
    identically, and is order-insensitive: the same files rejected for the
    same reasons hash the same however the executor ordered them.

    Returns ``""`` when nothing usable can be derived, which callers read
    as "no flap evidence" and leave to the normal budget.
    """
    parts: List[str] = []
    seen: set = set()
    for entry in _dropped_file_entries(scaffold_failed, apply_failed):
        path = str(entry.get("file") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        err = _SIGNATURE_NOISE.sub(" ", str(entry.get("error") or "")).lower()
        parts.append(
            f"{path}:{' '.join(err.split())[:_SIGNATURE_ERROR_CHARS]}")
    if not parts:
        return ""
    raw = ";".join(sorted(parts)).encode("utf-8", errors="replace")
    return "scaffold|" + hashlib.sha1(raw).hexdigest()[:16]


def _appended_signature(budget: LoopBudget,
                        signature: str) -> Optional[List[str]]:
    """Return the flap ledger to thread onto a regenerated SCAFFOLD.

    Recording the signature is what makes the *next* round's
    ``budget.seen`` check meaningful; without it every attempt looks
    like the first. Returns ``None`` for an underivable or
    already-recorded signature so ``propose_regenerate`` leaves the
    inherited ledger untouched.
    """
    if not signature or budget.seen(signature):
        return None
    return list(budget.prior_failure_signatures) + [signature]


# Contract-warning kinds a regenerate can actually satisfy: a declared
# function/constant/schema names the module that must provide it, so the
# retry has a concrete, satisfiable target. Endpoints are omitted -- a
# planner placeholder path (e.g. ``/api/x``) is frequently unsatisfiable
# and would only spin the budget.
_CONTRACT_REGENERATE_KINDS = {"function", "constant", "schema"}


def _actionable_contract_warnings(
        outputs: Dict[str, object]) -> List[Dict[str, object]]:
    """Return the contract warnings a regenerate can act on.

    Keeps only a declared function/constant/schema that names a concrete
    ``module`` -- the offending module is known and the goal is
    satisfiable. Endpoint warnings and any warning without a module are
    dropped so an unsatisfiable planner contract never forces a retry.
    """
    out: List[Dict[str, object]] = []
    warnings = outputs.get("contract_warnings")
    if not isinstance(warnings, (list, tuple)):
        return out
    for w in warnings:
        if not isinstance(w, dict):
            continue
        if (w.get("kind") in _CONTRACT_REGENERATE_KINDS
                and str(w.get("module") or "").strip()):
            out.append(w)
    return out


def _contract_regenerate_constraint(
        warnings: List[Dict[str, object]]) -> Dict[str, object]:
    """Fold unmet contract items into a SCAFFOLD regenerate constraint."""
    items = [f"{w.get('kind')} {w.get('name')!r} in module "
             f"{str(w.get('module'))!r}" for w in warnings[:6]]
    rationale = ("the generated files do not satisfy declared contract(s): "
                 + "; ".join(items)
                 + ". Implement each named symbol in its module.")
    return {"kind": "unmet_contract", "rationale": rationale,
            "unmet_contracts": items}


def _scaffold_contract_regenerate_actions(
        completed: TaskNode,
        tasks: List[TaskNode]) -> List[RouterAction]:
    """Regenerate a clean-but-noncompliant SCAFFOLD once per budget step.

    Complements :func:`_scaffold_failed_files_actions`: that path owns a
    scaffold that *dropped* files (``failed_count > 0``); this one handles
    a scaffold where every file generated but a file-attributable contract
    (a declared function/constant/schema whose named module never provides
    it) is unmet, folding the unmet contracts in as a whole-tree
    regenerate constraint. Bounded by
    :data:`~cgx.session.budget.REGENERATE_BUDGET`, by the same flap
    signature the dropped-file paths use (a retry that leaves the
    identical contracts unmet has shown the constraint is one this
    generator cannot satisfy), and deliberately **non-terminal**: on
    either bound the empty
    return lets the dispatcher take SCAFFOLD -> APPLY so VERIFY -- which
    exercises the contract against a real suite -- makes the final call
    rather than failing the session on a static gate. Returns an empty
    list (normal edge) for a compliant scaffold, one that dropped files,
    a repeated failure, or a spent budget.
    """
    from cgx.session.repair.propose import propose_regenerate  # dep direction

    outputs = completed.outputs or {}
    if int(outputs.get("failed_count") or 0) > 0:
        return []
    actionable = _actionable_contract_warnings(outputs)
    if not actionable:
        return []
    budget = LoopBudget.from_inputs(completed.inputs)
    signature = _scaffold_failure_signature(
        [{"file": str(w.get("module") or ""),
          "error": f"unmet contract {w.get('kind')} {w.get('name')}"}
         for w in actionable])
    if budget.regenerate_exhausted or (signature and budget.seen(signature)):
        return []
    constraint = _contract_regenerate_constraint(actionable)
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in _collect_descendants(completed.task_id, tasks):
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    actions.append(CreateTask(propose_regenerate(
        completed, constraint,
        prior_failure_signatures=_appended_signature(budget, signature))))
    return actions


def _scaffold_resume_actions(
        failed: TaskNode, tasks: List[TaskNode],
        resume_scaffold_artifact_id: Optional[str]) -> List[RouterAction]:
    """Re-queue a SCAFFOLD that crashed mid-run to resume from a checkpoint.

    A SCAFFOLD executor checkpoints its SCAFFOLD_PATCHES artifact after
    every layer, so a crash or timeout leaves the completed files under
    an incomplete checkpoint. When the runner resolves that checkpoint
    and threads its id here, and the shared regenerate budget is not
    spent, abandon any live descendants and re-queue a fresh SCAFFOLD
    carrying ``resume_scaffold_artifact_id`` -- the new attempt seeds
    every checkpointed file and regenerates only the remainder, so the
    completed work is not discarded. The incremented ``regenerate_attempt``
    doubles as the crash-loop guard: a second crash exhausts the budget
    and falls through to terminal ``FAILED``. Returns an empty list (the
    caller then ends the session ``FAILED``) for a non-SCAFFOLD failure,
    an absent checkpoint, or a spent budget.
    """
    from cgx.session.repair.propose import propose_regenerate  # dep direction

    if failed.kind is not TaskKind.SCAFFOLD:
        return []
    resume_id = str(resume_scaffold_artifact_id or "").strip()
    if not resume_id:
        return []
    if LoopBudget.from_inputs(failed.inputs).regenerate_exhausted:
        return []
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in _collect_descendants(failed.task_id, tasks):
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    actions.append(CreateTask(propose_regenerate(
        failed, {}, resume_scaffold_artifact_id=resume_id)))
    return actions


def _invalid_scaffold_constraint(
        failed_count: int,
        *, apply_failed: object,
        scaffold_failed: object) -> Dict[str, object]:
    """Build the ``invalid_scaffold_syntax`` regenerate constraint.

    Enumerates each dropped file with its concrete error so the next
    SCAFFOLD gets actionable feedback rather than a bare count. Draws
    from two sources, both shaped ``{"file", "error"}``: the SCAFFOLD's
    own ``failed`` generations (e.g. an empty patch for a missing
    entrypoint) and APPLY's ``failed_files`` (files whose source did not
    parse and were skipped before write). De-duplicated by path and
    capped so the constraint stays prompt-sized.
    """
    seen: set = set()
    details: List[str] = []
    for entry in _dropped_file_entries(scaffold_failed, apply_failed):
        path = str(entry.get("file") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        err = str(entry.get("error") or "").strip()
        details.append(f"{path} ({err})" if err else path)
        if len(details) >= 12:
            break
    files_blurb = "; ".join(details) if details else f"{failed_count} file(s)"
    rationale = (
        "The previous attempt was abandoned because these generated files "
        f"were invalid and dropped before write: {files_blurb}. Regenerate "
        "each dropped file so it parses as valid Python: keep decorated "
        "defs indented inside their class/function body, use consistent "
        "indentation and complete statements, avoid stray or trailing "
        "commas, define every referenced module and symbol, and import only "
        "modules that exist in this project or its declared dependencies.")
    return {
        "kind": "invalid_scaffold_syntax",
        "rationale": rationale,
        "failed_files": details,
    }


def _failed_scaffold_paths(scaffold_failed: object,
                           apply_failed: object) -> List[str]:
    """Return the de-duplicated file paths dropped by SCAFFOLD/APPLY.

    Draws from the same two ``{"file", "error"}`` sources as
    :func:`_invalid_scaffold_constraint` -- the SCAFFOLD's own ``failed``
    generations and APPLY's ``failed_files`` -- but returns just the
    paths so the router can hand SCAFFOLD a targeted regenerate set
    (regenerate only these; reuse every prior-good diff).
    """
    out: List[str] = []
    seen: set = set()
    for entry in _dropped_file_entries(scaffold_failed, apply_failed):
        path = str(entry.get("file") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _dropped_file_entries(scaffold_failed: object,
                          apply_failed: object) -> List[Dict[str, object]]:
    """Return the ``{"file", "error"}`` dicts from both dropped sources.

    The SCAFFOLD's own ``failed`` generations and APPLY's
    ``failed_files`` are read straight off persisted task outputs, so
    neither is guaranteed to be a list of dicts on a resumed or
    hand-edited session. Both are validated once here rather than in
    each of the three consumers (constraint, targeted paths, flap
    signature), which keeps them keyed off exactly the same entries.
    """
    out: List[Dict[str, object]] = []
    for source in (scaffold_failed, apply_failed):
        if not isinstance(source, (list, tuple)):
            continue
        out.extend(e for e in source if isinstance(e, dict))
    return out


# Environment-manifest files whose absence structurally breaks recovery.
# BOOTSTRAP_ENV keys project-type detection off exactly these (see
# ``cgx.session.tasks.bootstrap_env._detect_project_type``), so a dropped
# requirements.txt / package.json misdetects the stack and the Python
# venv (or node_modules) is never provisioned -- every downstream gate
# then fails against an unprovisioned environment, a state the per-file
# regenerate loop cannot escape because re-asking the same weak model for
# the file it just dropped reproduces the same drop.
_FOUNDATIONAL_FILES: frozenset = frozenset({
    "pyproject.toml", "setup.py", "setup.cfg", "package.json",
})


def _is_foundational_path(path: str) -> bool:
    """True when ``path`` is an environment manifest (see _FOUNDATIONAL_FILES).

    Matches the ``requirements*.txt`` family by shape and the remaining
    manifests by exact basename, both case-folded and directory-stripped.
    """
    base = path.strip().lower().rsplit("/", 1)[-1]
    if base.startswith("requirements") and base.endswith(".txt"):
        return True
    return base in _FOUNDATIONAL_FILES


def _dropped_foundational_files(scaffold_failed: object,
                                apply_failed: object) -> List[str]:
    """Return the dropped file paths that are environment manifests."""
    return [p for p in _failed_scaffold_paths(scaffold_failed, apply_failed)
            if _is_foundational_path(p)]


def _foundational_failure_note(foundational: List[str]) -> str:
    """Re-plan note for a dropped environment manifest.

    Distinct from :func:`_invalid_scaffold_constraint` (which asks the
    generator to re-emit the same file): naming the manifests and why they
    matter steers the DECOMPOSE toward a plan that emits each as a minimal
    valid file, since regenerating them in place has already failed.
    """
    files = ", ".join(sorted(set(foundational)))
    return ("Foundational environment manifest(s) were dropped as invalid: "
            f"{files}. These files drive project-type detection and "
            "environment provisioning, so regenerating them in place cannot "
            "recover the build -- restructure the plan so each is emitted as "
            "a minimal, valid manifest.")


def _verify_lesson_actions(completed: TaskNode,
                           tasks: List[TaskNode]) -> List[RouterAction]:
    """Emit :class:`RecordLesson` when a VERIFY-pass repairs a prior failure.

    Phase 7.1: a VERIFY whose outputs say ``outcome=passed`` and whose
    ancestor chain includes a REPAIR is, by construction, a successful
    repair cycle -- the REPAIR's diff (or its regenerate's fresh
    SCAFFOLD output) is what brought the test suite back to green. We
    surface that pair to the runner via a single :class:`RecordLesson`
    action carrying the VERIFY id, the REPAIR id (most recent ancestor),
    and the SCAFFOLD id if one exists on the chain (used as the
    lesson's ``scope`` provenance).
    """
    outputs = completed.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    if outcome != "passed":
        return []
    by_id = {t.task_id: t for t in tasks}
    repair: Optional[TaskNode] = None
    scaffold: Optional[TaskNode] = None
    cur_id = completed.parent_task_id
    visited: set = set()
    while cur_id and cur_id not in visited:
        visited.add(cur_id)
        cur = by_id.get(cur_id)
        if cur is None:
            break
        if repair is None and cur.kind is TaskKind.REPAIR:
            repair = cur
        if scaffold is None and cur.kind is TaskKind.SCAFFOLD:
            scaffold = cur
        cur_id = cur.parent_task_id
    if repair is None:
        return []
    return [RecordLesson(
        verify_task_id=completed.task_id,
        repair_task_id=repair.task_id,
        scaffold_task_id=scaffold.task_id if scaffold else None,
    )]


def _find_ancestor_by_kind(start: TaskNode, tasks: List[TaskNode],
                           kind: TaskKind) -> Optional[TaskNode]:
    """Walk up ``parent_task_id`` chain to the nearest task of ``kind``."""
    by_id = {t.task_id: t for t in tasks}
    visited: set = set()
    cur_id = start.parent_task_id
    while cur_id and cur_id not in visited:
        visited.add(cur_id)
        cur = by_id.get(cur_id)
        if cur is None:
            return None
        if cur.kind is kind:
            return cur
        cur_id = cur.parent_task_id
    return None


def _find_scaffold_ancestor(start: TaskNode,
                            tasks: List[TaskNode]) -> Optional[TaskNode]:
    """Walk up ``parent_task_id`` chain to the nearest SCAFFOLD task."""
    return _find_ancestor_by_kind(start, tasks, TaskKind.SCAFFOLD)


def _collect_descendants(root_task_id: str,
                         tasks: List[TaskNode]) -> List[TaskNode]:
    """Return every task whose ancestor chain includes ``root_task_id``.

    Bread-first walk over ``parent_task_id`` edges; the root itself is
    not included in the result -- only its successors are abandoned.
    """
    children_by_parent: Dict[str, List[TaskNode]] = {}
    for t in tasks:
        if t.parent_task_id:
            children_by_parent.setdefault(t.parent_task_id, []).append(t)
    out: List[TaskNode] = []
    queue: List[str] = [root_task_id]
    while queue:
        pid = queue.pop(0)
        for child in children_by_parent.get(pid, []):
            out.append(child)
            queue.append(child.task_id)
    return out
