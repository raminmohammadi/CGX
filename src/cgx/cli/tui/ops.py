

"""Engine-touching operations for the dashboard.

Kept apart from :mod:`cgx.cli.tui.app` so the state machine and command
dispatch stay import-light and unit-testable; the heavy imports (parser,
embeddings, agent loop) only load when an operation actually runs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

from cgx.cli.tui import ansi

# Default code-embedding model; mirrors ``run_index_auto``/``run_query_auto``
# so the CLI reads back an index it built with matching FAISS dimensions.
DEFAULT_EMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"

Event = Tuple[str, Any]


def default_out_dir(project_root: str) -> str:
    return os.path.join(os.path.abspath(project_root), ".cgx", "index")


def index_paths(out_dir: str) -> Tuple[str, str]:
    """Return ``(index_dir, records_path)`` for an index built at ``out_dir``."""
    return os.path.join(out_dir, "indices"), os.path.join(out_dir, "records.jsonl")


def find_existing_index(project_root: str) -> Optional[Tuple[str, str]]:
    """Locate a *completed* index under ``<project>/.cgx/index``.

    ``meta.json`` is written last by ``save_indices``, so its presence (next to
    ``records.jsonl``) is the completion marker: a build that was Ctrl-C'd mid
    way leaves only ``parse_cache.json`` and is correctly reported as absent.
    """
    index_dir, records = index_paths(default_out_dir(project_root))
    meta = os.path.join(index_dir, "meta.json")
    if os.path.isdir(index_dir) and os.path.exists(meta) and os.path.exists(records):
        return index_dir, records
    return None


def index_info(project_root: str) -> Optional[Dict[str, Any]]:
    """Return the manifest (``meta.json``) of the project's index, if built."""
    idx = find_existing_index(project_root)
    if not idx:
        return None
    try:
        with open(os.path.join(idx[0], "meta.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def probe_status(state: Any) -> str:
    """Human-readable provider + hardware status block."""
    from cgx.answer import ollama_discovery

    lines = [f"Provider : {state.provider_kind}",
             f"Model    : {state.model or '(unset)'}"]
    if state.profile_name:
        lines.append(f"Profile  : {state.profile_name}")
    if state.provider_kind == "ollama":
        try:
            health = ollama_discovery.health_check(state.base_url)
            ok = "reachable" if health.get("ok") else "unreachable"
        except Exception as exc:
            ok = f"error ({type(exc).__name__})"
        lines.append(f"Ollama   : {state.base_url} -- {ok}")
    try:
        hw = ollama_discovery.detect_hardware()
        ram = hw.get("ram_gb")
        vram = hw.get("gpu_vram_gb")
        lines.append(f"Hardware : RAM {ram} GB / VRAM {vram} GB")
    except Exception:
        pass
    info = index_info(state.project_root)
    if info:
        when = info.get("indexed_at") or "?"
        lines.append(f"Index    : ready (built {when})")
        lines.append(f"  model  : {info.get('embed_model') or '?'}")
        counts = info.get("counts") or {}
        if counts:
            lines.append(f"  counts : {counts}")
        stored_root = info.get("project_root")
        if stored_root and os.path.abspath(stored_root) != os.path.abspath(state.project_root):
            lines.append(f"  ⚠ built for a different project: {stored_root}")
    else:
        lines.append("Index    : not built (run /index)")
    return "\n".join(lines)


def provider_kwargs(state: Any) -> Dict[str, Any]:
    """Map :class:`DashboardState` onto the ``stream_*`` provider params.

    A saved profile takes precedence (``use_profile``); otherwise the raw
    kind/model/base_url from the dashboard are forwarded and the handler's
    :func:`build_provider` resolves any API key from the environment.
    """
    use_profile = bool(getattr(state, "profile_name", None))
    return {
        "use_profile": use_profile,
        "profile_name": getattr(state, "profile_name", None),
        "kind": state.provider_kind,
        "model": state.model,
        "base_url": state.base_url,
        "api_key": None,
        "temperature": 0.2,
        "num_predict": 1024,
    }


def index_events(state: Any, *, cancel_event=None) -> Iterator[Event]:
    """Stream an index build for the active project (progress → result)."""
    from cgx.webui import handlers

    out_dir = default_out_dir(state.project_root)
    yield from handlers.stream_index(
        project_root=state.project_root, out_dir=out_dir,
        embed_model=DEFAULT_EMBED_MODEL, metric="cosine", index_type="flat",
        zip_path=None, cancel_event=cancel_event,
    )


def resolve_index(state: Any, *, index_dir: Optional[str] = None,
                  records: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Resolve an ``(index_dir, records)`` pair for the active project.

    An explicit ``index_dir`` + ``records`` override wins (so a CLI user can
    point at an index built with ``cgx index --out-dir``); otherwise fall
    back to the auto-discovered ``<project>/.cgx/index`` layout.
    """
    if index_dir and records:
        return index_dir, records
    return find_existing_index(state.project_root)


def ask_events(state: Any, question: str, *, index_dir: Optional[str] = None,
               records: Optional[str] = None, think: bool = False,
               cancel_event=None) -> Iterator[Event]:
    """Stream a fast, read-only grounded answer (thought + answer tokens)."""
    from cgx.webui import handlers

    idx = resolve_index(state, index_dir=index_dir, records=records)
    if not idx:
        yield "error", {"message": "no index -- run /index first"}
        return
    resolved_index, resolved_records = idx
    yield from handlers.stream_ask(
        index_dir=resolved_index, records=resolved_records, question=question,
        embed_model=DEFAULT_EMBED_MODEL, think=think, cancel_event=cancel_event,
        **provider_kwargs(state),
    )


def plan_events(state: Any, task: str, *, index_dir: Optional[str] = None,
                records: Optional[str] = None, self_test: bool = False,
                run_tests: bool = False, cancel_event=None) -> Iterator[Event]:
    """Stream a code-change plan (sketch → plan_md + structured diffs)."""
    from cgx.webui import handlers

    idx = resolve_index(state, index_dir=index_dir, records=records)
    if not idx:
        yield "error", {"message": "no index -- run /index first"}
        return
    resolved_index, resolved_records = idx
    yield from handlers.stream_plan(
        index_dir=resolved_index, records=resolved_records, task=task,
        embed_model=DEFAULT_EMBED_MODEL, self_test=self_test,
        run_tests=run_tests, project_root=state.project_root,
        cancel_event=cancel_event, **provider_kwargs(state),
    )


# ----------------------------------------------------------------------
# Session agent loop -- the TUI front-end for :mod:`cgx.session`.
# ----------------------------------------------------------------------

# One SessionRunner (and its SQLite store) per project root, mirroring
# the webui's per-root runner cache so both surfaces share sessions.
_SESSION_RUNNERS: Dict[str, Any] = {}


def _session_runner(project_root: str) -> Any:
    import cgx.session.tasks  # noqa: F401 -- registers executors
    from cgx.session import SessionRunner, SessionStore

    key = os.path.abspath(project_root or ".")
    runner = _SESSION_RUNNERS.get(key)
    if runner is None:
        runner = SessionRunner(SessionStore(project_root=key))
        _SESSION_RUNNERS[key] = runner
    return runner


def _session_deps(state: Any, *, index_dir: Optional[str],
                  records: Optional[str], store: Any) -> Any:
    """Build ExecutorDeps from dashboard state (provider + index)."""
    from cgx.session.llm_trace import TracingProvider
    from cgx.session.tasks.base import ExecutorDeps
    from cgx.webui.handlers import _resolve_provider

    provider = _resolve_provider(**provider_kwargs(state))
    if provider is not None and not isinstance(provider, TracingProvider):
        provider = TracingProvider(provider)
    idx = resolve_index(state, index_dir=index_dir, records=records)
    return ExecutorDeps(
        project_root=state.project_root,
        index_dir=idx[0] if idx else None,
        records_path=idx[1] if idx else None,
        embed_model=DEFAULT_EMBED_MODEL,
        provider=provider,
        store=store,
    )


def agent_events(state: Any, text: str, *, index_dir: Optional[str] = None,
                 records: Optional[str] = None, auto: bool = False,
                 cancel_event=None) -> Iterator[Event]:
    """Drive one turn of the session agent loop and stream its events.

    The plain-message surface for :mod:`cgx.session`: the first message
    starts a session (mode auto-detected -- greenfield for an empty
    directory, explore otherwise), a message while an ASK_USER is open
    answers it, and any other message is posted as a follow-up
    objective. Each turn drains READY tasks until the session pauses on
    a question, quiesces, or reaches a terminal status. With ``auto``
    (the one-shot ``cgx agent`` command) clarify/approval questions are
    answered with defaults so the run is unattended.
    """
    from cgx.session.mode import detect_mode
    from cgx.session.models import SessionStatus

    runner = _session_runner(state.project_root)
    store = runner.store
    try:
        deps = _session_deps(state, index_dir=index_dir, records=records,
                             store=store)
    except ValueError as exc:
        yield "error", {"message": str(exc)}
        return

    sid = getattr(state, "agent_session_id", None)
    session = store.get_session(sid) if sid else None
    if session is not None and session.status in (SessionStatus.COMPLETED,
                                                  SessionStatus.FAILED,
                                                  SessionStatus.ABANDONED):
        session = None

    if session is None:
        state.pending_ask = None
        mode = detect_mode(project_root=state.project_root,
                           index_dir=deps.index_dir,
                           records_path=deps.records_path)
        session = runner.start_session(
            objective=text, project_root=state.project_root, mode=mode)
        state.agent_session_id = session.session_id
        yield "status", {"message": f"session started ({mode.value})"}
    else:
        pending = getattr(state, "pending_ask", None)
        if pending:
            ok = yield from _answer_pending_ask(
                state, runner, session, pending, text)
            if not ok:
                return
        else:
            runner.post_message(session_id=session.session_id, message=text)

    yield from _drive_session(state, runner, session.session_id, deps,
                              auto=auto, cancel_event=cancel_event)


def _answer_pending_ask(state: Any, runner: Any, session: Any,
                        pending: Dict[str, Any], text: str):
    """Resolve the open ASK_USER from a freeform reply. Returns True on
    success; on a validation error re-renders the question and returns
    False so the user can answer again (the pending ask is kept)."""
    from cgx.session.tasks.ask import build_decision

    task = runner.store.get_task(str(pending.get("task_id") or ""))
    if task is None:  # stale pending state -- treat as a follow-up
        state.pending_ask = None
        runner.post_message(session_id=session.session_id, message=text)
        return True
    chosen, rationale = _chosen_from_text(pending, text)
    try:
        decision = build_decision(session_id=session.session_id, task=task,
                                  chosen=chosen, rationale=rationale)
    except ValueError as exc:
        yield "error", {"message": str(exc)}
        yield "ask_user", pending
        return False
    runner.post_decision(session_id=session.session_id, decision=decision)
    state.pending_ask = None
    return True


def _drive_session(state: Any, runner: Any, sid: str, deps: Any, *,
                   auto: bool, cancel_event=None) -> Iterator[Event]:
    """Drain READY tasks, relaying live bus events as TUI events.

    ``run_next`` blocks for the duration of a task, so it runs on an
    inner thread while this generator forwards the session's bus
    events (task lifecycle, scaffold file progress) as they happen.
    The 64-step ceiling mirrors the webui drain's safety valve.
    """
    import queue as queue_mod
    import threading

    from cgx.session.events import get_default_bus
    from cgx.session.tasks.ask import build_decision

    store = runner.store
    bus_q: "queue_mod.Queue[Any]" = queue_mod.Queue()
    unsubscribe = get_default_bus().subscribe(
        "*", lambda ev: bus_q.put(ev) if ev.session_id == sid else None)
    try:
        for _ in range(64):
            if cancel_event is not None and cancel_event.is_set():
                yield "cancelled", {}
                return
            holder: Dict[str, Any] = {}

            def _step() -> None:
                try:
                    holder["task"] = runner.run_next(session_id=sid,
                                                     deps=deps)
                except Exception as exc:  # surfaced below on this thread
                    holder["error"] = exc

            worker = threading.Thread(target=_step, name="cgx-session-step",
                                      daemon=True)
            worker.start()
            while worker.is_alive():
                try:
                    ev = bus_q.get(timeout=0.2)
                except queue_mod.Empty:
                    continue
                yield from _bus_to_tui(ev)
            worker.join()
            while True:
                try:
                    ev = bus_q.get_nowait()
                except queue_mod.Empty:
                    break
                yield from _bus_to_tui(ev)
            if "error" in holder:
                exc = holder["error"]
                yield "error", {"message": f"{type(exc).__name__}: {exc}"}
                return

            ask = _pending_ask_task(store, sid)
            if ask is not None:
                payload = _ask_payload(store, ask)
                if auto:
                    chosen = _auto_chosen(payload)
                    if chosen is not None:
                        yield "status", {"message": "auto-answering: "
                                         + str(payload.get("question") or "")}
                        decision = build_decision(
                            session_id=sid, task=ask, chosen=chosen)
                        runner.post_decision(session_id=sid,
                                             decision=decision)
                        continue
                state.pending_ask = payload
                yield "ask_user", payload
                return
            if holder.get("task") is None:  # quiesced, no open question
                break
        yield from _session_epilogue(store, sid)
    finally:
        unsubscribe()


def _bus_to_tui(ev: Any) -> Iterator[Event]:
    """Translate a session bus event into the dashboard's vocabulary."""
    from cgx.session.events import EventType

    p = ev.payload or {}
    kind = str(p.get("kind") or "")
    if kind == "ask_user":  # rendered separately as an ``ask_user`` event
        return
    if ev.type is EventType.TASK_STATUS_CHANGED:
        if str(p.get("status") or "") == "in_progress":
            yield "task_start", {"name": p.get("name"), "kind": kind}
    elif ev.type is EventType.TASK_COMPLETED:
        yield "task_done", {"kind": kind}
    elif ev.type is EventType.TASK_FAILED:
        yield "task_failed", {"kind": kind, "error": p.get("error")}
    elif ev.type is EventType.TASK_OUTPUT_PARTIAL:
        prog = p.get("progress") or {}
        path, idx, total = (prog.get("path"), prog.get("index"),
                            prog.get("total"))
        msg = "working…"
        if path:
            msg = f"generating {path}"
            if idx and total:
                msg += f" ({idx}/{total})"
        yield "status", {"message": msg}


def _pending_ask_task(store: Any, sid: str) -> Optional[Any]:
    from cgx.session.models import TaskKind, TaskNodeStatus

    for t in store.list_tasks(sid):
        if (t.kind is TaskKind.ASK_USER
                and t.status is TaskNodeStatus.IN_PROGRESS):
            return t
    return None


def _ask_payload(store: Any, task: Any) -> Dict[str, Any]:
    """Serialise an open ASK_USER (question + options) for rendering."""
    from cgx.session.models import DecisionKind

    expected = str(task.inputs.get("expected_kind")
                   or DecisionKind.FREEFORM.value)
    payload: Dict[str, Any] = {
        "task_id": task.task_id,
        "expected_kind": expected,
        "question": task.description or task.name,
        "questions": [],
        "options": [],
    }

    def _artifact_content(key: str) -> Dict[str, Any]:
        art = store.get_artifact(str(task.inputs.get(key) or ""))
        content = getattr(art, "content", None)
        return content if isinstance(content, dict) else {}

    if expected == DecisionKind.CLARIFY_ANSWERS.value:
        payload["questions"] = list(
            _artifact_content("requirements_artifact_id")
            .get("questions") or [])
    elif expected == DecisionKind.APPROVE_PLAN.value:
        payload["plan_md"] = str(
            _artifact_content("work_plan_artifact_id").get("plan_md") or "")
    elif expected == DecisionKind.CHOOSE_PATH.value:
        payload["options"] = list(
            _artifact_content("directions_artifact_id").get("options") or [])
    elif expected == DecisionKind.CHOOSE_RECOMMENDATION.value:
        content = _artifact_content("recommendations_artifact_id")
        payload["options"] = list(content.get("options")
                                  or content.get("recommendations") or [])
    payload["text"] = _render_ask_text(payload)
    return payload


def _render_ask_text(payload: Dict[str, Any]) -> str:
    """Plain-text block for an open question (rendered by map_event)."""
    from cgx.session.models import DecisionKind

    lines = [str(payload.get("question") or "").strip()]
    for i, q in enumerate(payload.get("questions") or [], 1):
        line = f"  {i}. " + str(q.get("prompt") or "").strip()
        hint = str(q.get("hint") or "").strip()
        if hint:
            line += f"  ({hint})"
        lines.append(line)
        sug = [str(s) for s in (q.get("suggested") or []) if str(s).strip()]
        if sug:
            lines.append("     e.g. " + " | ".join(sug))
    for i, o in enumerate(payload.get("options") or [], 1):
        title = str(o.get("title") or o.get("kind") or "").strip()
        why = str(o.get("rationale") or "").strip()
        lines.append(f"  {i}. {title}" + (f" -- {why}" if why else ""))
    plan_md = str(payload.get("plan_md") or "").strip()
    if plan_md:
        preview = plan_md.splitlines()
        if len(preview) > 30:
            preview = preview[:30] + ["…"]
        lines.append("")
        lines.extend(preview)
    tips = {
        DecisionKind.CLARIFY_ANSWERS.value:
            "Reply with one line per question (or one line for all).",
        DecisionKind.APPROVE_PLAN.value:
            "Reply 'yes' to approve, or describe what to change.",
        DecisionKind.APPROVE.value:
            "Reply 'yes' to approve, or 'no' to reject.",
        DecisionKind.CHOOSE_PATH.value: "Reply with an option number.",
        DecisionKind.CHOOSE_RECOMMENDATION.value:
            "Reply with an option number.",
    }
    lines.append("")
    lines.append(tips.get(str(payload.get("expected_kind")),
                          "Reply in the chat to answer."))
    return "\n".join(lines)


_YES_WORDS = {"y", "yes", "ok", "okay", "approve", "approved", "go",
              "lgtm", "sure", "proceed"}
_NO_WORDS = {"n", "no", "reject", "rejected", "stop", "cancel"}


def _chosen_from_text(pending: Dict[str, Any],
                      text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Map a freeform reply onto the ``chosen`` slots the decision needs.

    Returns ``(chosen, rationale)``. Anything that fails
    :func:`build_decision` validation re-renders the question, so this
    only has to be a best-effort mapping, not a strict parser.
    """
    from cgx.session.models import DecisionKind

    kind = str(pending.get("expected_kind") or "")
    text = (text or "").strip()
    if kind == DecisionKind.CLARIFY_ANSWERS.value:
        qids = [str(q.get("id") or f"q{i + 1}") for i, q in
                enumerate(pending.get("questions") or [])] or ["q1"]
        lines = [re.sub(r"^\d+[.)]\s*", "", ln.strip())
                 for ln in text.splitlines() if ln.strip()]
        if len(lines) == len(qids):
            return {"answers": dict(zip(qids, lines))}, None
        return {"answers": {qid: text for qid in qids}}, None
    if kind in (DecisionKind.APPROVE.value, DecisionKind.APPROVE_PLAN.value):
        low = text.lower().rstrip(".!")
        if low in _YES_WORDS:
            return {"approved": True}, None
        if low in _NO_WORDS:
            return {"approved": False}, None
        # Anything else is revision feedback -> not approved, with the
        # reply as rationale so a re-plan can honour it.
        return {"approved": False}, text
    if kind == DecisionKind.CHOOSE_PATH.value:
        opt = _pick_option(pending.get("options"), text)
        anchor = str((opt or {}).get("chunk_id") or "") or text
        return {"anchor_chunk_id": anchor}, None
    if kind == DecisionKind.CHOOSE_RECOMMENDATION.value:
        opt = _pick_option(pending.get("options"), text) or {}
        chosen = {"kind": str(opt.get("kind") or text)}
        anchor = str(opt.get("anchor_chunk_id") or "").strip()
        if anchor:
            chosen["anchor_chunk_id"] = anchor
        return chosen, None
    return {"text": text}, None


def _pick_option(options: Any, text: str) -> Optional[Dict[str, Any]]:
    """Resolve ``text`` as a 1-based index into ``options``."""
    if not isinstance(options, list):
        return None
    m = re.match(r"^(\d+)\b", (text or "").strip())
    if not m:
        return None
    i = int(m.group(1)) - 1
    if 0 <= i < len(options) and isinstance(options[i], dict):
        return options[i]
    return None


def _auto_chosen(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Default answer for unattended runs (``cgx agent``), or None.

    Clarify questions take their first suggested option; plan/change
    approvals are approved. Freeform and choose-* questions have no
    safe default, so the turn ends with the question printed.
    """
    from cgx.session.models import DecisionKind

    kind = str(payload.get("expected_kind") or "")
    if kind == DecisionKind.CLARIFY_ANSWERS.value:
        answers = {}
        for i, q in enumerate(payload.get("questions") or []):
            sug = [str(s) for s in (q.get("suggested") or [])
                   if str(s).strip()]
            answers[str(q.get("id") or f"q{i + 1}")] = (
                sug[0] if sug else "Use your best judgment.")
        return {"answers": answers or {"q1": "Use your best judgment."}}
    if kind in (DecisionKind.APPROVE.value, DecisionKind.APPROVE_PLAN.value):
        return {"approved": True}
    return None


def _session_epilogue(store: Any, sid: str) -> Iterator[Event]:
    """Terminal render after the session quiesced with no open question."""
    from cgx.session.models import SessionStatus, TaskNodeStatus

    session = store.get_session(sid)
    if session is None:
        return
    tasks = store.list_tasks(sid)
    done = sum(1 for t in tasks if t.status is TaskNodeStatus.DONE)
    failed = sum(1 for t in tasks if t.status is TaskNodeStatus.FAILED)
    if session.status is SessionStatus.FAILED:
        errors = [t.error for t in tasks if t.error]
        yield "error", {"message": errors[-1] if errors
                        else "session failed"}
    yield "session_done", {"status": session.status.value,
                           "done": done, "failed": failed}


@dataclass
class Render:
    """A pure render instruction produced by :func:`map_event`.

    ``op`` is one of ``status`` (update the spinner label), ``inline``
    (stream raw tokens), ``line`` (write a full scrollback line), or
    ``nothing`` (ignore the event).
    """

    op: str = "nothing"
    text: str = ""


def _summary_answer(payload: Dict[str, Any]) -> str:
    """Extract the agent's final prose from a ``summary`` payload's plan."""
    plan = payload.get("plan") or {}
    for task in reversed(plan.get("tasks") or []):
        if task.get("kind") in ("summarize", "ask"):
            ans = str((task.get("output") or {}).get("answer_md") or "").strip()
            if ans:
                return ans
    return ""


def map_event(etype: str, payload: Optional[Dict[str, Any]], *,
              enabled: bool = True) -> Render:
    """Turn a handler ``(type, payload)`` event into a :class:`Render`.

    Pure and total: unknown event types collapse to ``Render("nothing")``
    so new backend events never crash the dashboard.
    """
    payload = payload or {}
    c = lambda s, col: ansi.paint(s, col, enabled=enabled)

    # --- terminal / shared across all three streams -------------------
    if etype == "error":
        return Render("line", c("✖ error: ", "red")
                      + str(payload.get("message") or "unknown error"))
    if etype == "cancelled":
        msg = payload.get("message")
        return Render("line", c("● cancelled", "yellow")
                      + (f": {msg}" if msg else ""))

    # --- ask ----------------------------------------------------------
    if etype == "intent":
        return Render("status", f"Analyzing ({payload.get('mode', '?')})…")
    if etype in ("thought", "thought_warning", "judge", "retry_skipped"):
        return Render("nothing")
    if etype == "answer_delta":
        return Render("inline", str(payload.get("delta") or ""))
    if etype == "answer":
        srcs = payload.get("sources") or []
        tail = ansi.dim(f"({len(srcs)} source(s))", enabled=enabled) if srcs else ""
        return Render("line", "\n" + tail)

    # --- index --------------------------------------------------------
    if etype == "progress":
        return Render("status", str(payload.get("message")
                                    or payload.get("stage") or "working…"))
    if etype == "result":
        summary = payload.get("summary") or {}
        counts = summary.get("counts") or {}
        model = payload.get("embed_model") or summary.get("embed_model")
        when = payload.get("indexed_at") or summary.get("indexed_at")
        bits = [b for b in (str(model) if model else "", str(when) if when else "") if b]
        tail = ansi.dim("  " + " · ".join(bits), enabled=enabled) if bits else ""
        return Render("line", c("✔ ", "green") + f"index ready: {counts}" + tail)

    # --- plan (code-change plan handler) ------------------------------
    # The plan *handler* emits a terminal ``plan`` event carrying ``plan_md``
    # + structured ``diffs``; the *agent* loop emits ``plan`` with a nested
    # ``plan.tasks`` list. Disambiguate on the presence of ``plan_md``.
    if etype == "plan" and "plan_md" in payload:
        plan_md = str(payload.get("plan_md") or "").strip()
        diffs = payload.get("diffs") or []
        files = ", ".join(str(d.get("file", "")) for d in diffs if isinstance(d, dict))
        tail = c(f"\n\n{len(diffs)} diff(s)", "cyan") + (f": {files}" if files else "")
        return Render("line", "\n" + (plan_md or "(no plan text)") + tail)

    # --- agent --------------------------------------------------------
    if etype == "status":
        return Render("status", str(payload.get("message")
                                    or payload.get("phase") or "working…"))
    if etype in ("plan", "retry_plan"):
        tasks = (payload.get("plan") or {}).get("tasks") or []
        kinds = ", ".join(str(t.get("kind", "")) for t in tasks)
        label = "Plan" if etype == "plan" else "Re-plan"
        return Render("line", c(label, "cyan")
                      + f": {len(tasks)} task(s) -> {kinds}")
    if etype == "task_start":
        name = (payload.get("name") or payload.get("description")
                or payload.get("kind") or "task")
        return Render("line", c("  ▶ ", "yellow") + str(name))
    if etype == "task_progress":
        el = payload.get("elapsed")
        secs = f" {float(el):.0f}s" if isinstance(el, (int, float)) else ""
        return Render("status", f"working…{secs}")
    if etype == "task_done":
        return Render("line", c("  ✔ ", "green") + str(payload.get("kind") or "done"))
    if etype == "task_failed":
        return Render("line", c("  ✖ ", "red")
                      + str(payload.get("error") or payload.get("kind") or "failed"))
    if etype == "task_skipped":
        reason = payload.get("reason")
        return Render("line", ansi.dim("  ⊝ skipped" + (f": {reason}" if reason else ""),
                                       enabled=enabled))
    if etype == "retry_start":
        return Render("line", c("↻ ", "yellow")
                      + f"retry attempt {payload.get('attempt', '?')}: "
                      + str(payload.get("reason") or ""))
    if etype == "summary":
        text = _summary_answer(payload)
        if text:
            return Render("line", "\n" + text.strip() + "\n")
        counts = (f"{payload.get('completed', 0)} done, "
                  f"{payload.get('failed', 0)} failed, "
                  f"{payload.get('skipped', 0)} skipped")
        return Render("line", ansi.dim(counts, enabled=enabled))

    # --- session agent loop --------------------------------------------
    if etype == "ask_user":
        return Render("line", "\n" + str(payload.get("text")
                                         or payload.get("question") or ""))
    if etype == "session_done":
        status = str(payload.get("status") or "")
        done, failed = payload.get("done"), payload.get("failed")
        counts = (ansi.dim(f"  ({done} task(s) done, {failed} failed)",
                           enabled=enabled) if done is not None else "")
        if status == "completed":
            return Render("line", c("✔ session complete", "green") + counts)
        if status == "failed":
            return Render("line", c("✖ session failed", "red") + counts)
        return Render("line", ansi.dim(f"session {status or 'paused'}",
                                       enabled=enabled))

    return Render("nothing")
