

"""Engine-touching operations for the dashboard.

Kept apart from :mod:`cgx.cli.tui.app` so the state machine and command
dispatch stay import-light and unit-testable; the heavy imports (parser,
embeddings, agent loop) only load when an operation actually runs.
"""

from __future__ import annotations

import json
import os
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


def ask_events(state: Any, question: str, *, cancel_event=None) -> Iterator[Event]:
    """Stream a fast, read-only grounded answer (thought + answer tokens)."""
    from cgx.webui import handlers

    idx = find_existing_index(state.project_root)
    if not idx:
        yield "error", {"message": "no index -- run /index first"}
        return
    index_dir, records = idx
    yield from handlers.stream_ask(
        index_dir=index_dir, records=records, question=question,
        embed_model=DEFAULT_EMBED_MODEL, cancel_event=cancel_event,
        **provider_kwargs(state),
    )


def agent_events(state: Any, goal: str, *, cancel_event=None) -> Iterator[Event]:
    """Stream the full Planner → Tracker → Judge agent loop."""
    from cgx.webui import handlers

    idx = find_existing_index(state.project_root)
    index_dir = idx[0] if idx else None
    records = idx[1] if idx else None
    yield from handlers.stream_agent(
        index_dir=index_dir, records=records, goal=goal,
        embed_model=DEFAULT_EMBED_MODEL, project_root=state.project_root,
        stop_on_fail=False, cancel_event=cancel_event,
        **provider_kwargs(state),
    )


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

    return Render("nothing")
