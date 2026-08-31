

# src/cgx/cli/main.py
from __future__ import annotations

"""
CLI for Codebase RAG (auto-wired).

This command-line tool wires the unused-but-important helpers into the actual run path:
- Uses `embeddings.build.build_embeddings` and `embeddings.index.build_faiss_index` during indexing.
- Uses `retrieval.orchestrator.hybrid_retrieve_two_view` for queries.
- Calls `retrieval.orchestrator.suggest_insertion_points` to propose anchors.
- Optionally exercises `embeddings.search.semantic_search` via --single-view.
- Touches config objects via .from_overrides()/.to_dict() to validate overrides surface.

This file is **add-only** with respect to behavior: the interface remains
compatible with prior flags seen in the project.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from cgx.pipeline.auto import run_index_auto, run_query_auto
from cgx.config import EmbeddingConfig, FaissConfig, HybridSearchConfig
from cgx.embeddings.loader import load_embedder_from_spec

# Provider kinds accepted by ``--provider``; mirrors the dashboard so the
# non-interactive CLI and the TUI resolve providers identically.
_PROVIDER_KINDS = ("ollama", "openai", "openai-compat", "gemini", "huggingface", "custom")


def _resolve_embedder_or_model(args: argparse.Namespace) -> tuple[Any, str]:
    """Return (embedder_obj_or_None, model_name).

    Users may provide either ``--embedder "module:attr"`` (advanced BYO) or
    ``--model NAME`` (sentence-transformers / HF id). They may also pass
    neither, in which case the default model in ``run_index_auto`` is used.
    """
    if getattr(args, "embedder", None):
        return load_embedder_from_spec(args.embedder), getattr(args, "model", None) or ""
    return None, getattr(args, "model", None) or "jinaai/jina-embeddings-v2-base-code"


def _cmd_index(args: argparse.Namespace) -> None:
    _ = EmbeddingConfig.from_overrides()
    _ = FaissConfig.from_overrides(metric=args.metric, index_type=args.index_type)
    _ = HybridSearchConfig.from_overrides()
    _ = _.to_dict() if hasattr(_, "to_dict") else None

    embedder, model_name = _resolve_embedder_or_model(args)
    summary = run_index_auto(
        project_root=args.project_root,
        out_dir=args.out_dir,
        metric=args.metric,
        index_type=args.index_type,
        model_name=model_name,
        embedder=embedder,
    )
    print(json.dumps(summary, indent=2))


def _cmd_query(args: argparse.Namespace) -> None:
    hy = HybridSearchConfig.from_overrides(rrf_k=60.0)
    _ = hy.to_dict()

    # Auto-discover sibling artifacts when not explicitly provided, mirroring
    # the behaviour of the web UI (handlers.py) and generate_code_plan.
    index_parent = Path(args.index_dir).parent
    graph_path = args.graph or None
    if graph_path is None:
        auto_graph = index_parent / "graph.json"
        if auto_graph.exists():
            graph_path = str(auto_graph)
    chunks_path = args.chunks or None
    if chunks_path is None:
        auto_chunks = index_parent / "chunks.jsonl"
        if auto_chunks.exists():
            chunks_path = str(auto_chunks)

    embedder, model_name = _resolve_embedder_or_model(args)
    res = run_query_auto(
        index_dir=args.index_dir,
        records_path=args.records,
        query=args.query,
        model_name=model_name,
        chunks_path=chunks_path,
        graph_path=graph_path,
        top_k_per_view=args.top_k,
        neighbor_depth=args.depth,
        use_lexical=(not args.no_lexical),
        single_view=args.single_view,
        embedder=embedder,
    )
    print(json.dumps(res, indent=2, default=str))


def _cmd_serve(args: argparse.Namespace) -> None:
    """Launch the FastAPI + React web UI."""
    try:
        from cgx.webui.launch import launch as _launch
    except Exception as e:
        raise SystemExit(f"Failed to import UI: {type(e).__name__}: {e}")
    _launch(host=args.host, port=args.port, no_browser=args.no_browser)


def _cmd_dash(args: argparse.Namespace) -> None:
    """Launch the interactive terminal dashboard."""
    from cgx.cli.tui import run_dashboard
    run_dashboard(project_root=getattr(args, "project_root", None))


# --- grounded ask / plan / agent (shared streaming path) ----------------

def _ensure_model(state: Any) -> None:
    """Fill in a sensible default Ollama model when none was supplied."""
    if state.model or state.provider_kind != "ollama":
        return
    try:
        from cgx.answer import ollama_discovery
        state.model = ollama_discovery.recommend_default_model()
    except Exception:
        state.model = "qwen2.5-coder:3b"


def _state_from_args(args: argparse.Namespace) -> Any:
    """Build a :class:`DashboardState` from shared provider flags.

    A ``--profile`` takes precedence (kind/model/base_url come from the saved
    profile); otherwise the explicit ``--provider``/``--model``/``--base-url``
    values are used. ``--model`` always overrides a profile's model when given.
    """
    from cgx.cli.tui.app import DashboardState

    root = os.path.abspath(getattr(args, "project_root", None) or os.getcwd())
    state = DashboardState(project_root=root)
    profile = getattr(args, "profile", None)
    if profile:
        from cgx.answer.profiles import get_profile
        prof = get_profile(profile)
        if prof is None:
            raise SystemExit(f"unknown provider profile: {profile!r}")
        state.profile_name = prof.name
        state.provider_kind = prof.kind
        state.model = prof.model or ""
        if prof.base_url:
            state.base_url = prof.base_url
    else:
        state.provider_kind = getattr(args, "provider", None) or "ollama"
        state.base_url = getattr(args, "base_url", None) or state.base_url
    if getattr(args, "model", None):
        state.model = args.model
    _ensure_model(state)
    return state


def _run_cli_stream(make_iter: Callable[[Any], Any]) -> None:
    """Drive a handler event stream to the terminal (spinner + tokens).

    Mirrors the dashboard's :meth:`Dashboard._stream` but for a one-shot
    command: maps ``(type, payload)`` events through :func:`ops.map_event`
    onto a :class:`~cgx.cli.tui.runner.Printer`, honouring Ctrl-C cancel.
    """
    import threading

    from cgx.cli.tui import ansi, ops
    from cgx.cli.tui.runner import Printer, run_stream

    enabled = ansi.color_enabled()
    printer = Printer(is_tty=sys.stdout.isatty(), enabled=enabled)
    cancel = threading.Event()

    def on_event(item: Any) -> None:
        etype, payload = item
        instr = ops.map_event(etype, payload, enabled=enabled)
        if instr.op == "status":
            printer.set_status(instr.text)
        elif instr.op == "inline":
            printer.inline(instr.text)
        elif instr.op == "line":
            printer.line(instr.text)

    try:
        status = run_stream(lambda: make_iter(cancel), on_event=on_event,
                            printer=printer, cancel_event=cancel)
    except Exception as exc:
        printer.line(f"error: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    if status == "cancelled":
        raise SystemExit(130)


def _cmd_ask(args: argparse.Namespace) -> None:
    """Stream a grounded, read-only answer over the project's index."""
    from cgx.cli.tui import ops

    state = _state_from_args(args)
    question = " ".join(args.question)
    _run_cli_stream(lambda ce: ops.ask_events(
        state, question, index_dir=args.index_dir, records=args.records,
        think=args.think, cancel_event=ce))


def _cmd_plan(args: argparse.Namespace) -> None:
    """Stream a code-change plan (plan_md + structured diffs)."""
    from cgx.cli.tui import ops

    state = _state_from_args(args)
    task = " ".join(args.task)
    _run_cli_stream(lambda ce: ops.plan_events(
        state, task, index_dir=args.index_dir, records=args.records,
        self_test=args.self_test, run_tests=args.run_tests, cancel_event=ce))


def _cmd_agent(args: argparse.Namespace) -> None:
    """Run one unattended turn of the session agent loop."""
    from cgx.cli.tui import ops

    if getattr(args, "target_dir", None):
        args.project_root = args.target_dir

    state = _state_from_args(args)
    goal = " ".join(args.goal)

    # Opt-in human-in-the-loop: --approve installs a gate that prompts at the
    # terminal before any risky tool call (arbitrary code execution, file
    # writes, external MCP calls). Off by default so unattended runs are
    # unchanged.
    if getattr(args, "approve", False):
        from cgx.session.approval import (
            ApprovalGate, set_default_gate, terminal_responder)
        set_default_gate(ApprovalGate(responder=terminal_responder))
    try:
        _run_cli_stream(lambda ce: ops.agent_events(
            state, goal, index_dir=args.index_dir, records=args.records,
            auto=True, mode=getattr(args, "mode", None), cancel_event=ce))
    finally:
        if getattr(args, "approve", False):
            from cgx.session.approval import set_default_gate
            set_default_gate(None)


def _cmd_status(args: argparse.Namespace) -> None:
    """Print provider + hardware + index status for the project."""
    from cgx.cli.tui import ops

    print(ops.probe_status(_state_from_args(args)))


def _add_provider_flags(p: argparse.ArgumentParser) -> None:
    """Register the provider/index flags shared by ask/plan/agent/status."""
    p.add_argument("--project-root", default=None,
                   help="Project directory (default: current dir).")
    p.add_argument("--model", default=None,
                   help="LLM model name (overrides a profile's model).")
    p.add_argument("--provider", default="ollama", choices=_PROVIDER_KINDS,
                   help="Provider kind when no --profile is given.")
    p.add_argument("--base-url", default="http://localhost:11434",
                   help="Provider base URL (ollama/openai-compat).")
    p.add_argument("--profile", default=None,
                   help="Saved provider profile (overrides --provider/--model).")
    p.add_argument("--index-dir", default=None,
                   help="Override auto-discovered index dir (<project>/.cgx/index).")
    p.add_argument("--records", default=None,
                   help="Override auto-discovered records.jsonl.")
    p.add_argument("--mode", default=None, choices=["explore", "greenfield", "swarm"],
                   help="Override auto-detected session mode.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cgx", description="Codebase RAG CLI")
    # No subcommand -> launch the interactive dashboard (the friendly
    # default surface). Explicit subcommands remain for scripted use.
    sub = parser.add_subparsers(dest="cmd", required=False)

    # index
    p_i = sub.add_parser("index", help="Parse -> records -> two-view embeddings -> FAISS -> persist")
    p_i.add_argument("--project-root", required=True)
    p_i.add_argument(
        "--model",
        default="jinaai/jina-embeddings-v2-base-code",
        help="Embedding model name (Sentence-Transformers or HF id). Used when --embedder is not given.",
    )
    p_i.add_argument(
        "--embedder",
        default=None,
        help="Optional advanced: import spec 'module:attr' that yields an object with .encode(list[str]). Overrides --model.",
    )
    p_i.add_argument("--out-dir", required=True)
    p_i.add_argument("--metric", default="cosine", choices=["cosine", "l2", "ip"])
    p_i.add_argument("--index-type", default="flat", choices=["flat", "ivf", "hnsw"])
    p_i.add_argument("--no-normalize-impl", action="store_true", help="(compat) Was used to affect impl-view text normalization.")
    p_i.add_argument("--strip-literals-impl", action="store_true", help="(compat) Was used to strip literals in impl view.")
    p_i.set_defaults(func=_cmd_index)

    # query
    p_q = sub.add_parser("query", help="Query two-view indices with hybrid fusion (semantic+lexical+graph).")
    p_q.add_argument("--index-dir", required=True, help="Path to 'indices' dir produced by `cgx index`.")
    p_q.add_argument("--records", required=True, help="Path to records.jsonl from `cgx index`.")
    p_q.add_argument(
        "--model",
        default="jinaai/jina-embeddings-v2-base-code",
        help="Embedding model name. Must match what was used at index time.",
    )
    p_q.add_argument(
        "--embedder",
        default=None,
        help="Optional advanced: import spec 'module:attr'. Overrides --model.",
    )
    p_q.add_argument("--query", required=True)
    p_q.add_argument("--chunks", help="Optional: chunks.jsonl for lexical.")
    p_q.add_argument("--graph", help="Optional: graph.json for graph expansion.")
    p_q.add_argument("--top-k", type=int, default=10)
    p_q.add_argument("--depth", type=int, default=1, help="Neighbor depth for graph expansion.")
    p_q.add_argument("--no-lexical", action="store_true", help="Disable lexical component.")
    p_q.add_argument("--single-view", choices=["intent","impl"], help="Also run direct semantic_search on a single view.")
    p_q.add_argument("--limit", type=int, default=10, help="Print top-N rows.")
    p_q.set_defaults(func=_cmd_query)

    # serve
    p_s = sub.add_parser("serve", help="Launch the CGX FastAPI + React web UI.")
    p_s.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p_s.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    p_s.add_argument("--no-browser", action="store_true",
                     help="Do not open a browser tab on startup.")
    p_s.set_defaults(func=_cmd_serve)

    # dash -- the interactive terminal dashboard (also the bare-`cgx` default)
    p_d = sub.add_parser(
        "dash", help="Launch the interactive terminal dashboard (default).")
    p_d.add_argument("--project-root", default=None,
                     help="Project directory to open (default: current dir).")
    p_d.set_defaults(func=_cmd_dash)

    # ask -- grounded, read-only Q&A over the project index
    p_ask = sub.add_parser(
        "ask", help="Ask a grounded question about the indexed project.")
    p_ask.add_argument("question", nargs="+", help="The question to ask.")
    p_ask.add_argument("--think", action="store_true",
                       help="Stream the model's reasoning before the answer.")
    _add_provider_flags(p_ask)
    p_ask.set_defaults(func=_cmd_ask)

    # plan -- generate a code-change plan (plan_md + structured diffs)
    p_plan = sub.add_parser(
        "plan", help="Generate a code-change plan for a task.")
    p_plan.add_argument("task", nargs="+", help="What to plan/change.")
    p_plan.add_argument("--self-test", action="store_true",
                        help="Have the planner critique/repair its own plan.")
    p_plan.add_argument("--run-tests", action="store_true",
                        help="Execute the project's tests as part of planning.")
    _add_provider_flags(p_plan)
    p_plan.set_defaults(func=_cmd_plan)

    # agent -- one unattended turn of the session agent loop
    p_ag = sub.add_parser(
        "agent", help="Run the session agent loop toward a goal.")
    p_ag.add_argument("goal", nargs="+", help="The goal for the agent.")
    p_ag.add_argument("--target-dir", default=None,
                      help="Explicit target directory for swarm outputs (overrides project-root).")
    p_ag.add_argument("--approve", action="store_true",
                      help="Prompt for approval at the terminal before any "
                           "risky tool call (code execution, file writes, MCP).")
    _add_provider_flags(p_ag)
    p_ag.set_defaults(func=_cmd_agent)

    # status -- provider + hardware + index summary
    p_st = sub.add_parser(
        "status", help="Show provider, hardware, and index status.")
    _add_provider_flags(p_st)
    p_st.set_defaults(func=_cmd_status)

    args = parser.parse_args(argv)
    # Bare `cgx` (no subcommand) opens the dashboard.
    if getattr(args, "func", None) is None:
        _cmd_dash(args)
        return
    # Opt-in anonymous telemetry; off unless ``CGX_TELEMETRY=1`` is set.
    try:
        from cgx import telemetry
        telemetry.ping()
    except Exception:
        pass
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
