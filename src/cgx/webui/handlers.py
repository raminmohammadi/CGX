"""Streaming handlers used by the SSE routes.

Each handler is a blocking generator that yields ``(event_name, payload)``
tuples. The SSE bridge runs them in a worker thread; the React frontend
consumes the named events directly so it can render incremental updates
without polling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

from cgx.answer.engine import _get_system_prompt, answer_with_llm, generate_code_plan
from cgx.answer.intent import detect_intent
from cgx.pipeline.auto import run_index_auto, run_query_auto
from cgx.webui.helpers import (
    build_provider,
    diffs_payload,
    json_safe,
    maybe_extract_zip,
    provider_from_profile_name,
    report_summary,
    stringify,
)


Event = Tuple[str, Dict[str, Any]]


def _resolve_provider(
    *, use_profile: bool, profile_name: Optional[str],
    kind: str, model: str, base_url: str, api_key: Optional[str],
    temperature: float, num_predict: int,
    endpoint_path: str = "/v1/chat/completions",
    allow_no_auth: bool = False,
) -> Any:
    if use_profile and profile_name:
        return provider_from_profile_name(profile_name)
    return build_provider(
        kind=kind, model=model, base_url=base_url, api_key=api_key or None,
        temperature=temperature, num_predict=num_predict,
        endpoint_path=endpoint_path, allow_no_auth=allow_no_auth,
    )


def stream_index(
    *, project_root: Optional[str], out_dir: str, embed_model: str,
    metric: str, index_type: str, zip_path: Optional[str],
    cancel_event=None,
) -> Iterator[Event]:
    """Index build — yields ``progress`` then a terminal ``result`` event."""
    logger.info("stream_index: starting project_root=%r out_dir=%r model=%s",
                project_root, out_dir, embed_model)
    try:
        if zip_path:
            logger.info("stream_index: extracting zip %r", zip_path)
            extracted = maybe_extract_zip(zip_path)
            if extracted:
                project_root = extracted
                logger.info("stream_index: extracted to %r", project_root)
        if not project_root or not os.path.exists(project_root):
            logger.error("stream_index: project_root not found: %r", project_root)
            yield "error", {"message": f"project_root not found: {project_root!r}"}
            return
        if cancel_event and cancel_event.is_set():
            logger.info("stream_index: cancelled before build")
            yield "cancelled", {"message": "Index build cancelled"}
            return
        os.makedirs(out_dir, exist_ok=True)
        logger.info("stream_index: starting index build")
        yield "progress", {"stage": "parse", "message": f"Parsing {project_root}…"}
        yield "progress", {"stage": "embed", "message": "Building embeddings…"}
        summary = run_index_auto(
            project_root=project_root, out_dir=out_dir,
            metric=metric, index_type=index_type, model_name=embed_model,
        )
        logger.info("stream_index: completed summary=%s", summary.get("counts", {}))
        yield "result", {
            "status": "ok",
            "project_root": project_root,
            "out_dir": out_dir,
            "summary": json_safe(summary),
        }
    except Exception as e:
        logger.exception("stream_index: failed with %s", e)
        yield "error", {"message": f"{type(e).__name__}: {e}"}


def stream_ask(
    *, index_dir: str, records: str, question: str, embed_model: str,
    use_profile: bool, profile_name: Optional[str], kind: str, model: str,
    base_url: str, api_key: Optional[str], temperature: float, num_predict: int,
    endpoint_path: str = "/v1/chat/completions", allow_no_auth: bool = False,
    cancel_event=None,
) -> Iterator[Event]:
    """Stream thoughts then the grounded answer with sources + meta."""
    logger.info("stream_ask: question=%r model=%s", question[:80], model)
    try:
        prov = _resolve_provider(
            use_profile=use_profile, profile_name=profile_name, kind=kind,
            model=model, base_url=base_url, api_key=api_key,
            temperature=temperature, num_predict=num_predict,
            endpoint_path=endpoint_path, allow_no_auth=allow_no_auth,
        )
    except Exception as e:
        logger.error("stream_ask: provider init failed: %s", e)
        yield "error", {"message": f"{type(e).__name__}: {e}"}
        return

    if cancel_event and cancel_event.is_set():
        yield "cancelled", {"message": "Cancelled"}
        return

    mode = detect_intent(question or "")
    logger.info("stream_ask: intent mode=%s", mode)
    yield "intent", {"mode": mode}

    out_dir = Path(index_dir).parent
    chunks_path = str(out_dir / "chunks.jsonl")
    graph_path = str(out_dir / "graph.json")

    logger.info("stream_ask: running retrieval index_dir=%r", index_dir)
    try:
        retrieval = run_query_auto(
            index_dir=index_dir, records_path=records, query=question,
            model_name=embed_model,
            chunks_path=chunks_path if os.path.exists(chunks_path) else None,
            graph_path=graph_path if os.path.exists(graph_path) else None,
            top_k_per_view=20, neighbor_depth=1, use_lexical=True,
        )
    except Exception as e:
        logger.error("stream_ask: retrieval failed: %s", e)
        yield "error", {"message": f"retrieval: {type(e).__name__}: {e}"}
        return

    hits = retrieval.get("hits", []) or []
    logger.info("stream_ask: retrieval returned %d hits", len(hits))

    if cancel_event and cancel_event.is_set():
        yield "cancelled", {"message": "Cancelled"}
        return

    sys_prompt = _get_system_prompt(mode)
    sketch_user = (
        f"QUESTION:\n{question}\n\nINTENT_MODE: {mode}\n\n"
        "Think aloud briefly about how you will answer using the SOURCES "
        "(we will produce a final grounded answer separately)."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": sketch_user},
    ]

    logger.info("stream_ask: streaming thought tokens")
    thought_tokens = 0
    try:
        for delta in prov.chat_stream(messages, temperature=float(temperature),
                                      max_tokens=min(int(num_predict), 512)):
            if cancel_event and cancel_event.is_set():
                yield "cancelled", {"message": "Cancelled during thought"}
                return
            if delta:
                thought_tokens += 1
                yield "thought", {"delta": delta}
    except Exception as e:
        logger.warning("stream_ask: thought stream unavailable: %s", e)
        yield "thought_warning", {"message": f"stream unavailable: "
                                             f"{type(e).__name__}: {e}"}

    logger.info("stream_ask: thought complete (%d tokens), generating answer", thought_tokens)
    try:
        result = answer_with_llm(index_dir, records, question, prov, hits=hits)
    except Exception as e:
        logger.error("stream_ask: answer_with_llm failed: %s", e)
        yield "error", {"message": f"answer: {type(e).__name__}: {e}"}
        return

    answer_md = stringify(result.get("answer_md", ""))
    sources = json_safe((result.get("debug") or {}).get("sources", []))
    meta = json_safe({k: v for k, v in result.items() if k != "debug"})
    logger.info("stream_ask: answer ready len=%d sources=%d", len(answer_md), len(sources))
    yield "answer", {"answer_md": answer_md, "sources": sources, "meta": meta}


def stream_plan(
    *, index_dir: str, records: str, task: str, embed_model: str,
    use_profile: bool, profile_name: Optional[str], kind: str, model: str,
    base_url: str, api_key: Optional[str], temperature: float, num_predict: int,
    self_test: bool, run_tests: bool, project_root: Optional[str],
    endpoint_path: str = "/v1/chat/completions", allow_no_auth: bool = False,
    cancel_event=None,
) -> Iterator[Event]:
    """Stream sketch thoughts, then the generated plan + structured diffs."""
    logger.info("stream_plan: task=%r self_test=%s model=%s", task[:80], self_test, model)
    try:
        prov = _resolve_provider(
            use_profile=use_profile, profile_name=profile_name, kind=kind,
            model=model, base_url=base_url, api_key=api_key,
            temperature=temperature, num_predict=num_predict,
            endpoint_path=endpoint_path, allow_no_auth=allow_no_auth,
        )
    except Exception as e:
        logger.error("stream_plan: provider init failed: %s", e)
        yield "error", {"message": f"{type(e).__name__}: {e}"}
        return

    if cancel_event and cancel_event.is_set():
        yield "cancelled", {"message": "Cancelled"}
        return

    sketch = [
        {"role": "system", "content": "You are a principal engineer thinking out loud."},
        {"role": "user", "content": (
            f"TASK:\n{task}\n\nBriefly sketch the change strategy you will pursue "
            "before producing diffs. Focus on which files to touch and risks."
        )},
    ]
    logger.info("stream_plan: streaming sketch thoughts")
    try:
        for delta in prov.chat_stream(sketch, temperature=float(temperature),
                                      max_tokens=min(int(num_predict), 400)):
            if cancel_event and cancel_event.is_set():
                yield "cancelled", {"message": "Cancelled during sketch"}
                return
            if delta:
                yield "thought", {"delta": delta}
    except Exception as e:
        logger.warning("stream_plan: thought stream unavailable: %s", e)
        yield "thought_warning", {"message": f"stream unavailable: "
                                             f"{type(e).__name__}: {e}"}

    if cancel_event and cancel_event.is_set():
        yield "cancelled", {"message": "Cancelled before codegen"}
        return

    logger.info("stream_plan: generating code plan")
    try:
        out = generate_code_plan(
            index_dir, records, task, prov,
            model_name=embed_model,
            project_root=(project_root or None),
            self_test=bool(self_test),
            run_tests=bool(run_tests),
            max_retries=1 if self_test else 0,
        )
    except Exception as e:
        logger.error("stream_plan: generate_code_plan failed: %s", e)
        yield "error", {"message": f"{type(e).__name__}: {e}"}
        return

    diffs = diffs_payload(out.get("diffs") or [])
    logger.info("stream_plan: plan ready diffs=%d", len(diffs))
    yield "plan", {
        "plan_md": stringify(out.get("plan_md", "")),
        "diffs": diffs,
        "report": report_summary(out.get("codegen_report")),
        "meta": json_safe({k: v for k, v in out.items()
                           if k not in {"debug", "diffs", "plan_md",
                                        "codegen_report"}}),
    }


def stream_agent(
    *, index_dir: Optional[str], records: Optional[str], goal: str,
    embed_model: str, use_profile: bool, profile_name: Optional[str],
    kind: str, model: str, base_url: str, api_key: Optional[str],
    temperature: float, num_predict: int, project_root: Optional[str],
    stop_on_fail: bool,
    endpoint_path: str = "/v1/chat/completions", allow_no_auth: bool = False,
    cancel_event=None,
) -> Iterator[Event]:
    """Bridge the Planner → Tracker → Judge loop into typed UI events."""
    from cgx.agents import run_agent

    logger.info("stream_agent: goal=%r model=%s", goal[:80], model)
    try:
        prov = _resolve_provider(
            use_profile=use_profile, profile_name=profile_name, kind=kind,
            model=model, base_url=base_url, api_key=api_key,
            temperature=temperature, num_predict=num_predict,
            endpoint_path=endpoint_path, allow_no_auth=allow_no_auth,
        )
    except Exception as e:
        logger.error("stream_agent: provider init failed: %s", e)
        yield "error", {"message": f"{type(e).__name__}: {e}"}
        return

    if not (goal or "").strip():
        yield "error", {"message": "goal is empty"}
        return

    if cancel_event and cancel_event.is_set():
        yield "cancelled", {"message": "Cancelled"}
        return

    # The Planner's LLM call inside run_agent() blocks before the event
    # generator yields anything; emit a synthetic status so the UI can show
    # "Planning…" instead of an empty stream during that window.
    yield "status", {"phase": "planning", "message": "Generating task plan…"}

    logger.info("stream_agent: calling run_agent (planner will block on LLM)")
    try:
        events_iter = run_agent(
            goal, provider=prov,
            index_dir=index_dir or None,
            records_path=records or None,
            project_root=(project_root or None),
            stop_on_fail=bool(stop_on_fail),
            stream=True,
            # Allow two re-plan cycles so a failed manifest or apply step
            # can be revisited (default of 1 only covers a single retry).
            max_retries=2,
        )
    except Exception as e:
        logger.error("stream_agent: run_agent init failed: %s", e)
        yield "error", {"message": f"{type(e).__name__}: {e}"}
        return

    yield "status", {"phase": "executing", "message": "Plan ready, dispatching tasks…"}

    for ev in events_iter:
        if cancel_event and cancel_event.is_set():
            logger.info("stream_agent: cancelled mid-execution")
            yield "cancelled", {"message": "Agent cancelled by user"}
            return
        try:
            payload = json_safe(ev.payload)
        except Exception:
            payload = {"_repr": str(ev.payload)}
        logger.debug("stream_agent: event type=%s", ev.type)
        yield ev.type, payload


def get_agent_plan(
    *, index_dir: Optional[str], records: Optional[str], goal: str,
    embed_model: str, use_profile: bool, profile_name: Optional[str],
    kind: str, model: str, base_url: str, api_key: Optional[str],
    temperature: float, num_predict: int, project_root: Optional[str],
) -> Dict[str, Any]:
    """Run the Planner only and return the serialised plan (no task execution).

    Used by the *Review plan* and *Plan only* execution modes so the UI can
    display the execution DAG before committing to running the agent loop.
    """
    logger.info("get_agent_plan: goal=%r model=%s", goal[:60], model)
    try:
        prov = _resolve_provider(
            use_profile=use_profile, profile_name=profile_name, kind=kind,
            model=model, base_url=base_url, api_key=api_key,
            temperature=temperature, num_predict=num_predict,
        )
    except Exception as e:
        logger.error("get_agent_plan: provider init failed: %s", e)
        return {"error": f"{type(e).__name__}: {e}"}

    try:
        from cgx.agents.planner import Planner
        planner = Planner(provider=prov)
        plan = planner.plan(goal)
        return {"plan": plan.to_dict()}
    except Exception as e:
        logger.exception("get_agent_plan: planner failed: %s", e)
        return {"error": f"{type(e).__name__}: {e}"}

    logger.info("stream_agent: complete")
