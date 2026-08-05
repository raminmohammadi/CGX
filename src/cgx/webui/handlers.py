

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

from cgx.answer.engine import answer_with_llm_stream, generate_code_plan
from cgx.answer.intent import detect_intent
from cgx.answer.model_caps import model_supports_thinking
from cgx.answer.scope import resolve_scope_for_intent
from cgx.governance import govern
from cgx.guardrails import (
    GuardrailConfig,
    assert_llm_enabled,
    check_diffs,
    record_findings,
    scan_context,
    scan_text,
)
from cgx.pipeline.auto import IndexBuildCancelled, run_index_auto, run_query_auto
from cgx.registry import new_run_id, register_known_prompts
from cgx.trace import set_trace_context
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


def _monitor_answer(result: Optional[Dict[str, Any]],
                    *, run_id: Optional[str] = None) -> None:
    """Feed a finished answer payload to the AIOps groundedness monitor.

    Best-effort: monitoring must never break a user request, so any failure
    (import, DB, or check error) is swallowed with a warning.
    """
    if not isinstance(result, dict):
        return
    try:
        from cgx.monitor import get_default_monitor
        get_default_monitor().observe_answer(result, run_id=run_id)
    except Exception as e:  # pragma: no cover - monitoring is non-critical
        logger.warning("monitor.observe_answer failed: %s", e)


def _monitor_codegen(report: Optional[Dict[str, Any]],
                     *, run_id: Optional[str] = None) -> None:
    """Feed a ``codegen_report`` to the AIOps repair-health monitor."""
    if not isinstance(report, dict):
        return
    try:
        from cgx.monitor import get_default_monitor
        get_default_monitor().observe_codegen(report, run_id=run_id)
    except Exception as e:  # pragma: no cover - monitoring is non-critical
        logger.warning("monitor.observe_codegen failed: %s", e)


def _guard_input(question: str, hits: Optional[List[Dict[str, Any]]],
                 *, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan the user question + retrieved chunks for prompt injection.

    Best-effort: guardrails annotate + alert, they never break a request.
    Returns the findings as dicts for the response ``meta``.
    """
    try:
        cfg = GuardrailConfig.from_env()
        if not (cfg.enabled and cfg.scan_input):
            return []
        findings = scan_text(question, source="user") + scan_context(hits or [])
        record_findings(findings, run_id=run_id, kind="input")
        return [f.to_dict() for f in findings]
    except Exception as e:  # pragma: no cover - guardrails are non-critical
        logger.warning("guardrail input scan failed: %s", e)
        return []


def _guard_output(diffs: List[Dict[str, Any]], *, project_root: Optional[str],
                  run_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """Scan generated diffs for secret literals + path escapes.

    Returns ``(findings, block)`` where ``block`` is True only when a critical
    finding fired *and* ``CGX_GUARDRAIL_BLOCK_SECRETS`` is set (hard-stop).
    """
    try:
        cfg = GuardrailConfig.from_env()
        if not (cfg.enabled and cfg.scan_output):
            return [], False
        findings = check_diffs(diffs or [], project_root=project_root)
        record_findings(findings, run_id=run_id, kind="output")
        block = cfg.block_secret_output and any(
            f.severity == "critical" for f in findings)
        return [f.to_dict() for f in findings], block
    except Exception as e:  # pragma: no cover - guardrails are non-critical
        logger.warning("guardrail output scan failed: %s", e)
        return [], False


def _format_stream_failure(e: BaseException) -> str:
    """Render an exception raised by ``provider.chat_stream`` for the
    ``thought_warning`` UI banner.

    Provider streaming paths now raise ``RuntimeError`` with a pre-scrubbed,
    human-readable message (see :class:`GeminiProvider`); for those we drop
    the redundant class prefix. Lower-level exceptions keep their class name
    so the cause is still visible.
    """
    msg = (str(e) or "").strip()
    if isinstance(e, RuntimeError) and msg:
        return msg
    return f"{type(e).__name__}: {msg or e!r}"


def _resolve_provider(
    *, use_profile: bool, profile_name: Optional[str],
    kind: str, model: str, base_url: str, api_key: Optional[str],
    temperature: float, num_predict: int,
    num_ctx: Optional[int] = None,
    endpoint_path: str = "/v1/chat/completions",
    allow_no_auth: bool = False,
) -> Any:
    # Subsystem K kill-switch: refuse to build any provider when the operator
    # has engaged CGX_LLM_DISABLED, so no LLM call can run this request.
    assert_llm_enabled()
    if use_profile and profile_name:
        prov = provider_from_profile_name(profile_name)
    else:
        prov = build_provider(
            kind=kind, model=model, base_url=base_url, api_key=api_key or None,
            temperature=temperature, num_predict=num_predict, num_ctx=num_ctx,
            endpoint_path=endpoint_path, allow_no_auth=allow_no_auth,
        )
    # Subsystem I: wrap at the single provider choke-point so every ask/plan
    # and agent-session call is quota-checked (hard-stop) and metered. A
    # disabled config returns the provider unchanged.
    return govern(prov)


def stream_index(
    *, project_root: Optional[str], out_dir: str, embed_model: str,
    metric: str, index_type: str, zip_path: Optional[str],
    cancel_event=None,
) -> Iterator[Event]:
    """Index build -- yields ``progress`` then a terminal ``result`` event."""
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
            cancel_event=cancel_event,
        )
        logger.info("stream_index: completed summary=%s", summary.get("counts", {}))
        yield "result", {
            "status": "ok",
            "project_root": project_root,
            "out_dir": out_dir,
            "embed_model": summary.get("embed_model"),
            "indexed_at": summary.get("indexed_at"),
            "summary": json_safe(summary),
        }
    except IndexBuildCancelled:
        # Cooperative cancel: the build stopped at a stage boundary before any
        # index files were written, so there is nothing to clean up.
        logger.info("stream_index: build cancelled by user")
        yield "cancelled", {"message": "Index build cancelled"}
    except Exception as e:
        logger.exception("stream_index: failed with %s", e)
        yield "error", {"message": f"{type(e).__name__}: {e}"}


def stream_ask(
    *, index_dir: str, records: str, question: str, embed_model: str,
    use_profile: bool, profile_name: Optional[str], kind: str, model: str,
    base_url: str, api_key: Optional[str], temperature: float, num_predict: int,
    num_ctx: Optional[int] = None,
    endpoint_path: str = "/v1/chat/completions", allow_no_auth: bool = False,
    think: bool = False,
    cancel_event=None,
) -> Iterator[Event]:
    """Stream thoughts then the grounded answer with sources + meta."""
    logger.info("stream_ask: question=%r model=%s", question[:80], model)
    # Provenance join key for this execution: stamped into the returned meta
    # (so the client can attach feedback to it) and onto the trace context
    # (so llm_trace stamps the same run_id onto every LLM_CALL of this run).
    run_id = new_run_id()
    set_trace_context(run_id=run_id)
    try:
        prov = _resolve_provider(
            use_profile=use_profile, profile_name=profile_name, kind=kind,
            model=model, base_url=base_url, api_key=api_key,
            temperature=temperature, num_predict=num_predict, num_ctx=num_ctx,
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
    scope = resolve_scope_for_intent(question or "", mode)
    logger.info("stream_ask: intent mode=%s scope=%s", mode, scope)
    yield "intent", {"mode": mode, "scope": scope}

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
            # 10 per view (was 20): fewer fused hits -> a smaller grounding
            # prompt -> faster prefill / time-to-first-token on local models,
            # with no measurable quality loss for typical Ask questions. Mirrors
            # the agent loop's retrieval budget.
            top_k_per_view=10, neighbor_depth=1, use_lexical=True,
            scope=scope,
        )
    except Exception as e:
        logger.error("stream_ask: retrieval failed: %s", e)
        yield "error", {"message": f"retrieval: {type(e).__name__}: {e}"}
        return

    hits = retrieval.get("hits", []) or []
    logger.info("stream_ask: retrieval returned %d hits", len(hits))

    # Subsystem K: scan user question + retrieved chunks for prompt injection
    # before they reach the model. Advisory -- annotate + alert, never block.
    guard_findings = _guard_input(question or "", hits, run_id=run_id)

    if cancel_event and cancel_event.is_set():
        yield "cancelled", {"message": "Cancelled"}
        return

    # Plain-prose system prompt for the streaming "thought" sketch. The
    # final grounded answer is produced by a separate call below which uses
    # the intent-specific JSON-output system prompt; mixing the two here
    # makes JSON-constrained models (e.g. Gemini) emit raw JSON during the
    # thinking phase, which leaks into the UI's THINKING panel.
    sketch_system = (
        "You are a senior codebase assistant sketching out how you will "
        "answer the user's question. Reply with brief PLAIN PROSE only -- "
        "no JSON, no markdown code fences, no citations, no final answer. "
        "Keep it under five sentences and focus on what you'll look for "
        "and how you'll structure the response."
    )
    sketch_user = (
        f"QUESTION:\n{question}\n\nINTENT_MODE: {mode}\n\n"
        "Briefly describe how you will approach this answer. Plain prose only."
    )
    messages = [
        {"role": "system", "content": sketch_system},
        {"role": "user", "content": sketch_user},
    ]

    # The thought sketch is a full extra generation pass, so it only runs
    # when the user opted in *and* the model is reasoning-capable. For every
    # other case (toggle off, or a non-reasoning model like gemma) we skip
    # straight to the grounded answer -- halving latency on local models.
    do_think = bool(think) and model_supports_thinking(model)
    thought_tokens = 0
    if not do_think:
        logger.info(
            "stream_ask: skipping thought phase (think=%s model=%s supported=%s)",
            think, model, model_supports_thinking(model),
        )
    else:
        logger.info("stream_ask: streaming thought tokens")
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
            yield "thought_warning", {"message": _format_stream_failure(e)}

        logger.info("stream_ask: thought complete (%d tokens), streaming answer", thought_tokens)
    answer_delta_tokens = 0
    result: Optional[Dict[str, Any]] = None
    try:
        for ev, data in answer_with_llm_stream(
            index_dir, records, question, prov,
            hits=hits, temperature=float(temperature),
            max_tokens=int(num_predict) if num_predict else None,
        ):
            if cancel_event and cancel_event.is_set():
                yield "cancelled", {"message": "Cancelled during answer"}
                return
            if ev == "answer_delta":
                answer_delta_tokens += 1
                yield "answer_delta", {"delta": str(data.get("delta") or "")}
            elif ev == "answer":
                result = data
    except Exception as e:
        logger.error("stream_ask: answer_with_llm_stream failed: %s", e)
        yield "error", {"message": f"answer: {type(e).__name__}: {e}"}
        return

    if result is None:
        yield "error", {"message": "answer: stream ended without final event"}
        return

    answer_md = stringify(result.get("answer_md", ""))
    sources = json_safe((result.get("debug") or {}).get("sources", []))
    meta = json_safe({k: v for k, v in result.items() if k != "debug"})
    meta.setdefault("run_id", run_id)
    meta.setdefault("model", model)
    if guard_findings:
        meta["guardrails"] = guard_findings
    try:
        meta.setdefault("prompt_version",
                        register_known_prompts().version_of(f"ask_stream:{mode}"))
    except Exception:  # pragma: no cover - provenance is best-effort
        pass
    logger.info(
        "stream_ask: answer ready len=%d sources=%d delta_tokens=%d",
        len(answer_md), len(sources), answer_delta_tokens,
    )
    _monitor_answer(result, run_id=run_id)
    yield "answer", {"answer_md": answer_md, "sources": sources, "meta": meta}


def stream_plan(
    *, index_dir: str, records: str, task: str, embed_model: str,
    use_profile: bool, profile_name: Optional[str], kind: str, model: str,
    base_url: str, api_key: Optional[str], temperature: float, num_predict: int,
    self_test: bool, run_tests: bool, project_root: Optional[str],
    num_ctx: Optional[int] = None,
    endpoint_path: str = "/v1/chat/completions", allow_no_auth: bool = False,
    cancel_event=None,
) -> Iterator[Event]:
    """Stream sketch thoughts, then the generated plan + structured diffs."""
    logger.info("stream_plan: task=%r self_test=%s model=%s", task[:80], self_test, model)
    run_id = new_run_id()
    set_trace_context(run_id=run_id)
    try:
        prov = _resolve_provider(
            use_profile=use_profile, profile_name=profile_name, kind=kind,
            model=model, base_url=base_url, api_key=api_key,
            temperature=temperature, num_predict=num_predict, num_ctx=num_ctx,
            endpoint_path=endpoint_path, allow_no_auth=allow_no_auth,
        )
    except Exception as e:
        logger.error("stream_plan: provider init failed: %s", e)
        yield "error", {"message": f"{type(e).__name__}: {e}"}
        return

    if cancel_event and cancel_event.is_set():
        yield "cancelled", {"message": "Cancelled"}
        return

    # Subsystem K: scan the task text for prompt injection before generation.
    guard_findings = _guard_input(task or "", None, run_id=run_id)

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
        yield "thought_warning", {"message": _format_stream_failure(e)}

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
    _monitor_codegen(out.get("codegen_report"), run_id=run_id)
    # Subsystem K: scan generated diffs for secret literals + path escapes.
    out_findings, block = _guard_output(diffs, project_root=project_root,
                                        run_id=run_id)
    if block:
        logger.error("stream_plan: guardrail hard-stop (secret in output)")
        yield "error", {"message": ("guardrail: generated code contains a "
                                    "secret-shaped literal; plan withheld")}
        return
    meta = json_safe({k: v for k, v in out.items()
                      if k not in {"debug", "diffs", "plan_md",
                                   "codegen_report"}})
    meta.setdefault("run_id", run_id)
    meta.setdefault("model", model)
    guard_all = guard_findings + out_findings
    if guard_all:
        meta["guardrails"] = guard_all
    yield "plan", {
        "plan_md": stringify(out.get("plan_md", "")),
        "diffs": diffs,
        "report": report_summary(out.get("codegen_report")),
        "meta": meta,
    }



