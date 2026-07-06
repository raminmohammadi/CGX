

"""SSE helpers shared by streaming routes.

The streaming handlers in :mod:`cgx.webui.handlers` are blocking generators
(they call ``provider.chat_stream`` which is synchronous). To avoid
blocking the FastAPI event loop we run each generator in a worker thread
and bridge its yielded values into an ``asyncio.Queue`` that
``sse-starlette`` consumes.

Each SSE message carries a JSON-encoded payload under a named ``event``
so the frontend can switch on the event type without parsing the body.

If a ``task_id`` is provided the bridge also persists every event to the
task store so the frontend can replay them after a tab switch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_SENTINEL = object()


def _safe_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"_repr": str(payload)})


def _terminal_failure(event_name: str, event_data: Any) -> Optional[str]:
    """Return a failure message when an event marks the run as degraded.

    Keeps the persisted task status honest: a stream that emits an
    ``error`` frame, or an agent ``summary`` whose tracker recorded any
    failed task, must not be stamped ``done``. ``skipped`` alone is not
    treated as a failure -- soft/optional steps skip on the happy path --
    but skips that accompany a failure are folded into the message so the
    run history shows the cascade. Non-agent streams never emit these
    events, so their behaviour is unchanged.
    """
    if event_name == "error":
        if isinstance(event_data, dict):
            return str(event_data.get("message") or "stream error")
        return "stream error"
    if event_name == "summary" and isinstance(event_data, dict):
        try:
            failed = int(event_data.get("failed") or 0)
            skipped = int(event_data.get("skipped") or 0)
        except (TypeError, ValueError):
            return None
        if failed > 0:
            msg = f"{failed} task(s) failed"
            if skipped > 0:
                msg += f", {skipped} skipped"
            return msg
    return None


async def bridge_generator(
    gen_factory: Callable[[], Iterator[Any]],
    *,
    to_event: Callable[[Any], Dict[str, Any]],
    task_id: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> AsyncIterator[Dict[str, str]]:
    """Run a blocking generator in a thread, yield SSE message dicts.

    Parameters
    ----------
    gen_factory
        Zero-arg callable returning the blocking iterator.
    to_event
        Maps each yielded value to ``{"event": str, "data": Any}``.
    task_id
        When set, every emitted event is appended to the task store for
        later replay (tab-switch resilience).
    cancel_event
        ``threading.Event`` that, when set, causes the bridge to drain
        remaining items without forwarding them and emit a final
        ``cancelled`` frame to the client.
    """
    from cgx.webui import task_store as _ts

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=512)
    # Signalled by the consumer when it stops reading (early break, error,
    # client disconnect). The worker checks it before each push so we
    # don't leave orphaned ``queue.put`` coroutines scheduled on a loop
    # that's already tearing down.
    consumer_done = threading.Event()

    def _put(item: Any) -> bool:
        """Push ``item`` onto the queue from the worker thread.

        Uses ``run_coroutine_threadsafe`` so the worker thread blocks while
        the queue is full instead of raising ``QueueFull`` (which is what
        ``put_nowait`` does and what previously crashed the bridge on
        chatty streams such as Ollama pull progress). Returns ``False``
        when the consumer has stopped reading or the loop is gone, so the
        worker can unwind cleanly.
        """
        if consumer_done.is_set():
            return False
        coro = queue.put(item)
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            fut.result()
            return True
        except RuntimeError:
            # Loop closed or stopped -- consumer disconnected mid-stream.
            coro.close()
            return False

    def _worker() -> None:
        try:
            for item in gen_factory():
                if cancel_event and cancel_event.is_set():
                    # Drain the generator without forwarding events.
                    _put(("cancelled", {"message": "Task cancelled by user"}))
                    break
                if not _put(item):
                    return
        except Exception as exc:
            _put({"__error__": f"{type(exc).__name__}: {exc}"})
        finally:
            _put(_SENTINEL)

    thread = threading.Thread(target=_worker, name="sse-bridge", daemon=True)
    thread.start()
    logger.debug("sse bridge: thread started task_id=%s", task_id)

    # Terminal-failure message for the persisted task row. The stream can
    # complete "successfully" (no worker exception) yet still report a
    # degraded outcome -- an ``error`` event, or an agent ``summary`` whose
    # judged tasks failed. Without capturing that, ``finish_task`` in the
    # ``finally`` block would stamp every finished stream ``done`` and hide
    # the failure from the run history.
    finish_error: Optional[str] = None
    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, dict) and "__error__" in item:
                err_payload = {"message": item["__error__"]}
                finish_error = str(item["__error__"])
                if task_id:
                    _ts.append_event(task_id, "error", err_payload)
                yield {"event": "error", "data": _safe_json(err_payload)}
                break
            ev = to_event(item)
            event_name = ev.get("event", "message")
            event_data = ev.get("data")
            failure = _terminal_failure(event_name, event_data)
            if failure and finish_error is None:
                finish_error = failure
            if task_id and event_name not in ("done",):
                try:
                    _ts.append_event(task_id, event_name, event_data)
                except Exception as e:
                    logger.warning("sse bridge: failed to persist event: %s", e)
            yield {"event": event_name, "data": _safe_json(event_data)}
        logger.debug("sse bridge: stream complete task_id=%s", task_id)
        yield {"event": "done", "data": "{}"}
    finally:
        # Tell the worker we're done so it stops scheduling ``queue.put``
        # coroutines, then drain anything already in flight so no
        # un-awaited coroutine survives loop teardown.
        consumer_done.set()
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if task_id:
            try:
                task = _ts.get_task(task_id)
                if task and task.get("status") == "running":
                    _ts.finish_task(task_id, error=finish_error)
            except Exception as e:
                logger.warning("sse bridge: task cleanup failed: %s", e)
