

"""Threaded event streaming with a transient spinner and cooperative cancel.

The dashboard's heavy operations (ask / agent / index) are blocking
generators of ``(event, payload)`` tuples that share the web UI's
handlers. Running them on the REPL thread would freeze the terminal and
swallow Ctrl-C, so :func:`run_stream` drives them on a daemon worker
thread and drains events through a queue. While the worker is busy the
main thread animates a spinner (so the UI never looks frozen), and a
``KeyboardInterrupt`` flips the shared ``cancel_event`` -- which the
handlers poll between tokens -- then returns control to the prompt
instead of tearing down the process.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any, Callable, Iterator, Optional, Tuple

from cgx.cli.tui import ansi

Event = Tuple[str, Any]


class Printer:
    """Manages a single transient spinner line plus normal scrollback output.

    ``tick`` redraws the spinner in place; ``line`` erases any spinner and
    writes a full line; ``inline`` streams raw tokens (used for answer
    deltas) after erasing the spinner. All colour is opt-in via ``enabled``
    and every escape is suppressed when ``is_tty`` is false so piped output
    stays clean and tests stay deterministic.
    """

    def __init__(self, *, write: Optional[Callable[[str], None]] = None,
                 is_tty: bool = True, enabled: bool = True) -> None:
        self._write = write or (lambda s: sys.stdout.write(s))
        self.is_tty = is_tty
        self.enabled = enabled
        self._label = ""
        self._transient = False
        self._frame = 0

    def set_status(self, label: str) -> None:
        self._label = label or ""

    def _erase(self) -> None:
        if self._transient and self.is_tty:
            self._write(ansi.clear_line())
        self._transient = False

    def tick(self, elapsed: float) -> None:
        if not self.is_tty or not self._label:
            return
        frame = ansi.SPINNER_FRAMES[self._frame % len(ansi.SPINNER_FRAMES)]
        self._frame += 1
        spin = ansi.paint(frame, "cyan", enabled=self.enabled)
        secs = ansi.dim(f"({elapsed:.0f}s, Ctrl-C to cancel)",
                        enabled=self.enabled)
        self._write(f"{ansi.clear_line()}{spin} {self._label} {secs}")
        self._flush()
        self._transient = True

    def line(self, text: str = "") -> None:
        self._erase()
        self._write(text + "\n")
        self._flush()

    def inline(self, text: str) -> None:
        self._erase()
        self._label = ""  # no spinner while raw tokens stream
        self._write(text)
        self._flush()

    def _flush(self) -> None:
        try:
            sys.stdout.flush()
        except Exception:
            pass


def run_stream(
    make_events: Callable[[], Iterator[Event]],
    *,
    on_event: Callable[[Event], None],
    printer: Printer,
    cancel_event: Optional[threading.Event] = None,
    tick_interval: float = 0.1,
) -> str:
    """Drive ``make_events()`` on a worker thread; render on the caller.

    Returns ``"ok"`` on natural completion, ``"cancelled"`` if the user
    pressed Ctrl-C. Exceptions raised by the generator are re-raised on the
    main thread after the spinner is cleared so callers can report them.
    """
    cancel_event = cancel_event or threading.Event()
    q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

    def worker() -> None:
        try:
            for item in make_events():
                q.put(("event", item))
                if cancel_event.is_set():
                    break
        except BaseException as exc:  # surfaced on the main thread
            q.put(("error", exc))
        finally:
            q.put(("end", None))

    thread = threading.Thread(target=worker, name="cgx-tui-stream", daemon=True)
    start = time.monotonic()
    thread.start()
    status = "ok"
    try:
        while True:
            try:
                kind, val = q.get(timeout=tick_interval)
            except queue.Empty:
                printer.tick(time.monotonic() - start)
                continue
            if kind == "end":
                break
            if kind == "error":
                printer._erase()
                raise val
            on_event(val)
    except KeyboardInterrupt:
        cancel_event.set()
        printer._erase()
        printer.line(ansi.paint("^C  cancelling current task…", "yellow",
                                enabled=printer.enabled))
        status = "cancelled"
    printer._erase()
    return status
