

"""Interactive CGX dashboard: state, command dispatch, and the REPL."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from cgx.cli.tui import ansi, ops, render

_PROVIDER_KINDS = {"ollama", "openai", "openai-compat", "gemini", "custom"}
_QUIT = {"/quit", "/exit", "/q"}


@dataclass
class DashboardState:
    project_root: str = "."
    provider_kind: str = "ollama"
    model: str = ""
    base_url: str = "http://localhost:11434"
    profile_name: Optional[str] = None
    context_pct: int = 100


@dataclass
class DispatchResult:
    action: str = "text"  # text|agent|ask|index|serve|quit|clear|status
    output: str = ""
    arg: str = ""


class Dashboard:
    """Terminal dashboard shell. I/O is injected for testability."""

    def __init__(self, project_root: Optional[str] = None, *,
                 state: Optional[DashboardState] = None,
                 out: Callable[[str], None] = print,
                 read_input: Callable[[str], str] = input) -> None:
        self.state = state or DashboardState(
            project_root=os.path.abspath(project_root or os.getcwd()))
        self._out = out
        self._read_input = read_input
        self.color = ansi.color_enabled()

    # ---- rendering ---------------------------------------------------
    def _status_bar(self) -> str:
        idx = ops.find_existing_index(self.state.project_root)
        return render.render_status_bar(
            cwd=self.state.project_root,
            index_state="index ready" if idx else "no index (/index)",
            model=self.state.model, context_pct=self.state.context_pct,
            width=ansi.term_width(), enabled=self.color)

    def home_screen(self) -> str:
        w = ansi.term_width()
        lead = ansi.clear_screen() if self.color else ""
        parts = [render.render_banner(enabled=self.color), "",
                 self._status_bar(), "",
                 render.render_tips(enabled=self.color), "",
                 render.render_input_box(
                     "Type your message or /command", width=w,
                     enabled=self.color)]
        return lead + "\n".join(parts)

    # ---- command dispatch (pure; no heavy side effects) --------------
    def dispatch(self, line: str) -> DispatchResult:
        text = (line or "").strip()
        if not text:
            return DispatchResult()
        if not text.startswith("/"):
            return DispatchResult(action="agent", arg=text)
        cmd, _, rest = text.partition(" ")
        rest = rest.strip()
        cmd = cmd.lower()
        if cmd in _QUIT:
            return DispatchResult(action="quit")
        if cmd == "/clear":
            return DispatchResult(action="clear")
        if cmd == "/help":
            return DispatchResult(output=render.render_help())
        if cmd == "/status":
            return DispatchResult(action="status")
        if cmd == "/serve":
            return DispatchResult(action="serve")
        if cmd == "/index":
            return DispatchResult(action="index", arg=rest)
        if cmd == "/ask":
            if not rest:
                return DispatchResult(output="usage: /ask <question>")
            return DispatchResult(action="ask", arg=rest)
        if cmd == "/model":
            if not rest:
                return DispatchResult(output="usage: /model <name>")
            self.state.model = rest
            return DispatchResult(output=f"model set to {rest}")
        if cmd == "/provider":
            return self._set_provider(rest)
        if cmd == "/project":
            return self._set_project(rest)
        return DispatchResult(output=f"unknown command: {cmd} (try /help)")

    def _set_provider(self, name: str) -> DispatchResult:
        if not name:
            return DispatchResult(output="usage: /provider <profile|kind>")
        try:
            from cgx.answer.profiles import get_profile
            prof = get_profile(name)
        except Exception:
            prof = None
        if prof is not None:
            self.state.profile_name = prof.name
            self.state.provider_kind = prof.kind
            self.state.model = prof.model or self.state.model
            if prof.base_url:
                self.state.base_url = prof.base_url
            return DispatchResult(output=f"using profile {prof.name!r} "
                                         f"({prof.kind}/{prof.model})")
        if name.lower() in _PROVIDER_KINDS:
            self.state.provider_kind = name.lower()
            self.state.profile_name = None
            return DispatchResult(output=f"provider kind set to {name.lower()}")
        return DispatchResult(output=f"no profile or kind named {name!r}")

    def _set_project(self, path: str) -> DispatchResult:
        if not path:
            return DispatchResult(output="usage: /project <path>")
        ap = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(ap):
            return DispatchResult(output=f"not a directory: {ap}")
        self.state.project_root = ap
        idx = "index ready" if ops.find_existing_index(ap) else "no index yet"
        return DispatchResult(output=f"project -> {ap} ({idx})")

    # ---- interactive loop --------------------------------------------
    def _emit(self, text: str) -> None:
        self._out(text)

    def _prompt(self) -> str:
        return ansi.paint("> ", "blue", enabled=self.color)

    def ensure_model(self) -> None:
        if self.state.model or self.state.provider_kind != "ollama":
            return
        try:
            from cgx.answer import ollama_discovery
            self.state.model = ollama_discovery.recommend_default_model()
        except Exception:
            self.state.model = "qwen2.5-coder:3b"

    def _stream(self, make_iter: Callable[[object], object]) -> None:
        """Run a handler event stream on a worker thread with a live spinner.

        ``make_iter`` takes a ``threading.Event`` (the cancel flag) and
        returns the ``(type, payload)`` generator to drive. Events are
        mapped to :class:`ops.Render` instructions and applied to a
        :class:`~cgx.cli.tui.runner.Printer`; Ctrl-C cancels and returns.
        """
        import threading

        from cgx.cli.tui.runner import Printer, run_stream

        cancel = threading.Event()
        printer = Printer(is_tty=self.color, enabled=self.color)

        def on_event(item: object) -> None:
            etype, payload = item  # type: ignore[misc]
            instr = ops.map_event(etype, payload, enabled=self.color)
            if instr.op == "status":
                printer.set_status(instr.text)
            elif instr.op == "inline":
                printer.inline(instr.text)
            elif instr.op == "line":
                printer.line(instr.text)

        try:
            run_stream(lambda: make_iter(cancel), on_event=on_event,
                       printer=printer, cancel_event=cancel)
        except Exception as exc:
            printer.line(f"error: {type(exc).__name__}: {exc}")

    def _run_action(self, res: DispatchResult) -> None:
        """Perform a side-effecting action resolved by :meth:`dispatch`."""
        if res.action == "clear":
            self._emit(self.home_screen())
            return
        if res.action == "status":
            self._emit(ops.probe_status(self.state))
            return
        if res.action == "serve":
            self._emit("Launching web UI (Ctrl-C to return)...")
            try:
                from cgx.webui.launch import launch
                launch()
            except Exception as exc:
                self._emit(f"serve failed: {type(exc).__name__}: {exc}")
            return
        if res.action == "index":
            if res.arg:
                self._set_project(res.arg)
            self._stream(lambda ce: ops.index_events(self.state, cancel_event=ce))
            return
        if res.action == "ask":
            if not ops.find_existing_index(self.state.project_root):
                self._emit("no index yet -- run /index first")
                return
            self._stream(lambda ce: ops.ask_events(
                self.state, res.arg, cancel_event=ce))
            return
        if res.action == "agent":
            self._stream(lambda ce: ops.agent_events(
                self.state, res.arg, cancel_event=ce))

    def run(self) -> None:
        """Blocking read-eval-print loop."""
        self.ensure_model()
        self._emit(self.home_screen())
        while True:
            try:
                line = self._read_input(self._prompt())
            except EOFError:
                self._emit("\nBye.")
                return
            except KeyboardInterrupt:
                # Ctrl-C at the prompt clears the current line rather than
                # exiting; only /quit or EOF (Ctrl-D) leave the dashboard.
                self._emit("")
                continue
            res = self.dispatch(line)
            if res.action == "quit":
                self._emit("Bye.")
                return
            if res.output:
                self._emit(res.output)
            if res.action not in ("text",):
                self._run_action(res)
            self._emit(self._status_bar())


def run_dashboard(project_root: Optional[str] = None) -> None:
    """Entry point used by the ``cgx`` console script (bare invocation)."""
    Dashboard(project_root=project_root).run()
