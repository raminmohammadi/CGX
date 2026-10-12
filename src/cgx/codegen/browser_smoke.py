"""Runtime smoke test for built frontends -- does the app actually MOUNT?

Static gates cannot see a render-time crash: a form that calls the wrong
helper and dereferences ``undefined`` throws during render, React unmounts the
whole tree and the user gets a blank page -- while the build is green. When a
Chromium-family browser is installed (optional; nothing is downloaded), the
Verifier serves the production bundle from a loopback port, loads each route
headlessly, dumps the rendered DOM and the console, and treats an empty root
or an uncaught error as a repair signal. It also saves a screenshot per route
under ``<root>/.cgx/screenshots/`` so a human (or a vision model) can look at
the result.

Zero new dependencies: a stdlib ``ThreadingHTTPServer`` with an SPA fallback
serves ``dist/``; the browser is driven with ``--headless=new --dump-dom`` and
``--enable-logging=stderr``. Everything degrades to a *skipped* result (never a
failure) when no browser or no build output exists. Disable with
``CGX_BROWSER_SMOKE=0``; point at a specific binary with ``CGX_BROWSER``.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["RouteResult", "BrowserSmokeResult", "find_browser", "find_dist", "run_browser_smoke",
           "smoke_enabled", "routes_from_sources"]

_MAC_APPS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Arc.app/Contents/MacOS/Arc",
)
_WIN_APPS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)
_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome",
          "msedge", "microsoft-edge", "brave-browser")

# Chrome's stderr console lines: [pid:tid:date:INFO:CONSOLE(12)] "message", source: url (12)
_CONSOLE_RE = re.compile(r'CONSOLE\(\d+\)\]\s*"(.*?)",\s*source:')
_ERROR_HINT_RE = re.compile(r"\b(Uncaught|TypeError|ReferenceError|SyntaxError|RangeError|Error:|"
                            r"is not defined|is not a function|Cannot read propert|Minified React error|"
                            r"Objects are not valid as a React child|Rendered more hooks|"
                            r"You cannot render a <Router>|Failed to fetch dynamically imported module)", re.I)
_ROUTE_RE = re.compile(r"""<Route\b[^>]*\bpath\s*=\s*['"]([^'"]+)['"]""")


def smoke_enabled() -> bool:
    return (os.environ.get("CGX_BROWSER_SMOKE") or "1").strip().lower() not in ("0", "false", "no", "off")


def find_browser() -> Optional[str]:
    """A Chromium-family binary, or ``None`` (``CGX_BROWSER`` overrides)."""
    override = (os.environ.get("CGX_BROWSER") or "").strip()
    if override:
        return override if (os.path.isfile(override) or shutil.which(override)) else None
    for name in _NAMES:
        hit = shutil.which(name)
        if hit:
            return hit
    for cand in _MAC_APPS + _WIN_APPS:
        if os.path.isfile(cand):
            return cand
    return None


def find_dist(root: str, ui_root: str = "") -> Optional[Path]:
    """The production bundle directory (``dist/`` or ``build/``) holding an index.html."""
    base = Path(root) / ui_root if ui_root else Path(root)
    for name in ("dist", "build", "out"):
        d = base / name
        if (d / "index.html").is_file():
            return d
    return None


def routes_from_sources(contents: Dict[str, str]) -> List[str]:
    """Client-side routes declared with ``<Route path=...>`` (always includes ``/``)."""
    routes: List[str] = ["/"]
    for src in (contents or {}).values():
        for m in _ROUTE_RE.finditer(src or ""):
            p = m.group(1).strip()
            if not p or p == "*" or ":" in p or "*" in p:
                continue
            if not p.startswith("/"):
                p = "/" + p
            if p not in routes:
                routes.append(p)
    return routes[:8]


@dataclass
class RouteResult:
    route: str
    mounted: bool
    errors: List[str] = field(default_factory=list)
    text_sample: str = ""
    screenshot: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.mounted and not self.errors

    def as_dict(self) -> Dict[str, object]:
        return {"route": self.route, "mounted": self.mounted, "errors": self.errors[:5],
                "text_sample": self.text_sample[:160], "screenshot": self.screenshot}


@dataclass
class BrowserSmokeResult:
    ran: bool
    skipped_reason: str = ""
    browser: str = ""
    results: List[RouteResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.ran and all(r.ok for r in self.results)

    def failures(self) -> List[RouteResult]:
        return [r for r in self.results if not r.ok]

    def as_dict(self) -> Dict[str, object]:
        return {"ran": self.ran, "ok": self.ok, "skipped_reason": self.skipped_reason,
                "browser": os.path.basename(self.browser) if self.browser else "",
                "routes": [r.as_dict() for r in self.results]}


# Headless Chrome does not print page console messages to stderr on every
# platform (macOS headless=new prints none), so the served index.html gets a
# tiny collector that mirrors uncaught errors + console.error calls into a
# hidden <div id="__cgx_errors"> that ``--dump-dom`` includes.
_ERROR_COLLECTOR = (
    "<script>(function(){var E=[];function flush(){try{var d=document.getElementById('__cgx_errors');"
    "if(!d){d=document.createElement('div');d.id='__cgx_errors';d.hidden=true;"
    "(document.body||document.documentElement).appendChild(d);}d.textContent=JSON.stringify(E.slice(0,8));}"
    "catch(e){}}window.addEventListener('error',function(e){E.push(String((e&&e.message)||e));flush();});"
    "window.addEventListener('unhandledrejection',function(e){E.push('Unhandled rejection: '+String(e&&e.reason&&"
    "(e.reason.message||e.reason)));flush();});var ce=console.error;console.error=function(){"
    "try{E.push(Array.prototype.map.call(arguments,function(a){return a&&a.message?a.message:String(a);}).join(' ')"
    ".slice(0,300));}catch(e){}flush();return ce.apply(console,arguments);};"
    "document.addEventListener('DOMContentLoaded',flush);setInterval(flush,250);})();</script>"
)
_ERRORS_DIV_RE = re.compile(r"""<div id=["']__cgx_errors["'][^>]*>(.*?)</div>""", re.S)


class _SpaHandler(SimpleHTTPRequestHandler):
    """Static files with an index.html fallback for client-side routes, and an
    error collector injected into every HTML document it serves."""

    def log_message(self, *args) -> None:  # silence
        return

    def do_GET(self) -> None:  # type: ignore[override]
        full = self.translate_path(self.path)
        if os.path.isfile(full) and full.lower().endswith((".html", ".htm")):
            try:
                with open(full, "rb") as fh:
                    body = fh.read()
                text = body.decode("utf-8", errors="replace")
                if "<head" in text:
                    idx = text.find(">", text.find("<head")) + 1
                    text = text[:idx] + _ERROR_COLLECTOR + text[idx:]
                else:
                    text = _ERROR_COLLECTOR + text
                data = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            except OSError:
                pass
        super().do_GET()

    def translate_path(self, path: str) -> str:  # type: ignore[override]
        full = super().translate_path(path)
        if os.path.isfile(full):
            return full
        if "." in os.path.basename(path):
            return full  # a real asset miss -> 404
        return os.path.join(self.directory, "index.html")  # type: ignore[attr-defined]


def _serve(directory: Path, host: str = "127.0.0.1") -> Tuple[ThreadingHTTPServer, int]:
    with socket.socket() as s:
        s.bind((host, 0))
        port = s.getsockname()[1]
    handler = partial(_SpaHandler, directory=str(directory))
    server = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


def serve_spa(directory: Path, host: str = "127.0.0.1") -> Tuple[ThreadingHTTPServer, int]:
    """Serve a built single-page app from ``directory`` on an ephemeral port.

    The same stdlib server the runtime smoke uses (static files + index.html
    fallback for client-side routes), exposed for the Agent page's **live
    preview** of a finished build: the user clicks through the real bundle
    before giving feedback. Runs on a daemon thread; call ``shutdown()`` on
    the returned server to stop it. Files are read per request, so a rebuilt
    bundle in the same directory is picked up without a restart.
    """
    return _serve(directory, host)


def _parse_console_errors(stderr: str, dom: str = "") -> List[str]:
    """Errors from Chrome's stderr console lines AND the injected DOM collector."""
    out: List[str] = []
    for m in _CONSOLE_RE.finditer(stderr or ""):
        msg = m.group(1).strip().replace("\\n", "\n")
        if _ERROR_HINT_RE.search(msg):
            first = msg.splitlines()[0][:300]
            if first not in out:
                out.append(first)
    m = _ERRORS_DIV_RE.search(dom or "")
    if m:
        import html as _html
        import json as _json
        try:
            items = _json.loads(_html.unescape(m.group(1)).strip() or "[]")
        except Exception:
            items = []
        for item in items if isinstance(items, list) else []:
            msg = str(item).strip()[:300]
            if msg and msg not in out and not msg.startswith("Download the React DevTools"):
                out.append(msg)
    return out


def _mounted(dom: str) -> Tuple[bool, str]:
    dom = _ERRORS_DIV_RE.sub("", dom or "")
    body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", dom or "")
    m = re.search(r"<body[^>]*>(.*)</body>", body, re.S)
    inner = m.group(1) if m else body
    text = re.sub(r"<[^>]+>", " ", inner)
    text = re.sub(r"\s+", " ", text).strip()
    has_root_child = bool(re.search(r"""id=["'](?:root|app)["'][^>]*>\s*<[a-zA-Z]""", dom or ""))
    return (len(text) >= 12 or has_root_child), text[:200]


def run_browser_smoke(root: str, routes: Sequence[str], *, ui_root: str = "",
                      timeout: float = 25.0, screenshots: bool = True,
                      window: str = "1280,900") -> BrowserSmokeResult:
    """Load each route of the built app headlessly; never raises."""
    if not smoke_enabled():
        return BrowserSmokeResult(ran=False, skipped_reason="disabled (CGX_BROWSER_SMOKE=0)")
    browser = find_browser()
    if not browser:
        return BrowserSmokeResult(ran=False, skipped_reason="no Chromium-family browser found "
                                  "(install Chrome/Chromium or set CGX_BROWSER)")
    dist = find_dist(root, ui_root)
    if dist is None:
        return BrowserSmokeResult(ran=False, skipped_reason="no production bundle (dist/) to serve")
    shots_dir = Path(root) / ".cgx" / "screenshots"
    try:
        server, port = _serve(dist)
    except Exception as exc:  # pragma: no cover - port binding issues
        return BrowserSmokeResult(ran=False, skipped_reason=f"could not serve bundle: {exc}")
    results: List[RouteResult] = []
    try:
        for route in (routes or ["/"]):
            url = f"http://127.0.0.1:{port}{route if route.startswith('/') else '/' + route}"
            base = [browser, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    "--no-first-run", "--disable-extensions", "--virtual-time-budget=6000",
                    f"--window-size={window}", "--enable-logging=stderr", "--v=0"]
            try:
                proc = subprocess.run(base + ["--dump-dom", url], capture_output=True, text=True,
                                      timeout=timeout)
                dom, err = proc.stdout or "", proc.stderr or ""
            except subprocess.TimeoutExpired:
                results.append(RouteResult(route, False, ["browser timed out loading the route"]))
                continue
            except Exception as exc:  # pragma: no cover - browser launch failure
                return BrowserSmokeResult(ran=False, skipped_reason=f"browser failed to launch: {exc}")
            mounted, sample = _mounted(dom)
            errors = _parse_console_errors(err, dom)
            shot: Optional[str] = None
            if screenshots:
                try:
                    shots_dir.mkdir(parents=True, exist_ok=True)
                    slug = re.sub(r"[^a-zA-Z0-9]+", "_", route).strip("_") or "index"
                    shot_path = shots_dir / f"{slug}.png"
                    subprocess.run(base + [f"--screenshot={shot_path}", url], capture_output=True,
                                   text=True, timeout=timeout)
                    if shot_path.is_file():
                        shot = str(shot_path)
                except Exception:  # pragma: no cover - screenshots are a bonus
                    shot = None
            results.append(RouteResult(route, mounted, errors, sample, shot))
    finally:
        try:
            server.shutdown()
            server.server_close()
        except Exception:  # pragma: no cover
            pass
    return BrowserSmokeResult(ran=True, browser=browser, results=results)
