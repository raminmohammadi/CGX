

"""``cgx-ui`` entrypoint -- boot uvicorn + open the browser.

Run with ``python app.py`` or ``cgx-ui`` from a console script. The
``--no-browser`` flag is useful in containers and the development
``--reload`` flag turns on uvicorn auto-reload for local hacking on
the FastAPI side. The React app uses its own Vite dev server on 5173
during frontend development.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import webbrowser
from typing import Any

import uvicorn


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _open_browser_later(url: str, delay: float = 0.8) -> None:
    def _go() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def _maybe_enable_hf_offline() -> None:
    """Flip HuggingFace into offline mode when the embedding model is cached.

    Thin wrapper over :func:`cgx.embeddings.build.maybe_enable_hf_offline`
    (the shared guard used by every entry point) so the web UI enables
    offline mode as early as possible -- before any HuggingFace import.
    """
    try:
        from cgx.embeddings.build import maybe_enable_hf_offline
        maybe_enable_hf_offline()
    except Exception:
        pass


def launch(**kwargs: Any) -> None:
    """Programmatic entry point used by ``app.py`` and the console script."""
    parser = argparse.ArgumentParser(description="CGX web UI")
    parser.add_argument("--host", default=os.environ.get("CGX_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("CGX_PORT", DEFAULT_PORT)))
    parser.add_argument("--reload", action="store_true",
                        help="Enable uvicorn auto-reload (dev only).")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser on startup.")
    args, _ = parser.parse_known_args()

    # Allow programmatic overrides (e.g. tests).
    host = kwargs.get("host", args.host)
    port = int(kwargs.get("port", args.port))

    from cgx.logging_setup import setup_logging
    setup_logging(level="INFO")

    import logging
    log = logging.getLogger("cgx.launch")
    log.info("CGX starting on http://%s:%d/", host, port)

    _maybe_enable_hf_offline()
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        log.info("HF_HUB_OFFLINE=1 (embedding model cached locally)")

    try:
        from cgx import telemetry
        telemetry.ping()
    except Exception:
        pass

    if not args.no_browser and not kwargs.get("no_browser"):
        _open_browser_later(f"http://{host}:{port}/")

    uvicorn.run(
        "cgx.webui.server:app",
        host=host,
        port=port,
        reload=bool(args.reload or kwargs.get("reload")),
        log_level=kwargs.get("log_level", "info"),
    )


if __name__ == "__main__":
    launch()
