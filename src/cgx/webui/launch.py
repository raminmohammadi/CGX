"""``averix-ui`` entrypoint — boot uvicorn + open the browser.

Run with ``python app.py`` or ``averix-ui`` from a console script. The
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

    Skips remote update checks (which can hang behind SSL timeouts on
    locked-down networks) at SentenceTransformer load time. Only opts in
    when the user hasn't explicitly set the flag and the default embed
    model has a snapshot in the local HF cache.
    """
    if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
        return
    try:
        from cgx.config import EmbeddingConfig
        model_name = EmbeddingConfig().model_name
    except Exception:
        return
    cache_root = (
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.expanduser("~/.cache/huggingface/hub")
    )
    if os.environ.get("HF_HOME"):
        cache_root = os.path.join(cache_root, "hub")
    safe = "models--" + model_name.replace("/", "--")
    snapshots = os.path.join(cache_root, safe, "snapshots")
    try:
        if os.path.isdir(snapshots) and any(os.scandir(snapshots)):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
    except OSError:
        pass


def launch(**kwargs: Any) -> None:
    """Programmatic entry point used by ``app.py`` and the console script."""
    parser = argparse.ArgumentParser(description="Averix web UI")
    parser.add_argument("--host", default=os.environ.get("AVERIX_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("AVERIX_PORT", DEFAULT_PORT)))
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
    log = logging.getLogger("averix.launch")
    log.info("Averix starting on http://%s:%d/", host, port)

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
