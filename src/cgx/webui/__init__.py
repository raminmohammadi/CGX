"""Averix web UI — FastAPI backend + prebuilt React SPA.

The Python side here wraps the existing :mod:`cgx` engines (indexing,
retrieval, answering, code planning, agent loop) behind a small REST +
Server-Sent-Events surface. The frontend lives under ``frontend/`` in
the repo root and is built into :mod:`cgx.webui.static`; FastAPI serves
the bundled assets in production.
"""

from cgx.webui.server import create_app  # noqa: F401
