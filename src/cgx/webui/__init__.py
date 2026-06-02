# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""CGX web UI — FastAPI backend + prebuilt React SPA.

The Python side here wraps the existing :mod:`cgx` engines (indexing,
retrieval, answering, code planning, agent loop) behind a small REST +
Server-Sent-Events surface. The frontend lives under ``frontend/`` in
the repo root and is built into :mod:`cgx.webui.static`; FastAPI serves
the bundled assets in production.

``create_app`` is exposed lazily via :pep:`562` ``__getattr__`` so
that purely-stdlib submodules (``task_store``, ``sse``) and helpers
that only depend on :mod:`cgx.answer` remain importable without the
``[ui]`` extra (``fastapi``/``uvicorn``/``sse-starlette``) installed.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from cgx.webui.server import create_app as _create_app
        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - type-checker only
    from cgx.webui.server import create_app  # noqa: F401
