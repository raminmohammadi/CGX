

"""FastAPI application factory for the CGX web UI.

Routes are split per-feature under :mod:`cgx.webui.routes`; this module
just composes them, mounts the prebuilt React bundle from
``cgx/webui/static`` (if present), and wires CORS for the Vite dev
server.

The single-page-app fallback is important: React Router uses
client-side URLs (``/ask``, ``/plan`` …) that the server must serve as
``index.html`` while still letting ``/api/*`` and the asset URLs win.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cgx.webui.routes import (
    agent,
    ask,
    embed,
    hardware,
    index as index_route,
    plan,
    profiles,
    rollback,
    sessions,
    setup,
    status,
    tasks,
)


HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
ASSETS_DIR = STATIC_DIR / "assets"


def create_app() -> FastAPI:
    app = FastAPI(
        title="CGX",
        description="Local-first codebase RAG — REST + SSE backend.",
        version="0.2.0",
        docs_url="/api/docs",
        redoc_url=None,
    )

    # CORS — permissive during Vite dev (localhost:5173). Production builds
    # are same-origin so CORS is a no-op there.
    dev_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=dev_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    # --- REST + SSE routes ---
    app.include_router(status.router, prefix="/api")
    app.include_router(setup.router, prefix="/api")
    app.include_router(profiles.router, prefix="/api")
    app.include_router(sessions.router, prefix="/api")
    app.include_router(hardware.router, prefix="/api")
    app.include_router(index_route.router, prefix="/api")
    app.include_router(embed.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")
    app.include_router(plan.router, prefix="/api")
    app.include_router(agent.router, prefix="/api")
    app.include_router(tasks.router, prefix="/api")
    app.include_router(rollback.router, prefix="/api")

    # --- Static SPA (built React app) ---
    _mount_spa(app)

    return app


def _mount_spa(app: FastAPI) -> None:
    """Mount the built React SPA, with a catch-all that serves index.html.

    During development the user runs Vite on :5173 and the React app
    points at the FastAPI server on :8765 through fetch + EventSource;
    we don't need to serve the SPA in that mode. If the static dir
    doesn't exist (frontend not built), we surface a helpful message.
    """
    has_static = STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists()
    has_assets = ASSETS_DIR.exists() and ASSETS_DIR.is_dir()

    if has_assets:
        app.mount(
            "/assets",
            StaticFiles(directory=str(ASSETS_DIR), html=False),
            name="spa-assets",
        )

    # Serve any extra top-level static files (favicon, og:image, ...).
    # Browsers implicitly hit /favicon.ico even when index.html points at an
    # SVG icon, so fall back to favicon.svg with the right media type to
    # avoid noisy 404s in the access log.
    @app.get("/favicon.ico", include_in_schema=False, response_model=None)
    def _favicon():
        ico = STATIC_DIR / "favicon.ico"
        if ico.exists():
            return FileResponse(str(ico))
        svg = STATIC_DIR / "favicon.svg"
        if svg.exists():
            return FileResponse(str(svg), media_type="image/svg+xml")
        return JSONResponse({"detail": "no favicon"}, status_code=404)

    @app.get("/", include_in_schema=False, response_model=None)
    def _root():
        if has_static:
            return FileResponse(str(STATIC_DIR / "index.html"))
        return JSONResponse(
            {
                "detail": "CGX frontend bundle not found.",
                "fix": "Run `cd frontend && npm install && npm run build` "
                       "to produce src/cgx/webui/static/.",
            },
            status_code=503,
        )

    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    def _spa_fallback(full_path: str):
        # API and asset paths are matched by their own routes above.
        if full_path.startswith("api/") or full_path.startswith("assets/"):
            return JSONResponse({"detail": "not found"}, status_code=404)
        # Pass through a real file under static/ if it exists (icons, etc.).
        candidate = STATIC_DIR / full_path
        if has_static and candidate.is_file():
            return FileResponse(str(candidate))
        if has_static:
            return FileResponse(str(STATIC_DIR / "index.html"))
        return JSONResponse(
            {"detail": "frontend not built", "path": full_path},
            status_code=503,
        )


app = create_app()
