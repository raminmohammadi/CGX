

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
import uuid
from pathlib import Path

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cgx import metrics as _metrics
from cgx.trace import (
    emit_trace as _trace_emit,
    is_trace_enabled,
    reset_trace_context,
    set_trace_context,
)

from cgx.webui.routes import (
    activity as activity_route,
    admin as admin_route,
    agent_profiles,
    agent_session,
    ask,
    embed,
    feedback as feedback_route,
    hardware,
    health as health_route,
    index as index_route,
    metrics as metrics_route,
    monitor as monitor_route,
    plan,
    profiles,
    rollback,
    sessions,
    settings as settings_route,
    setup,
    skills as skills_route,
    status,
    tasks,
    usage as usage_route,
)


HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
ASSETS_DIR = STATIC_DIR / "assets"


def _route_template(request: Request, fallback: str) -> str:
    """Low-cardinality route label for metrics.

    Prefers the matched route's path template (``/api/tasks/{id}/events``)
    over the raw URL so per-id paths don't explode metric cardinality.
    Falls back to a coarse bucket for unmatched paths.
    """
    route = request.scope.get("route")
    tmpl = getattr(route, "path", None)
    if tmpl:
        return str(tmpl)
    if fallback.startswith("/api/"):
        return "/api/_unmatched"
    return "/_spa"


def _record_red(request: Request, method: str, elapsed_ms: float,
                status_code: int) -> None:
    """Record Rate/Errors/Duration metrics for one HTTP request."""
    try:
        path = _route_template(request, request.url.path)
        _metrics.inc("cgx_http_requests_total",
                     help="HTTP requests by method/route/status.",
                     method=method, route=path, status=str(status_code))
        _metrics.observe("cgx_http_request_duration_ms", elapsed_ms,
                         help="HTTP request duration in milliseconds.",
                         method=method, route=path)
    except Exception:  # pragma: no cover - metrics must never break a request
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="CGX",
        description="Local-first codebase RAG -- REST + SSE backend.",
        version="0.2.0",
        docs_url="/api/docs",
        redoc_url=None,
    )

    # CORS -- permissive during Vite dev (localhost:5173). Production builds
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

    # Per-request observability bracket. Three concerns, cheapest first:
    #   1. request_id -- always assigned and propagated on the trace context
    #      (+ echoed as ``X-Request-ID``) so every downstream record for one
    #      request correlates end-to-end.
    #   2. RED metrics -- always recorded (rate/errors/duration per route),
    #      independent of the trace toggle, for the /api/metrics scrape.
    #   3. Curated trace enter/exit -- only when the global trace flag is on.
    @app.middleware("http")
    async def _observe_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        traced = is_trace_enabled()
        t0 = time.perf_counter()
        method = request.method
        path = request.url.path
        token = set_trace_context(request_id=request_id)
        if traced:
            _trace_emit("trace_enter", category="http",
                        fn=f"{method} {path}", request_id=request_id)
        status_code = 500
        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 200)
        except BaseException as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _record_red(request, method, elapsed_ms, status_code)
            if traced:
                _trace_emit("trace_error", category="http",
                            fn=f"{method} {path}", request_id=request_id,
                            elapsed_ms=int(elapsed_ms),
                            error_type=type(exc).__name__, error=str(exc)[:300])
            reset_trace_context(token)
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _record_red(request, method, elapsed_ms, status_code)
        response.headers["X-Request-ID"] = request_id
        if traced:
            _trace_emit("trace_exit", category="http", fn=f"{method} {path}",
                        request_id=request_id, elapsed_ms=int(elapsed_ms),
                        status_code=status_code)
        reset_trace_context(token)
        return response

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
    app.include_router(agent_session.router, prefix="/api")
    app.include_router(agent_profiles.router, prefix="/api")
    app.include_router(skills_route.router, prefix="/api")
    app.include_router(tasks.router, prefix="/api")
    app.include_router(rollback.router, prefix="/api")
    app.include_router(settings_route.router, prefix="/api")
    app.include_router(metrics_route.router, prefix="/api")
    app.include_router(monitor_route.router, prefix="/api")
    app.include_router(feedback_route.router, prefix="/api")
    app.include_router(usage_route.router, prefix="/api")
    app.include_router(activity_route.router, prefix="/api")
    app.include_router(admin_route.router, prefix="/api")

    # Liveness/readiness probes at the root (``/healthz``, ``/readyz``) -- no
    # ``/api`` prefix, and registered before the SPA catch-all so the React
    # fallback can't shadow them.
    app.include_router(health_route.router)

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
        # Normalize the request path then containment-check it against
        # STATIC_DIR via the recognized ``startswith`` prefix guard before
        # any filesystem access, so ``../`` can't escape (CodeQL path-injection).
        static_root = os.path.realpath(str(STATIC_DIR))
        candidate = os.path.realpath(os.path.join(static_root, full_path))
        if (has_static and candidate.startswith(static_root + os.sep)
                and os.path.isfile(candidate)):
            return FileResponse(candidate)
        if has_static:
            return FileResponse(str(STATIC_DIR / "index.html"))
        return JSONResponse(
            {"detail": "frontend not built", "path": full_path},
            status_code=503,
        )


app = create_app()
