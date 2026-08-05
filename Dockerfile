# CGX container image -- multi-stage.
#
#   Stage 1 (frontend): build the React SPA into src/cgx/webui/static so the
#     FastAPI server can serve it same-origin (vite outDir points there).
#   Stage 2 (runtime): a slim Python image that installs the package and the
#     prebuilt bundle, runs uvicorn as a non-root user, and self-reports
#     health via /healthz.
#
# Extras are selectable at build time:
#   docker build -t cgx:latest .                       # core + web UI
#   docker build --build-arg CGX_EXTRAS=all -t cgx:ml .  # + ML (torch/faiss)

# ---------------------------------------------------------------------------
# Stage 1: build the SPA bundle.
# ---------------------------------------------------------------------------
FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend

# Install deps first for layer caching.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build: `tsc -b && vite build` emits to ../src/cgx/webui/static (see
# frontend/vite.config.ts), i.e. /build/src/cgx/webui/static here.
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Python runtime.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# core+ui by default; pass --build-arg CGX_EXTRAS=all for the ML image.
ARG CGX_EXTRAS=ui

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CGX_HOST=0.0.0.0 \
    CGX_PORT=8765 \
    CGX_CONFIG_DIR=/data

WORKDIR /app

# Package metadata + sources (package-dir maps cgx->src/cgx, skills->skills).
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY skills/ ./skills/

# Prebuilt SPA from stage 1 (the server serves src/cgx/webui/static).
COPY --from=frontend /build/src/cgx/webui/static ./src/cgx/webui/static

RUN pip install --upgrade pip && pip install ".[${CGX_EXTRAS}]"

# Run as an unprivileged user; persist config/observation DBs under /data.
RUN useradd --create-home --uid 10001 cgx \
    && mkdir -p /data \
    && chown -R cgx:cgx /data /app
USER cgx
VOLUME ["/data"]

EXPOSE 8765

# Liveness self-probe against the in-container server.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3).status==200 else 1)"

# Host/port come from CGX_HOST/CGX_PORT; --no-browser is mandatory in a container.
CMD ["cgx-ui", "--no-browser"]
