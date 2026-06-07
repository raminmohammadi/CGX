

"""Discovery endpoints used by the Settings page.

Wraps :mod:`cgx.answer.ollama_discovery` so the React app can populate
the model dropdown and re-detect hardware without re-implementing the
heuristics client-side.

Also provides ``/provider/ping`` so the frontend can validate any
provider configuration before saving it.
"""

from __future__ import annotations

import time
from typing import List, Optional

import asyncio
import json as _json
import logging
import threading

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cgx.answer import ollama_discovery
from cgx.answer.profiles import get_profile, load_api_key
from cgx.answer.providers import GeminiProvider
from cgx.webui.models import HardwareInfo, ModelChoicesResponse


logger = logging.getLogger(__name__)
router = APIRouter(tags=["setup"])


class PingRequest(BaseModel):
    kind: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:3b"
    api_key: Optional[str] = None
    endpoint_path: str = "/v1/chat/completions"
    allow_no_auth: bool = False


class PingResponse(BaseModel):
    ok: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@router.post("/provider/ping", response_model=PingResponse)
def ping_provider(req: PingRequest) -> PingResponse:
    """Send a minimal request to the configured provider and report latency.

    For Ollama: GET /api/tags (lightweight list models call).
    For Gemini: a short generateContent request.
    For OpenAI-compat / custom: GET the base URL root or a HEAD request.
    """
    start = time.monotonic()
    try:
        if req.kind == "ollama":
            import requests as _req
            base = (req.base_url or "http://localhost:11434").replace("/v1", "").rstrip("/")
            r = _req.get(f"{base}/api/tags", timeout=8)
            r.raise_for_status()
            # Verify the selected model is actually installed.
            data = r.json() if r.content else {}
            installed_names = {
                (m.get("name") or m.get("model") or "")
                for m in (data.get("models") or [])
                if isinstance(m, dict)
            }
            model = (req.model or "").strip()
            if model and installed_names and model not in installed_names:
                # Check without the tag suffix too (e.g. "llama3.1:8b" vs "llama3.1:8b-instruct")
                base_name = model.split(":")[0]
                if not any(n.startswith(base_name) for n in installed_names):
                    elapsed = (time.monotonic() - start) * 1000
                    return PingResponse(
                        ok=False,
                        latency_ms=round(elapsed, 1),
                        error=f"Model '{model}' is not installed. Use Pull to download it first.",
                    )

        elif req.kind == "gemini":
            import requests as _req
            api_key = req.api_key or ""
            if not api_key:
                return PingResponse(ok=False, error="Gemini requires an API key")
            model = req.model or "gemini-2.5-flash"
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            body = {
                "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                "generationConfig": {"maxOutputTokens": 1},
            }
            r = _req.post(url, json=body, timeout=15)
            r.raise_for_status()

        else:
            # openai-compat or custom: attempt a GET to the base URL to verify reachability.
            import requests as _req
            base = (req.base_url or "").rstrip("/")
            if not base:
                return PingResponse(ok=False, error="base_url is required")
            path = req.endpoint_path or "/v1/chat/completions"
            # Lightweight OPTIONS/HEAD is usually enough to confirm the host is up.
            headers = {}
            if req.api_key and not req.allow_no_auth:
                headers["Authorization"] = f"Bearer {req.api_key}"
            try:
                r = _req.options(f"{base}{path}", headers=headers, timeout=8)
            except Exception:
                r = _req.head(base, headers=headers, timeout=8)
            # Accept any response — a 405 (Method Not Allowed) still proves the server is up.
            if r.status_code >= 500:
                r.raise_for_status()

        elapsed = (time.monotonic() - start) * 1000
        return PingResponse(ok=True, latency_ms=round(elapsed, 1))

    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        # Scrub Gemini-style ?key=... query params from the error string so
        # an HTTP failure carrying the request URL never leaks the API key.
        scrubbed = GeminiProvider._scrub_secret(str(exc))
        return PingResponse(ok=False, latency_ms=round(elapsed, 1), error=scrubbed[:300])


@router.get("/setup/models", response_model=ModelChoicesResponse)
def models(base_url: str = "http://localhost:11434") -> ModelChoicesResponse:
    from cgx.answer.hardware_matrix import LOCAL_MODEL_CATALOG

    ollama_reachable = False
    try:
        installed_list = ollama_discovery.list_installed_models(base_url)
        installed = [m["name"] for m in installed_list]
        # list_installed_models returns [] both when unreachable and when no
        # models are installed — do a lightweight health check to distinguish.
        health = ollama_discovery.health_check(base_url)
        ollama_reachable = bool(health.get("ok"))
        choices = ollama_discovery.model_choices(base_url)
    except Exception:
        installed = []
        choices = [tag for tag, *_ in ollama_discovery.RECOMMENDED_LADDER]

    # Merge the full hardware-catalog so every known local model appears
    # in the presets dropdown.
    seen: set = set(choices)
    for entry in LOCAL_MODEL_CATALOG:
        name = entry["name"]
        if name not in seen:
            choices.append(name)
            seen.add(name)

    # Cluster the dropdown by family / version / size so related models
    # appear together (all gemma*, all qwen*, all llama* …) instead of
    # interleaved by global parameter count. Catalog entries supply
    # exact ``params_b`` for the size tiebreaker; installed-only tags
    # fall back to the size-hint regex inside the helper.
    params_lookup = {e["name"]: float(e["params_b"]) for e in LOCAL_MODEL_CATALOG}
    choices = ollama_discovery.sort_model_choices_by_family(choices, params_lookup)

    try:
        default = ollama_discovery.recommend_default_model(base_url=base_url)
    except Exception:
        default = choices[0] if choices else "qwen2.5-coder:3b"
    return ModelChoicesResponse(
        choices=choices,
        recommended_default=default,
        installed=installed,
        ollama_reachable=ollama_reachable,
    )


@router.get("/setup/hardware", response_model=HardwareInfo)
def hardware_probe() -> HardwareInfo:
    try:
        hw = ollama_discovery.detect_hardware()
    except Exception:
        hw = {}
    return HardwareInfo(**hw)


class CloudModelsRequest(BaseModel):
    kind: str  # "gemini" | "openai-compat" | "custom"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    profile_name: Optional[str] = None


# Static fallback used when an API call fails or no key is available; kept short
# and up to date so the dropdown is never empty for new users picking a kind.
_GEMINI_FALLBACK = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]
_OPENAI_FALLBACK = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]

# OpenAI returns every model id including embeddings/audio/image; this filter
# keeps the dropdown focused on chat-capable text models.
_OPENAI_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_OPENAI_NONCHAT_SUBSTR = (
    "embedding", "whisper", "tts", "audio", "image", "vision-preview",
    "dall-e", "moderation", "search", "transcribe",
)


def _resolve_api_key(req: CloudModelsRequest) -> str:
    """Return the API key from the request body or the saved profile."""
    if req.api_key:
        return req.api_key
    if req.profile_name:
        return load_api_key(req.profile_name) or ""
    return ""


_GEMINI_CHAT_PREFIXES = ("gemini-", "gemma-")
_GEMINI_NONCHAT_SUBSTR = (
    "embedding", "aqa", "tts", "audio", "image-gen", "vision-preview",
    "-image", "robotics", "computer-use",
)


def _gemini_list(api_key: str) -> List[str]:
    import requests as _req
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    r = _req.get(url, timeout=15)
    try:
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Gemini ListModels HTTP {r.status_code}: "
            f"{GeminiProvider._scrub_secret(str(exc))}"
        )
    data = r.json() if r.content else {}
    out: List[str] = []
    for m in (data.get("models") or []):
        if not isinstance(m, dict):
            continue
        methods = m.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        name = str(m.get("name") or "")
        if name.startswith("models/"):
            name = name[len("models/"):]
        if not name or not name.startswith(_GEMINI_CHAT_PREFIXES):
            continue
        if any(s in name for s in _GEMINI_NONCHAT_SUBSTR):
            continue
        out.append(name)
    out.sort()
    return out


def _openai_list(base_url: str, api_key: str) -> List[str]:
    import requests as _req
    base = (base_url or "https://api.openai.com").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = _req.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json() if r.content else {}
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for m in items:
        mid = str((m or {}).get("id") or "")
        if not mid:
            continue
        if not mid.startswith(_OPENAI_CHAT_PREFIXES):
            continue
        low = mid.lower()
        if any(s in low for s in _OPENAI_NONCHAT_SUBSTR):
            continue
        out.append(mid)
    out.sort()
    return out


def _pick_default(kind: str, choices: List[str]) -> str:
    """Pick a sensible default highlighting newest flash-tier model."""
    if not choices:
        return ""
    if kind == "gemini":
        for prefer in ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"):
            if prefer in choices:
                return prefer
    else:
        for prefer in ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"):
            if prefer in choices:
                return prefer
    return choices[0]


class PullRequest(BaseModel):
    model: str
    base_url: str = "http://localhost:11434"


@router.post("/ollama/pull")
async def ollama_pull(req: PullRequest) -> EventSourceResponse:
    """Stream progress of `ollama pull <model>` as SSE events.

    Each SSE event has name ``progress`` and a JSON payload matching the
    Ollama pull NDJSON format: ``{status, digest?, total?, completed?}``.
    A final ``done`` event is emitted when the stream closes.
    """
    import requests as _req

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _SENTINEL = object()

    def _worker() -> None:
        base = (req.base_url or "http://localhost:11434").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        logger.info("ollama_pull: starting model=%r base=%s", req.model, base)
        line_count = 0
        saw_success = False
        saw_error: Optional[str] = None
        try:
            with _req.post(
                f"{base}/api/pull",
                json={"model": req.model, "stream": True},
                stream=True,
                timeout=600,
            ) as r:
                # Surface HTTP errors with status code + body so the UI can
                # tell apart "tag not found" (404), "Ollama too old for this
                # model manifest" (412), auth/network, etc. Plain raise_for
                # _status loses that detail.
                if r.status_code >= 400:
                    body = ""
                    try:
                        body = r.text[:300]
                    except Exception:
                        pass
                    msg = (f"ollama /api/pull returned HTTP {r.status_code}"
                           f" for model={req.model!r}"
                           + (f": {body}" if body else ""))
                    logger.warning("ollama_pull: HTTP %s for model=%r body=%r",
                                   r.status_code, req.model, body)
                    err = _json.dumps({"status": "error",
                                       "error": msg[:400]}).encode()
                    loop.call_soon_threadsafe(queue.put_nowait, err)
                    return
                for line in r.iter_lines():
                    if not line:
                        continue
                    line_count += 1
                    # Inspect for terminal states so we can log a one-line
                    # summary on close. Ollama's NDJSON has two failure
                    # shapes: {"status":"error","error":...} and the
                    # field-only {"error":"..."} — handle both.
                    try:
                        parsed = _json.loads(line)
                        if isinstance(parsed, dict):
                            if parsed.get("status") == "success":
                                saw_success = True
                            err_field = parsed.get("error")
                            if err_field and saw_error is None:
                                saw_error = str(err_field)[:300]
                    except Exception:
                        pass
                    loop.call_soon_threadsafe(queue.put_nowait, line)
        except Exception as exc:
            logger.exception("ollama_pull: worker crashed model=%r", req.model)
            err = _json.dumps({"status": "error",
                               "error": f"{type(exc).__name__}: {exc}"[:300]
                               }).encode()
            loop.call_soon_threadsafe(queue.put_nowait, err)
        finally:
            if saw_error:
                logger.warning("ollama_pull: finished model=%r lines=%d "
                               "result=error err=%r",
                               req.model, line_count, saw_error)
            elif saw_success:
                logger.info("ollama_pull: finished model=%r lines=%d "
                            "result=success", req.model, line_count)
            else:
                logger.warning("ollama_pull: finished model=%r lines=%d "
                               "result=incomplete (no success/error event)",
                               req.model, line_count)
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    threading.Thread(target=_worker, daemon=True, name="ollama-pull").start()

    async def _gen():
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            try:
                data = _json.loads(item)
                # Normalise Ollama's bare-error shape ({"error": "..."}) into
                # the same {"status":"error","error":...} envelope the rest
                # of the stack expects, so the frontend's single error path
                # catches both variants.
                if (isinstance(data, dict) and data.get("error")
                        and data.get("status") != "error"):
                    data = {**data, "status": "error"}
                yield {"event": "progress", "data": _json.dumps(data)}
            except Exception:
                logger.debug("ollama_pull: unparseable NDJSON line dropped")
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(_gen())


@router.post("/setup/cloud_models", response_model=ModelChoicesResponse)
def cloud_models(req: CloudModelsRequest) -> ModelChoicesResponse:
    """List chat-capable models for a cloud provider.

    Looks the API key up from a saved profile when ``profile_name`` is given
    so the frontend doesn't have to round-trip the secret. Falls back to a
    short static list of current models when the call fails so the dropdown
    is never empty.
    """
    kind = (req.kind or "").lower()
    api_key = _resolve_api_key(req)

    if kind == "gemini":
        try:
            choices = _gemini_list(api_key) if api_key else []
        except Exception:
            choices = []
        if not choices:
            choices = list(_GEMINI_FALLBACK)
        return ModelChoicesResponse(
            choices=choices, recommended_default=_pick_default("gemini", choices),
        )

    if kind in ("openai-compat", "custom"):
        try:
            choices = _openai_list(req.base_url or "", api_key) if api_key else []
        except Exception:
            choices = []
        if not choices:
            choices = list(_OPENAI_FALLBACK)
        return ModelChoicesResponse(
            choices=choices, recommended_default=_pick_default("openai", choices),
        )

    return ModelChoicesResponse(choices=[], recommended_default="")
