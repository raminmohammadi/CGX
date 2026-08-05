

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
import re
import threading
from urllib.parse import urlparse

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cgx.answer import ollama_discovery
from cgx.answer.profiles import get_profile, load_api_key
from cgx.answer.providers import GeminiProvider
from cgx.webui.models import HardwareInfo, ModelChoicesResponse


logger = logging.getLogger(__name__)
router = APIRouter(tags=["setup"])

# Restrict outbound model-discovery requests to explicitly approved providers.
# Keep this list aligned with supported OpenAI-compatible backends.
# Map each approved host to its canonical, compile-time-constant base URL.
# The validated host is used only as a lookup key; the value returned to the
# caller is a constant string, which is what keeps attacker-controlled text
# out of the outbound request (SSRF barrier).
_OPENAI_ALLOWED_BASES = {
    "api.openai.com": "https://api.openai.com",
}

# Hugging Face hosts CGX may reach server-side. Both are compile-time
# constants so no attacker-controlled text ever flows into the outbound
# request host (SSRF barrier): the router serves OpenAI-compatible inference
# and the Hub serves the model catalog browsed from the Settings page.
_HF_ROUTER_BASE = "https://router.huggingface.co"
_HF_HUB_BASE = "https://huggingface.co"


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
            # Validate the user-supplied base URL before fetching it
            # server-side (SSRF guard: http/https + real host only).
            base = ollama_discovery.validate_base_url(
                (req.base_url or "http://localhost:11434").replace("/v1", ""))
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
            # The model name is interpolated into the request path, so restrict
            # it to a bare model token: this stops a value like ``../`` or one
            # carrying ``/``/``?`` from redirecting the request off the fixed
            # Google host (SSRF guard).
            if not re.fullmatch(r"[A-Za-z0-9._\-]+", model):
                return PingResponse(ok=False, error="invalid model name")
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
            )
            body = {
                "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                "generationConfig": {"maxOutputTokens": 1},
            }
            r = _req.post(url, params={"key": api_key}, json=body, timeout=15)
            r.raise_for_status()

        elif req.kind == "huggingface":
            import requests as _req
            # The router host is a fixed constant, so there is no SSRF surface
            # here. When a token + model are supplied we validate both with a
            # 1-token chat completion; otherwise we fall back to the public
            # model list, which confirms reachability without a key. The model
            # id travels in the JSON body (never the URL), so it can't redirect
            # the request off the fixed host.
            model = (req.model or "").strip()
            headers = {"Content-Type": "application/json"}
            if req.api_key:
                headers["Authorization"] = f"Bearer {req.api_key}"
            if req.api_key and model:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "stream": False,
                }
                r = _req.post(f"{_HF_ROUTER_BASE}/v1/chat/completions",
                              headers=headers, json=body, timeout=20)
                r.raise_for_status()
            else:
                r = _req.get(f"{_HF_ROUTER_BASE}/v1/models",
                             headers=headers, timeout=10)
                r.raise_for_status()

        else:
            # openai-compat or custom: attempt a GET to the base URL to verify reachability.
            import requests as _req
            # Validate before issuing the server-side request (SSRF guard).
            try:
                base = ollama_discovery.validate_base_url(req.base_url or "")
            except ValueError as ve:
                return PingResponse(ok=False, error=str(ve))
            # Constrain the user-supplied endpoint path to a leading-slash
            # relative path so it can only extend ``base`` (already validated
            # above) and cannot inject a new scheme/host (SSRF guard).
            path = req.endpoint_path or "/v1/chat/completions"
            # Validate user-controlled path to prevent partial SSRF/path abuse.
            if not path.startswith("/"):
                return PingResponse(ok=False, error="endpoint_path must start with '/'")
            if any(token in path for token in ("?", "#", "://", "@", "\\")):
                return PingResponse(ok=False, error="Invalid endpoint_path")
            if ".." in path:
                return PingResponse(ok=False, error="Invalid endpoint_path")
            if not re.fullmatch(r"/[A-Za-z0-9._~/%-]*", path):
                return PingResponse(ok=False, error="Invalid endpoint_path")
            # Lightweight OPTIONS/HEAD is usually enough to confirm the host is up.
            headers = {}
            if req.api_key and not req.allow_no_auth:
                headers["Authorization"] = f"Bearer {req.api_key}"
            try:
                r = _req.options(f"{base}{path}", headers=headers, timeout=8)
            except Exception:
                r = _req.head(base, headers=headers, timeout=8)
            # Accept any response -- a 405 (Method Not Allowed) still proves the server is up.
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
        # models are installed -- do a lightweight health check to distinguish.
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
# Short list of strong chat/coder models served by HF Inference Providers,
# used when the live router listing is unavailable so the dropdown is never
# empty for a user who just picked the Hugging Face kind.
_HF_FALLBACK = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
    "meta-llama/Llama-3.3-70B-Instruct",
    "zai-org/GLM-4.5",
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
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    # The API key travels in the x-goog-api-key header, never the query
    # string, so it can't leak into request logs/proxies or taint the URL
    # (matches GeminiProvider._auth_headers).
    headers = {"x-goog-api-key": api_key} if api_key else {}
    r = _req.get(url, headers=headers, timeout=15)
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


def _validate_openai_base_url(base_url: str) -> str:
    raw = (base_url or "https://api.openai.com").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Invalid base_url scheme")
    if not parsed.hostname:
        raise ValueError("Invalid base_url host")

    # Disallow URL components that can alter request routing/semantics.
    if parsed.username or parsed.password:
        raise ValueError("Userinfo is not allowed in base_url")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError("Params/query/fragment are not allowed in base_url")

    path = (parsed.path or "").rstrip("/")
    if path not in ("", "/v1"):
        raise ValueError("Only empty path or /v1 is allowed in base_url")

    # SSRF barrier: the validated host is used only as a lookup key into a
    # constant allowlist, and the base URL handed back is a compile-time
    # constant string. No attacker-controlled text flows into the outbound
    # request, so model discovery can only ever reach an approved provider.
    canonical = _OPENAI_ALLOWED_BASES.get(parsed.hostname.lower())
    if canonical is None:
        raise ValueError("Host is not in the allowed list")
    if path == "/v1":
        canonical += "/v1"
    return canonical


def _openai_list(base_url: str, api_key: str) -> List[str]:
    import requests as _req
    base = _validate_openai_base_url(base_url)
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


def _hf_inference_list(api_key: str) -> List[str]:
    """List chat models served by HF Inference Providers via the router.

    ``GET https://router.huggingface.co/v1/models`` is public (no key needed
    to list), so this populates the dropdown before the user pastes a token;
    the key is only required for the actual inference call. Keeps text->text
    models and drops image/audio-only ones.
    """
    import requests as _req
    url = f"{_HF_ROUTER_BASE}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = _req.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json() if r.content else {}
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or "")
        if not mid:
            continue
        arch = m.get("architecture") if isinstance(m.get("architecture"), dict) else {}
        in_mods = arch.get("input_modalities") or []
        out_mods = arch.get("output_modalities") or []
        # Keep chat models: text must be an accepted input and a produced
        # output. Empty modality lists (older entries) are treated as text.
        if in_mods and "text" not in in_mods:
            continue
        if out_mods and "text" not in out_mods:
            continue
        out.append(mid)
    return sorted(set(out))


def _pick_default(kind: str, choices: List[str]) -> str:
    """Pick a sensible default highlighting newest flash-tier model."""
    if not choices:
        return ""
    if kind == "gemini":
        for prefer in ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"):
            if prefer in choices:
                return prefer
    elif kind == "huggingface":
        for prefer in ("Qwen/Qwen2.5-Coder-32B-Instruct", "openai/gpt-oss-20b",
                       "deepseek-ai/DeepSeek-V3-0324"):
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
    # Optional clean local name to give the model once pulled. When the source
    # tag is an ``hf.co/<repo>`` one, Ollama otherwise keeps the full web
    # address as the model name; supplying ``local_name`` re-aliases it to a
    # short, human-friendly tag (via /api/copy + /api/delete of the source).
    local_name: Optional[str] = None


# A valid Ollama model tag is ``name[:tag]`` -- lowercase-ish path segment plus
# an optional tag. We restrict a user-supplied ``local_name`` to that shape so
# it can never smuggle a registry host/namespace (``a/b``) or odd characters
# into the copy destination.
_OLLAMA_NAME_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")


def _sanitize_local_name(raw: str) -> Optional[str]:
    """Coerce ``raw`` into a safe ``name[:tag]`` Ollama model name or None."""
    raw = (raw or "").strip()
    if not raw:
        return None
    name, _, tag = raw.partition(":")
    if not _OLLAMA_NAME_RE.fullmatch(name):
        return None
    tag = tag or "latest"
    if not _OLLAMA_NAME_RE.fullmatch(tag):
        return None
    return f"{name}:{tag}"


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
        # Validate the user-supplied base URL before fetching it server-side:
        # ``req.base_url`` comes straight off the Settings page, so passing it
        # unchecked into ``requests`` is a server-side request forgery vector
        # (py/partial-ssrf). ``validate_base_url`` restricts it to http/https
        # with a real host and rebuilds it from validated components.
        raw = (req.base_url or "http://localhost:11434").rstrip("/")
        if raw.endswith("/v1"):
            raw = raw[:-3]
        try:
            base = ollama_discovery.validate_base_url(raw)
        except ValueError as ve:
            err = _json.dumps({"status": "error",
                               "error": f"invalid base_url: {ve}"[:300]
                               }).encode()
            loop.call_soon_threadsafe(queue.put_nowait, err)
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
            return
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
                    # field-only {"error":"..."} -- handle both.
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
            # On a clean pull, optionally re-alias the model to a short local
            # name so it no longer shows up as the full ``hf.co/<repo>`` web
            # address. Best-effort: any failure here leaves the original tag in
            # place and never turns a successful download into a UI error.
            dest = _sanitize_local_name(req.local_name or "")
            if saw_success and not saw_error and dest and dest != req.model:
                try:
                    cr = _req.post(f"{base}/api/copy",
                                   json={"source": req.model, "destination": dest},
                                   timeout=60)
                    if cr.status_code < 400:
                        _req.delete(f"{base}/api/delete",
                                    json={"model": req.model}, timeout=60)
                        logger.info("ollama_pull: re-aliased %r -> %r",
                                    req.model, dest)
                        note = _json.dumps({"status": "success",
                                            "renamed_to": dest}).encode()
                        loop.call_soon_threadsafe(queue.put_nowait, note)
                    else:
                        logger.warning("ollama_pull: copy to %r returned HTTP %s"
                                       " -- keeping original tag",
                                       dest, cr.status_code)
                except Exception:
                    logger.warning("ollama_pull: re-alias to %r failed -- "
                                   "keeping original tag", dest, exc_info=True)
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

    if kind == "huggingface":
        # The router's model list is public, so we list even without a key --
        # this lets the dropdown populate the moment the user picks the kind,
        # before they've pasted a token.
        try:
            choices = _hf_inference_list(api_key)
        except Exception:
            choices = []
        if not choices:
            choices = list(_HF_FALLBACK)
        return ModelChoicesResponse(
            choices=choices, recommended_default=_pick_default("huggingface", choices),
        )

    return ModelChoicesResponse(choices=[], recommended_default="")


# --------------------- Hugging Face Hub browse (Part B) ---------------------

# Accepted (friendly) sort keys -> the exact value the Hub /api/models endpoint
# expects. The Hub uses camelCase ``trendingScore`` and rejects anything else
# with HTTP 400, so we translate rather than forward the query value verbatim
# (this doubles as the SSRF/abuse allowlist). ``trendingScore`` is also accepted
# as an alias so either spelling from a client resolves correctly.
_HF_HUB_SORTS = {
    "trending_score": "trendingScore",
    "trendingScore": "trendingScore",
    "downloads": "downloads",
    "likes": "likes",
    "lastModified": "lastModified",
    "createdAt": "createdAt",
}
# Quantization labels embedded in GGUF filenames (e.g. ``...-Q4_K_M.gguf``).
# Used to surface per-quant pull tags (``hf.co/<repo>:Q4_K_M``) for Ollama.
_HF_QUANT_RE = re.compile(r"(?i)(Q\d[0-9A-Z_]*|IQ\d[0-9A-Z_]*|BF16|F16|F32)")


class HfHubModel(BaseModel):
    id: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: Optional[str] = None
    gated: bool = False
    # ``ollama pull hf.co/<repo>`` grabs a default quant; append ``:<quant>``
    # to pick a specific one. The frontend uses ``pull_tag`` verbatim.
    pull_tag: str = ""
    quants: List[str] = []


class HfModelsResponse(BaseModel):
    models: List[HfHubModel] = []


def _hf_hub_gguf_list(search: str, sort: str, limit: int) -> List[HfHubModel]:
    """Browse GGUF repositories on the Hub that Ollama can pull directly.

    Queries the public Hub catalog at ``huggingface.co/api/models`` filtered
    to ``gguf`` models. The host is a fixed constant and every user value goes
    through the ``params`` mapping (URL-encoded), so nothing attacker-supplied
    can redirect the request off-host (SSRF barrier).
    """
    import requests as _req

    sort_key = _HF_HUB_SORTS.get(sort, "trendingScore")
    n = max(1, min(int(limit or 30), 100))
    params = {"filter": "gguf", "sort": sort_key, "limit": n, "full": "true"}
    if search.strip():
        params["search"] = search.strip()
    r = _req.get(f"{_HF_HUB_BASE}/api/models", params=params, timeout=15)
    r.raise_for_status()
    data = r.json() if r.content else []
    if not isinstance(data, list):
        return []
    out: List[HfHubModel] = []
    for m in data:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or m.get("modelId") or "")
        if not mid:
            continue
        quants: List[str] = []
        for sib in (m.get("siblings") or []):
            fname = str((sib or {}).get("rfilename") or "")
            if not fname.lower().endswith(".gguf"):
                continue
            hit = _HF_QUANT_RE.search(fname)
            if hit:
                q = hit.group(1).upper()
                if q not in quants:
                    quants.append(q)
        out.append(HfHubModel(
            id=mid,
            downloads=int(m.get("downloads") or 0),
            likes=int(m.get("likes") or 0),
            pipeline_tag=(str(m.get("pipeline_tag")) if m.get("pipeline_tag") else None),
            gated=bool(m.get("gated")),
            pull_tag=f"hf.co/{mid}",
            quants=quants,
        ))
    return out


@router.get("/setup/hf_models", response_model=HfModelsResponse)
def hf_models(search: str = "", sort: str = "trending_score", limit: int = 30) -> HfModelsResponse:
    """List GGUF models on the Hugging Face Hub for one-click Ollama pulls.

    Powers the Settings "Browse Hugging Face" panel. Each entry carries a
    ready-to-use ``pull_tag`` (``hf.co/<repo>``) plus any detected quant
    labels so the user can pull a specific quantization. Returns an empty
    list on any upstream failure so the panel degrades gracefully.
    """
    try:
        return HfModelsResponse(models=_hf_hub_gguf_list(search, sort, limit))
    except Exception:
        return HfModelsResponse(models=[])
