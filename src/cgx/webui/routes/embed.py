

"""Embedding model discovery + Hugging Face download streaming.

Mirrors the ``/ollama/*`` setup endpoints so the React app can list the
curated embedding models, see which are already cached locally, and pull
fresh ones with the same ``{status,total,completed}`` SSE shape used by
``/ollama/pull``.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import ssl
import threading
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cgx.embeddings.catalog import EMBED_MODEL_CATALOG, find_by_name, is_cached

logger = logging.getLogger(__name__)
router = APIRouter(tags=["embed"])


class EmbedModelInfo(BaseModel):
    name: str
    label: str
    kind: str
    dim: int
    max_tokens: int
    size_gb: float
    description: str
    cached: bool


class EmbedModelsResponse(BaseModel):
    choices: List[EmbedModelInfo]
    recommended_default: str


@router.get("/embed/models", response_model=EmbedModelsResponse)
def list_embed_models() -> EmbedModelsResponse:
    choices = [
        EmbedModelInfo(
            name=m.name, label=m.label, kind=m.kind, dim=m.dim,
            max_tokens=m.max_tokens, size_gb=m.size_gb,
            description=m.description, cached=is_cached(m.name),
        )
        for m in EMBED_MODEL_CATALOG
    ]
    return EmbedModelsResponse(
        choices=choices,
        recommended_default="jinaai/jina-embeddings-v2-base-code",
    )


class EmbedPullRequest(BaseModel):
    model: str


# Hugging Face siblings we never want to fetch — duplicate weights or runtime
# variants that sentence-transformers / transformers doesn't load.
_SKIP_EXTS = (".onnx", ".gguf", ".mlpackage", ".msgpack", ".h5", ".tflite")


def _prepare_online_pull() -> dict:
    """Warm OpenSSL on the main thread and clear offline-mode flags.

    Returns the prior env-var values so the caller can restore them. Two
    distinct issues have to be neutralised before the worker thread can
    talk to huggingface.co:

    * ``cgx.webui.launch`` opts the process into ``HF_HUB_OFFLINE=1`` when
      the default embed model is already cached, which short-circuits the
      Hub client. ``hf_hub_download`` and ``HfApi.model_info`` both read
      ``huggingface_hub.constants.HF_HUB_OFFLINE``, so we override both the
      env var *and* the already-imported module constant.
    * After ``torch.cuda`` initialisation, ``SSLContext(PROTOCOL_TLS_CLIENT)``
      raises ``_ssl.c:3076 "unknown error"`` the first time it runs on a
      non-main thread. Materialising one default context on the main thread
      first avoids that.
    """
    try:
        ssl.create_default_context()
    except Exception:
        pass
    prev = {
        "HF_HUB_OFFLINE": os.environ.pop("HF_HUB_OFFLINE", None),
        "TRANSFORMERS_OFFLINE": os.environ.pop("TRANSFORMERS_OFFLINE", None),
    }
    try:
        import huggingface_hub.constants as _hc
        prev["_hub_offline_const"] = _hc.HF_HUB_OFFLINE
        _hc.HF_HUB_OFFLINE = False
    except Exception:
        prev["_hub_offline_const"] = None
    return prev


def _restore_online_pull(prev: dict) -> None:
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        v = prev.get(k)
        if v is not None:
            os.environ[k] = v
    try:
        import huggingface_hub.constants as _hc
        if prev.get("_hub_offline_const") is not None:
            _hc.HF_HUB_OFFLINE = prev["_hub_offline_const"]
    except Exception:
        pass


@router.post("/embed/pull")
async def embed_pull(req: EmbedPullRequest) -> EventSourceResponse:
    """Stream Hugging Face download progress as SSE events.

    Each ``progress`` event payload is ``{status, total, completed}`` so the
    existing ollama-pull frontend consumer can render the bar unchanged. A
    final ``done`` event closes the stream. Files known to be redundant are
    filtered out so a repo with both ``model.safetensors`` and a
    ``pytorch_model.bin`` only downloads the safetensors copy.
    """
    if not find_by_name(req.model):
        logger.info("embed_pull: %r not in catalog — proceeding anyway", req.model)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _SENTINEL = object()
    prev_env = _prepare_online_pull()

    def _emit(status: str, total: int, completed: int,
              error: Optional[str] = None) -> None:
        payload = {"status": status, "total": total, "completed": completed}
        if error is not None:
            payload["error"] = error
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    def _worker() -> None:
        try:
            from huggingface_hub import HfApi, hf_hub_download

            api = HfApi()
            _emit("Resolving files…", 0, 0)
            info = api.model_info(req.model, files_metadata=True)
            siblings = list(info.siblings or [])

            has_st = any((s.rfilename or "").endswith(".safetensors")
                         for s in siblings)
            files: List[tuple[str, int]] = []
            for s in siblings:
                fname = s.rfilename or ""
                if not fname or fname.endswith(_SKIP_EXTS):
                    continue
                # Prefer .safetensors over legacy .bin / .pt duplicates.
                if has_st and fname.endswith((".bin", ".pt")):
                    continue
                files.append((fname, int(s.size or 0)))

            total_bytes = sum(sz for _, sz in files)
            completed_bytes = 0
            _emit(f"Downloading {len(files)} files…", total_bytes, 0)

            for fname, sz in files:
                _emit(f"pulling {fname}", total_bytes, completed_bytes)
                hf_hub_download(repo_id=req.model, filename=fname)
                completed_bytes += sz
                _emit(f"pulled {fname}", total_bytes, completed_bytes)

            _emit("success", total_bytes, total_bytes)
        except Exception as exc:
            logger.exception("embed_pull failed for %r: %s", req.model, exc)
            _emit("error", 0, 0, error=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            _restore_online_pull(prev_env)
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    threading.Thread(target=_worker, daemon=True, name="hf-embed-pull").start()

    async def _gen():
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            yield {"event": "progress", "data": _json.dumps(item)}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(_gen())
