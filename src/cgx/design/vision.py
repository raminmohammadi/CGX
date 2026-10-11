"""Vision critic: let a local vision model LOOK at the rendered screens.

Text checks and a text-only critic cannot judge layout: a header squeezed into
the chat column, a composer overflowing its card, a wall of same-weight text.
When the runtime smoke has produced screenshots and an Ollama vision model is
available (``qwen3-vl``, ``llava``, ``minicpm-v``, ``gemma3`` ...), one bounded
call shows the model the brief and the screenshots and asks for concrete,
per-file fixes, which the Verifier regenerates like any other critic output.

Optional and quiet: ``CGX_VISION_MODEL`` names the model (``auto`` -- the
default -- picks an installed vision-capable model from the active Ollama
server; ``off`` disables). Only the Ollama HTTP API is used (``requests`` is a
core dependency); any failure degrades to "no fixes".
"""

from __future__ import annotations

import base64
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from cgx.design.brief import DesignBrief
from cgx.design.critic import MAX_FIXES, parse_fixes

logger = logging.getLogger(__name__)

__all__ = ["vision_enabled", "resolve_vision_model", "critique_screenshots"]

_VISION_NAME_RE = re.compile(r"(vl|vision|llava|minicpm-v|moondream|pixtral|gemma3(?!n)|qwen.*vl|bakllava)", re.I)
_MAX_SHOTS = 3
_TIMEOUT_S = 240.0

_PROMPT = (
    "You are a strict senior product designer reviewing screenshots of a web app a coding agent just built. "
    "Judge LAYOUT and VISUAL QUALITY. You must NOT reply PASS if ANY of these is visible: text like 0, NaN, "
    "undefined or null where real values belong; an empty chart/graph area or a chart card with no chart; "
    "KPI/stat tiles stacked one per row on a wide screen instead of side by side; duplicated controls (two "
    "search boxes or two filters for the same list); a stray badge/button that belongs nowhere; a header "
    "narrower than the page; controls overflowing their container; an unstyled native select/button; missing "
    "navigation or page title; cramped or inconsistent spacing; unreadable contrast; a broken-looking brand "
    "name. Ignore copy and behaviour you cannot see.\n"
    "Reply exactly PASS only when none of the above applies. Otherwise reply with up to "
    f"{MAX_FIXES} lines, each `<file path>: <one concrete CSS/markup fix>`, most visible problem first, "
    "naming ONLY files from the list given. No prose."
)


def vision_enabled() -> bool:
    return (os.environ.get("CGX_VISION_MODEL") or "auto").strip().lower() not in ("0", "off", "false", "no", "none")


def _ollama_base_url(provider: Any) -> Optional[str]:
    for obj in (provider, getattr(provider, "inner", None), getattr(provider, "_inner", None)):
        url = getattr(obj, "base_url", None) if obj is not None else None
        if url and "11434" in str(url) or (url and getattr(obj, "kind", "") == "ollama"):
            return str(url).rstrip("/")
    env = os.environ.get("OLLAMA_HOST") or os.environ.get("CGX_OLLAMA_URL")
    if env:
        return env if env.startswith("http") else f"http://{env}"
    return "http://localhost:11434"


def resolve_vision_model(provider: Any) -> Optional[Tuple[str, str]]:
    """``(base_url, model)`` for a usable vision model, or ``None``."""
    if not vision_enabled():
        return None
    base = _ollama_base_url(provider)
    if not base:
        return None
    wanted = (os.environ.get("CGX_VISION_MODEL") or "auto").strip()
    try:
        import requests
        resp = requests.get(f"{base}/api/tags", timeout=4)
        names = [m.get("name", "") for m in (resp.json() or {}).get("models", [])]
    except Exception:
        return None
    if wanted.lower() != "auto":
        hits = [n for n in names if n == wanted or n.split(":")[0] == wanted]
        return (base, hits[0]) if hits else None
    cands = [n for n in names if _VISION_NAME_RE.search(n)]
    if not cands:
        return None
    cands.sort(key=lambda n: (0 if "vl" in n.lower() else 1, n))
    return base, cands[0]


def _b64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return None


def critique_screenshots(brief: Any, shots: Dict[str, str], ui_files: List[str], provider: Any,
                         model: Optional[Tuple[str, str]] = None) -> Dict[str, str]:
    """Return ``{path: fix}`` from a vision model's review of ``shots`` (route -> png)."""
    b = brief if isinstance(brief, DesignBrief) else DesignBrief.from_dict(brief)
    if not b.applies or not shots:
        return {}
    model = model or resolve_vision_model(provider)
    if not model:
        return {}
    base, name = model
    images: List[str] = []
    routes: List[str] = []
    for route, path in list(shots.items())[:_MAX_SHOTS]:
        enc = _b64(path)
        if enc:
            images.append(enc)
            routes.append(route)
    if not images:
        return {}
    user = (f"{_PROMPT}\n\nPRODUCT: {b.product_name} -- {b.archetype_name}. Theme: {b.theme}. Layout spec: {b.layout}\n"
            f"SCREENSHOTS (in order): {', '.join(routes)} at 1280x900.\n"
            f"FILES you may name:\n" + "\n".join(f"- {p}" for p in ui_files))
    try:
        import requests
        resp = requests.post(f"{base}/api/chat", json={
            "model": name, "stream": False,
            "messages": [{"role": "user", "content": user, "images": images}],
            "options": {"temperature": 0, "num_predict": 600},
        }, timeout=_TIMEOUT_S)
        text = str(((resp.json() or {}).get("message") or {}).get("content") or "")
    except Exception as exc:  # pragma: no cover - depends on a live server
        logger.debug("vision critic failed: %s", exc)
        return {}
    return parse_fixes(text, ui_files)
