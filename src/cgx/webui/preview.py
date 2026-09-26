"""Shared read-only web-file serving for live previews.

Serves files from within a project/site root for rendering in a *sandboxed*
iframe. Two protections, used by every preview route:

* **Path containment** -- realpath + ``startswith(root + os.sep)`` before any
  filesystem access, so ``../`` / absolute paths can't escape the root.
* **Web-content allowlist** -- only render-relevant extensions, so the
  endpoint can never be turned into a general read of arbitrary project files
  (``.env``, keys, source of a non-static project).
"""

from __future__ import annotations

import mimetypes
import os

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

__all__ = ["serve_web_file", "PREVIEW_ALLOWED_EXT"]

PREVIEW_ALLOWED_EXT = frozenset({
    ".html", ".htm", ".css", ".js", ".mjs", ".json", ".map", ".txt",
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".ico", ".bmp",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".webmanifest",
})
_MAX_BYTES = 15 * 1024 * 1024


def serve_web_file(root_real: str, rel_path: str) -> Response:
    """Serve ``root_real/rel_path`` (defaulting to index.html) safely."""
    rel = (rel_path or "").strip()
    if not rel or rel.endswith("/"):
        rel = (rel + "index.html").lstrip("/")
    candidate = os.path.realpath(os.path.join(root_real, rel))
    if not (candidate == root_real or candidate.startswith(root_real + os.sep)):
        raise HTTPException(status_code=403, detail="path escapes site root")
    ext = os.path.splitext(candidate)[1].lower()
    if ext not in PREVIEW_ALLOWED_EXT:
        raise HTTPException(status_code=404, detail=f"{ext or 'file'} is not previewable")
    if not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="file not found")
    try:
        if os.path.getsize(candidate) > _MAX_BYTES:
            raise HTTPException(status_code=413, detail="file too large to preview")
    except OSError:
        raise HTTPException(status_code=404, detail="file not found") from None
    media_type = mimetypes.guess_type(candidate)[0] or "application/octet-stream"
    return FileResponse(candidate, media_type=media_type, headers={
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
    })
