

"""cgx -- Code Graph eXecution.

This top-level module ensures the sibling ``skills`` package (which lives
at the repo root alongside ``src/``) is importable even when cgx is run
from an editable install whose .pth file only points at ``src/``. Without
this bootstrap, ``import skills`` silently fails when ``cgx-ui`` is
launched from outside the repo root, which leaves every skill-detection
call returning an empty list and lets the LLM substitute the wrong
framework (e.g. Vue SFC syntax in a React .jsx file).
"""

from __future__ import annotations

import os as _os
import sys as _sys

# faiss-cpu and torch each vendor their own OpenMP runtime; on macOS importing
# both into one process (the local-embedding index path: torch encoder +
# faiss.IndexFlat) trips "OMP: Error #15 ... libomp already initialized" and the
# interpreter aborts. Opt into the standard single-runtime workaround here --
# the earliest import point, before either library loads -- so local jina
# embeddings + faiss actually run instead of crashing. ``setdefault`` respects
# an explicit user override. Harmless when torch/faiss aren't installed.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ``cgx`` resolves to ``<repo_root>/src/cgx``. Two parents up is the repo
# root, where ``skills/`` lives in the editable layout. Only prepend it
# when the directory contains the package -- installations that bundle
# ``skills`` into site-packages don't need this.
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if (_os.path.isdir(_os.path.join(_repo_root, "skills"))
        and _repo_root not in _sys.path):
    _sys.path.insert(0, _repo_root)
