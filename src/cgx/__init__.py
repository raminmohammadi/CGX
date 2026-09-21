

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

# faiss-cpu and torch each vendor their own OpenMP runtime. Loading both into
# one process -- exactly the local-embedding index AND query paths (torch
# encoder + faiss.IndexFlat) -- is unstable on multi-core machines: it either
# aborts with "OMP: Error #15 ... libomp already initialized" or, once torch's
# multi-thread OpenMP pool contends with faiss's, SEGFAULTS the interpreter
# (reproduced on macOS; possible on Linux depending on the wheels). Two guards,
# set here at the earliest import point -- BEFORE torch/faiss/numpy load, which
# is the only point early enough for torch to size its thread pool from the env
# and for a BYO embedder (which never calls our build path) to be covered:
#   * KMP_DUPLICATE_LIB_OK=TRUE  -> tolerate the duplicate runtime (no abort).
#   * OMP_NUM_THREADS=1          -> bound the shared OpenMP pool so torch's
#     threads can't collide with faiss's (empirically the actual segfault fix;
#     capping faiss alone does not help, capping torch does). GPU/MPS matmuls
#     use CUDA/Metal, not OpenMP, so GPU embedding speed is unaffected; the cost
#     is single-threaded CPU embedding, which the incremental cache amortizes.
# Both use setdefault so an explicit user value wins -- a user on ABI-matched
# faiss/torch builds can export OMP_NUM_THREADS=<n> to restore CPU parallelism.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

# ``cgx`` resolves to ``<repo_root>/src/cgx``. Two parents up is the repo
# root, where ``skills/`` lives in the editable layout. Only prepend it
# when the directory contains the package -- installations that bundle
# ``skills`` into site-packages don't need this.
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if (_os.path.isdir(_os.path.join(_repo_root, "skills"))
        and _repo_root not in _sys.path):
    _sys.path.insert(0, _repo_root)
