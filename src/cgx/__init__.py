

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

# ``cgx`` resolves to ``<repo_root>/src/cgx``. Two parents up is the repo
# root, where ``skills/`` lives in the editable layout. Only prepend it
# when the directory contains the package -- installations that bundle
# ``skills`` into site-packages don't need this.
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if (_os.path.isdir(_os.path.join(_repo_root, "skills"))
        and _repo_root not in _sys.path):
    _sys.path.insert(0, _repo_root)
