"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config):  # noqa: D401 - pytest hook
    # Ensure profiles tests don't touch the developer's real ~/.cgx.
    if "CGX_CONFIG_DIR" not in os.environ:
        tmp = REPO_ROOT / ".pytest_cgx_config"
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["CGX_CONFIG_DIR"] = str(tmp)
