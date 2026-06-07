#!/usr/bin/env python3
"""CGX web UI launcher -- boots the FastAPI + React stack."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cgx.webui.launch import launch  # noqa: E402


if __name__ == "__main__":
    launch()
