# src/cgx/ast/module_path.py
from __future__ import annotations

import os
from src.cgx.logging_setup import get_logger

logger = get_logger("module_path")


def compute_module_path(project_root: str, file_path: str) -> str:
    """
    Compute a deterministic dotted module path for a Python file within a project.

    Rules (purely deterministic; no runtime inspection of __init__.py):
      1) Normalize paths and compute relpath from project_root.
      2) Strip the trailing ".py".
      3) Replace path separators with dots.
      4) If the file is "__init__.py", drop the final segment (package module path).

    Examples
    --------
    project_root=/repo
      /repo/pkg/mod.py           -> "pkg.mod"
      /repo/pkg/__init__.py      -> "pkg"
      /repo/app/sub/utils.py     -> "app.sub.utils"

    Notes
    -----
    - If file_path is outside project_root, we return a dotted path built from the
      normalized absolute path sans drive letter (on Windows) and log a warning.
    - This function is deterministic and side-effect-free.

    Parameters
    ----------
    project_root : str
        Absolute or relative path to the repository root.
    file_path : str
        Absolute or relative path to a Python source file.

    Returns
    -------
    str
        Dotted module path (may be empty string if relpath cannot be derived).
    """
    try:
        project_root = os.path.abspath(project_root)
        file_path = os.path.abspath(file_path)

        # Defensive normalization for separators
        rel_path = os.path.relpath(file_path, project_root)
        rel_path = rel_path.replace("\\", "/")  # normalize Windows slashes

        # If file is outside root, fallback to absolute sans drive
        if rel_path.startswith(".."):
            logger.warning("File %s is outside project root %s", file_path, project_root)
            drive, tail = os.path.splitdrive(file_path)
            rel_path = tail.lstrip(os.sep).replace("\\", "/")

        # Strip extension
        if rel_path.endswith(".py"):
            rel_path = rel_path[:-3]

        # Replace separators with dots
        mod_path = rel_path.replace("/", ".")

        # Drop trailing .__init__
        if mod_path.endswith(".__init__"):
            mod_path = mod_path.rsplit(".", 1)[0]

        # Guarantee non-empty
        if not mod_path:
            logger.warning("Empty module path for file %s (root=%s)", file_path, project_root)
            return os.path.splitext(os.path.basename(file_path))[0]

        return mod_path

    except Exception as e:
        logger.error("Failed to compute module path for %s: %s", file_path, e)
        return os.path.splitext(os.path.basename(file_path))[0]
