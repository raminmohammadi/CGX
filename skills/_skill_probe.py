"""Standalone subprocess probe used by ``skills.loader.validate_skill_source``.

Run as: ``python _skill_probe.py <path-to-candidate-skill.py>``.

Prints exactly one JSON line to stdout: ``{"ok": true, "meta": {...}}`` on
success, or ``{"ok": false, "error_kind": ..., "error_detail": ...}`` (exit
code 1) on failure. Kept stdlib-only and repo-path-independent since it
runs as a bare subprocess spawned from a validate-on-save request.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _fail(error_kind: str, error_detail: str) -> None:
    print(json.dumps({"ok": False, "error_kind": error_kind, "error_detail": error_detail}))
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        _fail("runtime_error", "usage: _skill_probe.py <path>")
        return
    path = Path(sys.argv[1])

    # Make the repo root importable so `from skills.base import Skill` works
    # regardless of the probe subprocess's own cwd.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from skills.base import Skill
    except Exception as e:
        _fail("runtime_error", f"probe setup failed: {type(e).__name__}: {e}")
        return

    try:
        spec = importlib.util.spec_from_file_location("cgx_candidate_skill", str(path))
        if spec is None or spec.loader is None:
            _fail("runtime_error", "could not create module spec")
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except SyntaxError as e:
        _fail("syntax_error", f"line {e.lineno}: {e.msg}")
        return
    except Exception as e:
        _fail("runtime_error", f"{type(e).__name__}: {e}")
        return

    classes = [
        v for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, Skill) and v is not Skill
        and v.__module__ == module.__name__
    ]
    if len(classes) == 0:
        _fail("no_skill_class", "no Skill subclass found in the submitted source")
        return
    if len(classes) > 1:
        _fail("multiple_skill_classes",
              f"found {len(classes)} Skill subclasses; exactly one is required")
        return

    try:
        instance = classes[0]()
        score = float(instance.detect("test scaffold objective"))
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"detect() returned {score!r}, expected a float in [0, 1]")
    except Exception as e:
        _fail("runtime_error", f"{type(e).__name__}: {e}")
        return

    meta = {
        "name": str(getattr(instance, "name", "") or ""),
        "role": str(getattr(instance, "role", "") or ""),
        "aliases": [str(a) for a in (getattr(instance, "aliases", ()) or ())],
        "description": str(getattr(instance, "description", "") or ""),
    }
    print(json.dumps({"ok": True, "meta": meta}))


if __name__ == "__main__":
    main()
