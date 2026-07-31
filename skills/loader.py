"""Dynamic loader for user-authored custom skills.

Custom skills live as standalone ``.py`` files under
``$CGX_CONFIG_DIR/skills/`` (default ``~/.cgx/skills/``), each defining
exactly one ``Skill`` subclass. Once loaded they participate in
detection / prompt composition / validation identically to the
built-in skills in :data:`skills.SKILLS`.

Kept free of any ``cgx.*`` import so the ``skills`` package stays
independent of the agent layer (see ``skills/__init__.py``'s docstring).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from skills.base import Skill

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(os.environ.get("CGX_CONFIG_DIR", str(Path.home() / ".cgx")))
CUSTOM_SKILLS_DIR = CONFIG_DIR / "skills"

_PROBE_PATH = Path(__file__).resolve().parent / "_skill_probe.py"
_PROBE_TIMEOUT_SECONDS = 8


@dataclass
class SkillValidationResult:
    ok: bool
    error_kind: str = ""  # "syntax_error" | "no_skill_class" |
                          # "multiple_skill_classes" | "name_collision" |
                          # "runtime_error" | "timeout"
    error_detail: str = ""
    meta: Optional[Dict[str, Any]] = None


def _ensure_dir() -> None:
    CUSTOM_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CUSTOM_SKILLS_DIR, 0o700)
    except Exception as e:
        logger.warning("skills.loader: chmod on %s failed: %s: %s",
                       CUSTOM_SKILLS_DIR, type(e).__name__, e)


def list_custom_skill_files() -> List[Path]:
    if not CUSTOM_SKILLS_DIR.exists():
        return []
    return sorted(CUSTOM_SKILLS_DIR.glob("*.py"))


def read_custom_skill_source(name: str) -> Optional[str]:
    path = CUSTOM_SKILLS_DIR / f"{name}.py"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _skill_subclasses_in_module(module: Any) -> List[type]:
    return [
        v for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, Skill) and v is not Skill
        and v.__module__ == module.__name__
    ]


def _import_skill_file(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"cgx_custom_skill_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Memoized on the custom-skills directory's max mtime so repeated
# detect_skills()/skills_by_names() calls (once per task) don't re-import
# every custom skill file on every call.
_cache_signature: Optional[float] = None
_cache: List[Skill] = []


def _dir_signature() -> float:
    files = list_custom_skill_files()
    return max((f.stat().st_mtime for f in files), default=0.0)


def load_custom_skills(force: bool = False) -> List[Skill]:
    """Load every custom skill under :data:`CUSTOM_SKILLS_DIR`.

    A broken file is logged and skipped rather than raised -- one bad
    custom skill must never take down the whole registry, since this
    runs on the hot path of every ``detect_skills()`` call.
    """
    global _cache_signature, _cache
    sig = _dir_signature()
    if not force and _cache_signature == sig:
        return list(_cache)
    out: List[Skill] = []
    for path in list_custom_skill_files():
        try:
            module = _import_skill_file(path)
            classes = _skill_subclasses_in_module(module)
            if len(classes) != 1:
                logger.warning(
                    "skills.loader: %s defines %d Skill subclasses (want 1); skipping",
                    path, len(classes))
                continue
            out.append(classes[0]())
        except Exception as e:
            logger.warning("skills.loader: failed to load %s: %s: %s",
                           path, type(e).__name__, e)
    _cache_signature = sig
    _cache = out
    return list(out)


def _run_probe(path: Path) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, str(_PROBE_PATH), str(path)],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_kind": "timeout",
                "error_detail": f"skill validation exceeded {_PROBE_TIMEOUT_SECONDS}s "
                                "(likely a hanging detect())"}
    lines = (proc.stdout or "").strip().splitlines()
    last = lines[-1] if lines else ""
    try:
        return json.loads(last)
    except Exception:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {"ok": False, "error_kind": "runtime_error",
                "error_detail": detail[-2000:] or f"probe exited {proc.returncode}"}


def validate_skill_source(source: str, known_names: Set[str]) -> SkillValidationResult:
    """Validate a candidate custom skill's source before persisting it.

    Order: syntax check (no execution) -> subprocess dry-import plus one
    bounded ``detect()`` call (catches hangs/crashes without risking the
    web server's own request thread) -> name/alias collision check.
    ``known_names`` should already be lower-cased.
    """
    try:
        ast.parse(source, filename="<custom-skill>")
    except SyntaxError as e:
        return SkillValidationResult(
            ok=False, error_kind="syntax_error",
            error_detail=f"line {e.lineno}: {e.msg}")

    fd, tmp_name = tempfile.mkstemp(suffix=".py", prefix="cgx_skill_")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(source)
        result = _run_probe(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    if not result.get("ok"):
        return SkillValidationResult(
            ok=False,
            error_kind=str(result.get("error_kind") or "runtime_error"),
            error_detail=str(result.get("error_detail") or ""),
        )

    meta = result.get("meta") or {}
    name = str(meta.get("name") or "").strip().lower()
    aliases = [str(a).strip().lower() for a in meta.get("aliases") or []]
    if not name:
        return SkillValidationResult(
            ok=False, error_kind="runtime_error",
            error_detail="skill's `name` attribute is empty")
    if name in known_names or any(a in known_names for a in aliases):
        return SkillValidationResult(
            ok=False, error_kind="name_collision",
            error_detail=f"'{name}' collides with an existing skill's name or alias")

    return SkillValidationResult(ok=True, meta=meta)


def save_custom_skill(name: str, source: str) -> None:
    _ensure_dir()
    path = CUSTOM_SKILLS_DIR / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def delete_custom_skill(name: str) -> bool:
    path = CUSTOM_SKILLS_DIR / f"{name}.py"
    if not path.exists():
        return False
    path.unlink()
    return True
