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
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from skills import markdown_skill as _md
from skills.base import Skill

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(os.environ.get("CGX_CONFIG_DIR", str(Path.home() / ".cgx")))
CUSTOM_SKILLS_DIR = CONFIG_DIR / "skills"

#: Per-repo skills live here, relative to a project root, so a team can
#: commit shared skills alongside the code they describe.
REPO_SKILLS_SUBDIR = os.path.join(".cgx", "skills")

_PROBE_PATH = Path(__file__).resolve().parent / "_skill_probe.py"
_PROBE_TIMEOUT_SECONDS = 8

# Skill names reach the CRUD helpers straight from a REST path parameter,
# so they are restricted to a conservative identifier allowlist before ever
# being turned into a ``<name>.py`` filename. This blocks ``..`` / absolute
# / separator traversal (path injection) at the door.
_SKILL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _safe_skill_path(name: str) -> Path:
    """Resolve ``<name>.py`` under :data:`CUSTOM_SKILLS_DIR`.

    Rejects any ``name`` that is not a plain identifier or that would
    resolve outside the custom-skills directory, raising :class:`ValueError`.
    """
    if not _SKILL_NAME_RE.match(name or ""):
        raise ValueError(f"invalid skill name: {name!r}")
    # Containment is re-checked with os.path.realpath + str.startswith rather
    # than Path.resolve()/relative_to: the two are equivalent, but only the
    # former is recognised as a path-injection barrier by static analysis.
    base = os.path.realpath(str(CUSTOM_SKILLS_DIR))
    candidate = os.path.realpath(os.path.join(base, f"{name}.py"))
    if not candidate.startswith(base + os.sep):
        raise ValueError(f"invalid skill name: {name!r}")
    return Path(candidate)


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
    try:
        path = _safe_skill_path(name)
    except ValueError:
        return None
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
    path = _safe_skill_path(name)
    path.write_text(source, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def delete_custom_skill(name: str) -> bool:
    try:
        path = _safe_skill_path(name)
    except ValueError:
        return False
    if not path.exists():
        return False
    path.unlink()
    return True


# ---------------------------------------------------------------------------
# Markdown skills (SKILL.md) -- the code-free, Anthropic-style skill format.
#
# On disk, either form is accepted under a skills root:
#     <root>/<name>/SKILL.md   (preferred; may sit beside references/, scripts/)
#     <root>/<name>.md         (quick single-file skill)
# The global root is CUSTOM_SKILLS_DIR (managed by the CRUD helpers below);
# a per-repo root is <project_root>/.cgx/skills (discovered, not CRUD-managed).
# ---------------------------------------------------------------------------


def _safe_markdown_skill_dir(name: str) -> Path:
    """Resolve the directory ``<name>/`` under :data:`CUSTOM_SKILLS_DIR`.

    Reuses the Python-skill path-injection barrier (identifier allowlist +
    realpath containment) so a hostile ``name`` cannot escape the skills dir.
    """
    if not _SKILL_NAME_RE.match(name or ""):
        raise ValueError(f"invalid skill name: {name!r}")
    base = os.path.realpath(str(CUSTOM_SKILLS_DIR))
    candidate = os.path.realpath(os.path.join(base, name))
    if not candidate.startswith(base + os.sep):
        raise ValueError(f"invalid skill name: {name!r}")
    return Path(candidate)


def _markdown_skill_file(root: Path, name: str) -> Optional[Path]:
    """Return the SKILL.md (dir form) or ``<name>.md`` (flat form), if present."""
    dir_form = root / name / "SKILL.md"
    if dir_form.is_file():
        return dir_form
    flat = root / f"{name}.md"
    if flat.is_file():
        return flat
    return None


def _iter_markdown_skill_files(root: Path):
    """Yield ``(name, path)`` for every markdown skill directly under ``root``.

    ``name`` is the on-disk directory/file stem; the loaded skill may adopt a
    different ``name:`` from its frontmatter. References/ and scripts/ inside a
    skill directory are intentionally not scanned.
    """
    if not root.exists():
        return
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_dir():
                skill_md = entry / "SKILL.md"
                if skill_md.is_file():
                    yield entry.name, skill_md
            elif entry.suffix.lower() == ".md" and entry.name != "SKILL.md":
                yield entry.stem, entry
        except OSError:
            continue


def _markdown_signature(root: Path) -> float:
    files = [p for _, p in _iter_markdown_skill_files(root)]
    return max((f.stat().st_mtime for f in files), default=0.0)


# Memoized per resolved root path -> (signature, skills), mirroring the
# Python-skill cache so per-task detect_skills() calls stay cheap.
_md_cache: Dict[str, "tuple"] = {}


def load_markdown_skills(root: Optional[Path] = None,
                         scope: str = "global",
                         force: bool = False) -> List[Skill]:
    """Load every markdown skill under ``root`` (default: the global dir).

    Fails soft per file: a skill that cannot be read/parsed is logged and
    skipped, never raised, since this runs on the hot detection path.
    """
    base = Path(root) if root is not None else CUSTOM_SKILLS_DIR
    key = os.path.realpath(str(base))
    sig = _markdown_signature(base)
    cached = _md_cache.get(key)
    if not force and cached is not None and cached[0] == sig:
        return list(cached[1])
    out: List[Skill] = []
    for name, path in _iter_markdown_skill_files(base):
        try:
            content = path.read_text(encoding="utf-8")
            out.append(_md.from_source(
                content, name_hint=name, scope=scope, source_path=str(path)))
        except Exception as e:  # noqa: BLE001 - one bad skill must not break all
            logger.warning("skills.loader: failed to load markdown skill %s: %s: %s",
                           path, type(e).__name__, e)
    _md_cache[key] = (sig, out)
    return list(out)


def load_repo_markdown_skills(project_root: str) -> List[Skill]:
    """Load markdown skills committed under ``<project_root>/.cgx/skills``."""
    if not project_root:
        return []
    root = Path(project_root) / ".cgx" / "skills"
    return load_markdown_skills(root=root, scope="repo")


def list_markdown_skill_names() -> List[str]:
    """Names of global (CRUD-managed) markdown skills."""
    return [name for name, _ in _iter_markdown_skill_files(CUSTOM_SKILLS_DIR)]


def read_markdown_skill_source(name: str) -> Optional[str]:
    """Return the raw ``SKILL.md`` text for a global markdown skill."""
    try:
        _safe_markdown_skill_dir(name)  # validate name shape only
    except ValueError:
        return None
    path = _markdown_skill_file(CUSTOM_SKILLS_DIR, name)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def validate_markdown_skill_source(content: str,
                                   known_names: Set[str],
                                   name_hint: str = "") -> SkillValidationResult:
    """Validate a candidate ``SKILL.md`` before persisting it.

    No code is executed -- markdown skills are parsed and checked statically,
    so they are strictly safer than Python skills. ``known_names`` should
    already be lower-cased.
    """
    ok, kind, detail, meta = _md.check_source(
        content, name_hint=name_hint, known_names=known_names)
    if not ok:
        return SkillValidationResult(ok=False, error_kind=kind, error_detail=detail)
    return SkillValidationResult(ok=True, meta=meta)


def save_markdown_skill(name: str, content: str) -> None:
    """Persist a global markdown skill at ``<dir>/<name>/SKILL.md`` (0600)."""
    _ensure_dir()
    skill_dir = _safe_markdown_skill_dir(name)
    skill_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(skill_dir, 0o700)
    except OSError:
        pass
    # If a flat <name>.md exists, migrate to the canonical dir form.
    flat = CUSTOM_SKILLS_DIR / f"{name}.md"
    if flat.exists():
        try:
            flat.unlink()
        except OSError:
            pass
    path = skill_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    _md_cache.pop(os.path.realpath(str(CUSTOM_SKILLS_DIR)), None)


def delete_markdown_skill(name: str) -> bool:
    """Delete a global markdown skill (dir form or flat form)."""
    try:
        skill_dir = _safe_markdown_skill_dir(name)
    except ValueError:
        return False
    removed = False
    if skill_dir.is_dir():
        import shutil
        shutil.rmtree(skill_dir, ignore_errors=True)
        removed = True
    flat = CUSTOM_SKILLS_DIR / f"{name}.md"
    if flat.is_file():
        try:
            flat.unlink()
            removed = True
        except OSError:
            pass
    if removed:
        _md_cache.pop(os.path.realpath(str(CUSTOM_SKILLS_DIR)), None)
    return removed
