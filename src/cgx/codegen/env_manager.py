

"""Dynamic dependency management for the code generation sandbox.

Before running pytest in a temp sandbox, scans the generated files for
import statements, cross-references them against requirements.txt /
package.json, and pip-installs any missing packages so tests are not
blocked by ModuleNotFoundError failures caused by a model choosing a
library that wasn't already declared.

If the tests pass after the dynamic install, the new package names are
appended to ``requirements.txt`` so the dependency becomes permanent.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import logging

from cgx.trace import traced

logger = logging.getLogger(__name__)


# Top-level names that are part of the Python standard library (3.10+).
# This list covers the most common ones; anything importable that isn't
# here will be caught by the live-import probe below.
# Import-name → PyPI distribution-name overrides. Most packages can be
# pip-installed under the same name they're imported as, but a handful
# of common ones differ. Keys are either bare roots (``PIL``) or
# top-two dotted segments for namespace packages (``google.generativeai``).
_IMPORT_TO_PYPI: Dict[str, str] = {
    "google.generativeai": "google-generativeai",
    "google.cloud": "google-cloud",
    "google.oauth2": "google-auth",
    "google.auth": "google-auth",
    "google.api_core": "google-api-core",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "jwt": "PyJWT",
    "yaml": "PyYAML",
    "Crypto": "pycryptodome",
    "magic": "python-magic",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "jose": "python-jose",
    "git": "GitPython",
    "OpenSSL": "pyOpenSSL",
    "serial": "pyserial",
    "usb": "pyusb",
}

# Top-level names that are themselves namespace packages -- pip-installing
# the bare root is meaningless (no such PyPI distribution), so when a
# dotted variant like ``google.generativeai`` is also seen in the source
# the bare root is dropped before the missing-package probe.
_NAMESPACE_ROOTS: frozenset = frozenset({"google", "azure"})

# Commonly imported top-level third-party framework and library roots that are
# never project-local first-party modules. Used by import resolution and
# scaffold validation so transitive framework dependencies (e.g. werkzeug) are
# never misclassified as missing local modules or flagged by phantom gates.
_COMMON_THIRDPARTY_ROOTS: frozenset = frozenset({
    "werkzeug", "flask", "flask_cors", "flask_sqlalchemy", "flask_login",
    "flask_jwt_extended", "flask_migrate", "flask_restful", "flask_wtf",
    "wtforms", "jinja2", "markupsafe", "itsdangerous", "click", "pydantic",
    "fastapi", "starlette", "uvicorn", "gunicorn", "httpx", "requests",
    "urllib3", "aiohttp", "websockets", "sqlalchemy", "alembic", "psycopg2",
    "asyncpg", "pymongo", "redis", "celery", "kombu", "pytest", "pytest_mock",
    "pytest_asyncio", "pytest_cov", "mock", "factory", "numpy", "pandas",
    "scipy", "matplotlib", "seaborn", "torch", "torchvision", "transformers",
    "huggingface_hub", "scikit_learn", "sklearn", "PIL", "cv2", "skimage",
    "nltk", "spacy", "boto3", "botocore", "google", "azure", "aws", "jwt",
    "jose", "yaml", "dotenv", "pydantic_settings", "git", "serial", "usb",
    "bs4", "rich", "typer", "tqdm", "joblib", "loguru", "tenacity",
})


_STDLIB_TOP = frozenset({
    "abc", "ast", "asyncio", "base64", "binascii", "builtins", "cgi",
    "collections", "concurrent", "configparser", "contextlib", "copy",
    "csv", "ctypes", "dataclasses", "datetime", "decimal", "dis",
    "email", "enum", "errno", "faulthandler", "fileinput", "fnmatch",
    "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "glob", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "idlelib", "imaplib", "importlib", "inspect", "io", "ipaddress",
    "itertools", "json", "keyword", "lib2to3", "linecache", "locale",
    "logging", "lzma", "mailbox", "math", "mimetypes", "mmap",
    "modulefinder", "multiprocessing", "netrc", "numbers", "operator",
    "os", "pathlib", "pickle", "pickletools", "platform", "pprint",
    "profile", "py_compile", "queue", "random", "re", "readline",
    "reprlib", "rlcompleter", "runpy", "secrets", "select", "shlex",
    "shutil", "signal", "site", "smtplib", "socket", "socketserver",
    "sqlite3", "ssl", "stat", "statistics", "string", "struct",
    "subprocess", "sys", "sysconfig", "tarfile", "tempfile", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "tomllib", "traceback", "tracemalloc", "types", "typing",
    "unicodedata", "unittest", "urllib", "uuid", "venv", "warnings",
    "weakref", "webbrowser", "wsgiref", "xml", "xmlrpc", "zipfile",
    "zipimport", "zlib", "zoneinfo",
    # always treat the project itself as installed
    "cgx",
})


def _extract_imports_python(source: str) -> Set[str]:
    """Return import names from a Python source string.

    The result contains top-level roots (``streamlit``, ``google``) and,
    for namespace packages listed in :data:`_NAMESPACE_ROOTS`, the
    top-two dotted prefix as well (``google.generativeai``). The latter
    is what downstream resolution maps to the proper PyPI distribution
    name -- the bare namespace root by itself isn't pip-installable.
    """
    roots: Set[str] = set()

    def _add(module: str) -> None:
        if not module:
            return
        parts = module.split(".")
        roots.add(parts[0])
        if len(parts) >= 2 and parts[0] in _NAMESPACE_ROOTS:
            roots.add(".".join(parts[:2]))

    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    _add(node.module)
    except SyntaxError:
        # Fallback regex for files that haven't been fixed yet. Captures
        # dotted module names so namespace packages are still detected.
        for m in re.finditer(
            r"^\s*(?:from\s+([\w.]+)|import\s+([\w.]+))", source, re.MULTILINE
        ):
            mod = m.group(1) or m.group(2)
            _add(mod or "")
    return roots


def _extract_imports_js(source: str) -> Set[str]:
    """Return npm package names from JS/TS import/require statements.

    Covers all common forms so dependency detection is complete:
      * ``import 'pkg'`` / ``import(...)`` / ``require('pkg')``
      * ``import X from 'pkg'`` / ``import {a} from 'pkg'`` / ``import * as X
        from 'pkg'`` / ``export {a} from 'pkg'`` -- the ES-module ``from`` clause
        the previous pattern missed (the bulk of real imports).
    Relative specifiers (``./`` / ``../`` / absolute) are excluded by the
    leading ``[^'"./]`` guard.
    """
    roots: Set[str] = set()
    patterns = (
        r"""(?:import|require)\s*[\(]?\s*['"]([^'"./][^'"]*?)['"]""",
        r"""\bfrom\s+['"]([^'"./][^'"]*?)['"]""",
    )
    for pat in patterns:
        for m in re.finditer(pat, source):
            pkg = m.group(1)
            if pkg.startswith("@"):
                parts = pkg.split("/")
                if len(parts) >= 2:
                    roots.add(f"{parts[0]}/{parts[1]}")
            else:
                roots.add(pkg.split("/")[0])
    return roots


def scan_file_imports(file_path: str) -> Set[str]:
    """Return import roots for a single file based on its extension."""
    p = Path(file_path)
    try:
        source = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()
    ext = p.suffix.lower()
    if ext == ".py":
        return _extract_imports_python(source)
    if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        return _extract_imports_js(source)
    return set()


def scan_imports(file_paths: List[str]) -> Set[str]:
    """Scan a list of files and return the union of all import roots."""
    all_imports: Set[str] = set()
    for fp in file_paths:
        all_imports.update(scan_file_imports(fp))
    return all_imports


def _read_requirements(project_root: str) -> Set[str]:
    """Return normalised package names from requirements.txt."""
    names: Set[str] = set()
    req_path = Path(project_root) / "requirements.txt"
    if not req_path.exists():
        return names
    for line in req_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        pkg = re.split(r"[>=<!;\[]", line)[0].strip().lower().replace("-", "_")
        if pkg:
            names.add(pkg)
    return names


def _read_package_json(project_root: str) -> Set[str]:
    """Return all dependency names from package.json (normalised)."""
    names: Set[str] = set()
    pj = Path(project_root) / "package.json"
    if not pj.exists():
        return names
    try:
        data = json.loads(pj.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return names
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        for pkg in (data.get(section) or {}):
            names.add(str(pkg).lower().replace("-", "_"))
    return names


# Directories that never hold first-party source; pruned from the
# nested-module scan so the walk stays bounded (``.venv`` is usually the
# largest tree by far).
_LOCAL_SCAN_EXCLUDES = frozenset({
    ".venv", "venv", "env", ".git", ".cgx", "__pycache__",
    "node_modules", ".mypy_cache", ".pytest_cache", ".tox",
    "build", "dist", ".eggs",
})


def _is_local_package(name: str, project_root: str) -> bool:
    """Return True when ``name`` matches a first-party file or directory.

    Covers flat layouts (``<root>/<name>/`` or ``<root>/<name>.py``),
    src-layouts (``<root>/src/<name>/`` or ``<root>/src/<name>.py``), and
    modules nested under an arbitrary package directory (e.g.
    ``<root>/backend/main.py`` for a bare ``import main``), including
    namespace packages without an ``__init__.py``. Used to avoid
    mistaking the project's own modules for a missing PyPI distribution
    and attempting to ``pip install`` them.
    """
    root = Path(project_root)
    candidates = (
        root / name,
        root / f"{name}.py",
        root / "src" / name,
        root / "src" / f"{name}.py",
    )
    for c in candidates:
        if c.is_dir() or c.is_file():
            return True
    # Nested layouts: walk the source tree for a matching module basename,
    # pruning virtualenvs / VCS / cache dirs (and any hidden dir) so the
    # traversal stays cheap.
    module_file = f"{name}.py"
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _LOCAL_SCAN_EXCLUDES
                       and not d.startswith(".")]
        if module_file in filenames or name in dirnames:
            return True
    return False


def _probe_importable(
    names: List[str],
    python: Optional[str] = None,
) -> Set[str]:
    """Return the subset of top-level import ``names`` that resolve.

    The check runs in the *target* interpreter (the project venv) via a
    subprocess whenever ``python`` differs from the running one, so a
    package that only happens to be installed in the CGX server process
    (e.g. ``uvicorn``) is not mistaken for a satisfied project
    dependency. Falls back to an in-process ``__import__`` probe when no
    distinct interpreter is given -- unit tests stub this function
    directly for determinism.
    """
    ordered = list(dict.fromkeys(n for n in names if n))
    if not ordered:
        return set()
    if python and python != sys.executable:
        script = (
            "import importlib.util, json, sys\n"
            "names = json.loads(sys.stdin.read())\n"
            "ok = []\n"
            "for n in names:\n"
            "    try:\n"
            "        if importlib.util.find_spec(n) is not None:\n"
            "            ok.append(n)\n"
            "    except Exception:\n"
            "        pass\n"
            "print(json.dumps(ok))\n"
        )
        try:
            proc = subprocess.run(
                [python, "-c", script], input=json.dumps(ordered),
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                return set(json.loads(proc.stdout or "[]"))
            logger.warning(
                "env_manager: import probe exited rc=%d: %s",
                proc.returncode, (proc.stderr or "")[:200])
        except Exception as exc:
            logger.warning(
                "env_manager: import probe subprocess raised %s", exc)
        # Fall through to the in-process probe on any subprocess failure.
    out: Set[str] = set()
    for n in ordered:
        try:
            __import__(n)
            out.add(n)
        except ImportError:
            continue
        except Exception:
            # Side-effectful import: assume present rather than reinstall.
            out.add(n)
    return out


def find_missing_python_packages(
    imports: Set[str],
    project_root: str,
    python: Optional[str] = None,
) -> List[str]:
    """Return PyPI names imported by the code but absent from the venv.

    A package is reported missing when it is imported by the scaffold
    but not importable under ``python`` (the project venv).
    Importability -- not mere declaration in ``requirements.txt`` -- is
    authoritative: a package declared with a version that failed to
    install (e.g. a malformed or unresolvable pin that aborted
    ``pip install -r``) is still reported so the caller can install it.
    Stdlib modules, first-party project packages, and bare namespace
    roots are filtered out, and import names that differ from their PyPI
    distribution name (``google.generativeai`` → ``google-generativeai``,
    ``PIL`` → ``Pillow``, …) are translated via :data:`_IMPORT_TO_PYPI`.
    """
    # Drop bare namespace roots when a dotted variant is also present:
    # ``import google.generativeai`` records both ``google`` and
    # ``google.generativeai`` and we only want to install the latter.
    dotted_roots = {n.split(".", 1)[0] for n in imports if "." in n}
    pruned = {
        n for n in imports
        if not (n in _NAMESPACE_ROOTS and n in dotted_roots)
    }
    # Resolve each surviving import to its PyPI distribution name,
    # dropping stdlib and first-party packages up front.
    candidates: List[Tuple[str, str]] = []
    for name in sorted(pruned):
        root = name.split(".")[0]
        # npm/JS package names are never Python import roots -- a scoped
        # name (``@scope/pkg``) or any name containing ``/`` cannot be a
        # PyPI distribution and must never reach ``pip install``. Guards
        # callers that scan a mixed-language tree and pass JS roots here.
        if name.startswith("@") or "/" in name:
            continue
        # stdlib check operates on the root regardless of dotted form.
        if root.lower().replace("-", "_") in _STDLIB_TOP:
            continue
        # Resolve to the PyPI distribution name. Dotted names without a
        # mapping aren't installable as-is (pip can't install
        # ``google.generativeai`` literally) -- skip them; the matching
        # root entry will have been handled separately.
        if name in _IMPORT_TO_PYPI:
            pypi_name = _IMPORT_TO_PYPI[name]
        elif "." in name:
            continue
        else:
            pypi_name = name
        # Skip first-party project packages -- the project's own top-level
        # folder is not a PyPI distribution and pip cannot install it.
        # Only meaningful for bare root names.
        if "." not in name and _is_local_package(name, project_root):
            continue
        candidates.append((name, pypi_name))
    # Probe importability in the target venv in a single batch so a
    # package present only in the server process isn't skipped.
    importable = _probe_importable([n for n, _ in candidates], python)
    missing: List[str] = []
    seen_pypi: Set[str] = set()
    for name, pypi_name in candidates:
        if name in importable:
            continue
        if pypi_name in seen_pypi:
            continue
        seen_pypi.add(pypi_name)
        missing.append(pypi_name)
    return missing


def _has_uv() -> bool:
    import shutil
    return shutil.which("uv") is not None

def install_packages(
    packages: List[str],
    python: Optional[str] = None,
) -> Dict[str, bool]:
    """pip-install each package; returns {name: success}.

    ``python`` is the interpreter path (defaults to the running one).
    This is designed to install into the SANDBOX's Python environment.
    Gracefully falls back to `uv pip` if standard pip fails or is unavailable.
    """
    if not packages:
        return {}
    py = python or sys.executable
    results: Dict[str, bool] = {}
    has_uv = _has_uv()
    
    for pkg in packages:
        logger.info("env_manager: installing missing package %r", pkg)
        try:
            # Try pip first
            cmd = [py, "-m", "pip", "install", "--quiet", "--no-input", pkg]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode == 0:
                results[pkg] = True
                continue
                
            # If pip failed (maybe missing), try uv if available
            if has_uv:
                cmd = ["uv", "pip", "install", "--python", py, "--quiet", pkg]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode == 0:
                    results[pkg] = True
                    continue
                    
            logger.warning(
                "env_manager: install %r failed (rc=%d): %s",
                pkg, proc.returncode, proc.stderr[:200],
            )
            results[pkg] = False
        except Exception as exc:
            logger.warning("env_manager: install %r raised %s", pkg, exc)
            results[pkg] = False
    return results


def update_requirements(project_root: str, new_packages: List[str]) -> None:
    """Append successfully-installed packages to requirements.txt.

    Only adds packages that aren't already listed there, so repeated
    runs are idempotent.
    """
    if not new_packages:
        return
    req_path = Path(project_root) / "requirements.txt"
    existing_text = req_path.read_text(encoding="utf-8") if req_path.exists() else ""
    existing_names = _read_requirements(project_root)
    to_add = [
        p for p in new_packages
        if p.lower().replace("-", "_") not in existing_names
    ]
    if not to_add:
        return
    tail = "\n" if existing_text and not existing_text.endswith("\n") else ""
    req_path.write_text(
        existing_text + tail + "\n".join(to_add) + "\n",
        encoding="utf-8",
    )
    logger.info("env_manager: added %d package(s) to requirements.txt: %s",
                len(to_add), to_add)


def remove_from_requirements(project_root: str,
                             packages: List[str]) -> List[str]:
    """Drop ``packages`` from requirements.txt -- the symmetric counterpart
    to :func:`update_requirements` (P1.4).

    Removes only the lines whose distribution name matches one of
    ``packages`` (normalised case-/dash-insensitively, mirroring
    :func:`_read_requirements`); comments, ``-r``/``-c`` includes, blank
    lines, and the version specifiers on kept lines are preserved verbatim.
    Idempotent: a package already absent is a no-op, so repeated runs are
    safe. Returns the distribution names actually removed.
    """
    if not packages:
        return []
    req_path = Path(project_root) / "requirements.txt"
    if not req_path.exists():
        return []
    targets = {p.lower().replace("-", "_") for p in packages}
    kept: List[str] = []
    removed: List[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#")[0].strip()
        if stripped and not stripped.startswith("-"):
            name = re.split(r"[>=<!;\[]", stripped)[0].strip() \
                .lower().replace("-", "_")
            if name in targets:
                removed.append(name)
                continue
        kept.append(line)
    if not removed:
        return []
    text = "\n".join(kept)
    if text and not text.endswith("\n"):
        text += "\n"
    req_path.write_text(text, encoding="utf-8")
    logger.info("env_manager: removed %d package(s) from requirements.txt: %s",
                len(removed), removed)
    return removed


# Marker recording that requirements.txt on disk is env-managed -- a
# repair re-pinned it to a self-consistent, conflict-free set, so a later
# whole-tree regenerate must carry it forward verbatim instead of
# re-emitting the model's stale manifest. Kept under the session's hidden
# ``.cgx`` dir so it never pollutes the generated project tree.
_REQUIREMENTS_LOCK_MARKER = os.path.join(".cgx", "requirements.locked")


def mark_requirements_locked(project_root: str) -> None:
    """Record that requirements.txt is env-managed (repair-resolved).

    A whole-tree regenerate reads this marker (:func:`requirements_locked`)
    and carries the resolved requirements.txt forward verbatim instead of
    re-generating it from the model, so a deterministic dependency fix
    survives a re-scaffold. Best-effort: a write failure is logged and
    swallowed so it never breaks the repair that resolved the conflict.
    """
    try:
        marker = Path(project_root) / _REQUIREMENTS_LOCK_MARKER
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort marker
        logger.warning(
            "env_manager: could not write requirements lock marker: %s", exc)


def requirements_locked(project_root: str) -> bool:
    """Return True when a repair marked requirements.txt env-managed."""
    try:
        return (Path(project_root) / _REQUIREMENTS_LOCK_MARKER).is_file()
    except Exception:  # pragma: no cover - defensive
        return False


@traced("codegen")
def preflight_install(
    generated_files: List[str],
    project_root: str,
    python: Optional[str] = None,
) -> Tuple[List[str], Dict[str, bool]]:
    """Scan generated files for imports, install any missing packages.

    Returns ``(missing_found, install_results)`` so the caller can decide
    whether to update requirements.txt after tests pass.

    Only Python sources are scanned: this step installs into the project
    venv via ``pip``, so JS/TS files are excluded to keep npm package
    names (which ``pip`` cannot satisfy) out of the installer entirely.
    """
    py_files = [f for f in generated_files
                if str(f).lower().endswith(".py")]
    imports = scan_imports(py_files)
    missing = find_missing_python_packages(imports, project_root, python=python)
    if not missing:
        return [], {}
    logger.info(
        "env_manager: %d missing package(s) detected: %s", len(missing), missing
    )
    results = install_packages(missing, python=python)
    return missing, results


# Leading distribution-name token of a requirements line: the run of name
# characters before any version specifier / extras / marker. Spelled out
# (no ``\w``) for cross-engine portability with the rest of the codebase.
_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _requirement_name(line: str) -> Optional[str]:
    """Return the raw distribution name a requirements line declares.

    Strips an inline comment and surrounding whitespace, then reads the
    leading name token (before any ``>=<!~;[`` specifier / extras / marker).
    Returns ``None`` for blank lines, comments, and pip flags
    (``-r`` / ``-e`` / ``--hash`` ...), which carry no package to re-pin.
    """
    text = line.split("#", 1)[0].strip()
    if not text or text.startswith("-"):
        return None
    m = _REQ_NAME_RE.match(text)
    return m.group(1) if m else None


def _import_root_to_pypi(root: str) -> str:
    """Map a top-level import root to its pip distribution name."""
    key = (root or "").strip()
    if not key:
        return ""
    if key in _IMPORT_TO_PYPI:
        return _IMPORT_TO_PYPI[key]
    return key.replace("_", "-")


def _pip_freeze_versions(python: Optional[str] = None) -> Dict[str, str]:
    """Return ``{normalised distribution name: installed version}``.

    Parses ``pip freeze`` output; editable / URL installs (which carry no
    ``==`` version) are skipped. Best-effort: any failure yields an empty
    map so the caller leaves the manifest untouched.
    """
    py = python or sys.executable
    out: Dict[str, str] = {}
    try:
        proc = subprocess.run(
            [py, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0 and _has_uv():
            proc = subprocess.run(
                ["uv", "pip", "freeze", "--python", py],
                capture_output=True, text=True, timeout=120,
            )
    except Exception as exc:
        logger.warning("env_manager: freeze raised %s", exc)
        return out
    if proc.returncode != 0:
        return out
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("-") or "==" not in line or " @ " in line:
            continue
        name, _, version = line.partition("==")
        key = name.strip().lower().replace("-", "_")
        version = version.strip()
        if key and version:
            out[key] = version
    return out


def _repin_requirements(text: str, installed: Dict[str, str]) -> str:
    """Rewrite each declared requirement to the version actually installed.

    For every line that declares a package present in ``installed``, replace
    whatever specifier the model wrote with an exact ``==`` pin on the
    resolved version -- so the manifest is reproducible *and* internally
    consistent after a conflict re-resolve. Comments, blank lines, pip
    flags, and packages with no installed version are preserved verbatim.
    """
    lines_out: List[str] = []
    for raw in text.splitlines():
        name = _requirement_name(raw)
        if name is None:
            lines_out.append(raw)
            continue
        version = installed.get(name.lower().replace("-", "_"))
        if not version:
            lines_out.append(raw)
            continue
        comment = ("  #" + raw.split("#", 1)[1]) if "#" in raw else ""
        lines_out.append(f"{name}=={version}{comment}".rstrip())
    result = "\n".join(lines_out)
    return result + "\n" if text.endswith("\n") else result


@traced("codegen")
def resolve_dependency_conflict(
    project_root: str,
    packages: List[str],
    python: Optional[str] = None,
) -> Dict[str, object]:
    """Re-resolve a transitive dependency conflict, then re-pin reproducibly.

    ``packages`` are the top-level import roots API_CHECK found implicated
    in an import-time version conflict -- the consumer whose stale pin is
    too old (``flask``) and the peer whose incompatible major got resolved
    in (``werkzeug``). Force-upgrades those distributions so pip moves off
    the satisfied-but-broken pins and resolves a self-consistent set, then
    re-pins every *declared* requirement to the version actually installed
    (:func:`_repin_requirements`). The result is a requirements.txt that
    both installs cleanly and stays reproducible -- never left unpinned.

    Non-fatal by contract: any pip failure is logged and reflected in the
    returned summary; BOOTSTRAP_ENV re-probes via API_CHECK regardless, and
    the shared repair budget halts a conflict that cannot be resolved.
    """
    py = python or sys.executable
    dists: List[str] = []
    for root in packages:
        name = _import_root_to_pypi(str(root))
        if name and name not in dists:
            dists.append(name)
    summary: Dict[str, object] = {
        "packages": list(dists), "upgraded": False, "repinned": []}
    if not dists:
        return summary
    try:
        proc = subprocess.run(
            [py, "-m", "pip", "install", "--upgrade", "--no-input", *dists],
            capture_output=True, text=True, timeout=300,
        )
    except Exception as exc:
        logger.warning("env_manager: conflict re-resolve raised %s", exc)
        return summary
    if proc.returncode != 0:
        logger.warning(
            "env_manager: conflict re-resolve upgrade failed (rc=%d): %s",
            proc.returncode, (proc.stderr or "")[:300])
        return summary
    summary["upgraded"] = True
    req_path = Path(project_root) / "requirements.txt"
    if not req_path.is_file():
        return summary
    installed = _pip_freeze_versions(py)
    if not installed:
        return summary
    before = req_path.read_text(encoding="utf-8", errors="ignore")
    after = _repin_requirements(before, installed)
    if after != before:
        req_path.write_text(after, encoding="utf-8")
        summary["repinned"] = sorted(
            n for n in (_requirement_name(ln) for ln in after.splitlines())
            if n)
        # Mark the file env-managed so a later whole-tree regenerate
        # carries these resolved pins forward instead of re-emitting the
        # model's stale manifest and reintroducing the conflict.
        mark_requirements_locked(project_root)
        logger.info(
            "env_manager: re-pinned requirements.txt after conflict "
            "re-resolve (%s)", ", ".join(dists))
    return summary
