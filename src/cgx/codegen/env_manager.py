

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
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import logging

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

# Top-level names that are themselves namespace packages — pip-installing
# the bare root is meaningless (no such PyPI distribution), so when a
# dotted variant like ``google.generativeai`` is also seen in the source
# the bare root is dropped before the missing-package probe.
_NAMESPACE_ROOTS: frozenset = frozenset({"google", "azure"})


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
    name — the bare namespace root by itself isn't pip-installable.
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
    """Return npm package names from JS/TS import/require calls."""
    roots: Set[str] = set()
    for m in re.finditer(
        r"""(?:import|require)\s*[\(]?\s*['"]([^'"./][^'"]*?)['"]""", source
    ):
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


def _is_local_package(name: str, project_root: str) -> bool:
    """Return True when ``name`` matches a first-party file or directory.

    Covers both flat layouts (``<root>/<name>/`` or ``<root>/<name>.py``)
    and src-layouts (``<root>/src/<name>/`` or ``<root>/src/<name>.py``),
    including namespace packages without an ``__init__.py``. Used to
    avoid mistaking the project's own top-level folder (e.g. ``backend``)
    for a missing PyPI distribution.
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
    return False


def find_missing_python_packages(
    imports: Set[str],
    project_root: str,
) -> List[str]:
    """Return Python package names that are imported but not declared.

    Filters out stdlib modules, packages already declared in
    ``requirements.txt``, first-party project directories, and packages
    that are already importable in the current environment so we don't
    re-install them. Import names that differ from their PyPI
    distribution name (``google.generativeai`` → ``google-generativeai``,
    ``PIL`` → ``Pillow``, …) are translated via :data:`_IMPORT_TO_PYPI`
    before being reported as missing.
    """
    declared = _read_requirements(project_root)
    # Drop bare namespace roots when a dotted variant is also present:
    # ``import google.generativeai`` records both ``google`` and
    # ``google.generativeai`` and we only want to install the latter.
    dotted_roots = {n.split(".", 1)[0] for n in imports if "." in n}
    pruned = {
        n for n in imports
        if not (n in _NAMESPACE_ROOTS and n in dotted_roots)
    }
    missing: List[str] = []
    seen_pypi: Set[str] = set()
    for name in sorted(pruned):
        root = name.split(".")[0]
        # stdlib check operates on the root regardless of dotted form.
        if root.lower().replace("-", "_") in _STDLIB_TOP:
            continue
        # Resolve to the PyPI distribution name. Dotted names without a
        # mapping aren't installable as-is (pip can't install
        # ``google.generativeai`` literally) — skip them; the matching
        # root entry will have been handled separately.
        if name in _IMPORT_TO_PYPI:
            pypi_name = _IMPORT_TO_PYPI[name]
        elif "." in name:
            continue
        else:
            pypi_name = name
        normalized = pypi_name.lower().replace("-", "_")
        if normalized in declared:
            continue
        # Skip first-party project packages — the project's own top-level
        # folder is not a PyPI distribution and pip cannot install it.
        # Only meaningful for bare root names.
        if "." not in name and _is_local_package(name, project_root):
            continue
        # Check if already importable (covers editable installs, etc.)
        try:
            __import__(name)
            continue
        except ImportError:
            pass
        except Exception:
            # Some packages have side effects on import; skip the probe.
            continue
        if pypi_name in seen_pypi:
            continue
        seen_pypi.add(pypi_name)
        missing.append(pypi_name)
    return missing


def install_packages(
    packages: List[str],
    python: Optional[str] = None,
) -> Dict[str, bool]:
    """pip-install each package; returns {name: success}.

    ``python`` is the interpreter path (defaults to the running one).
    This is designed to install into the SANDBOX's Python environment.
    """
    if not packages:
        return {}
    py = python or sys.executable
    results: Dict[str, bool] = {}
    for pkg in packages:
        logger.info("env_manager: installing missing package %r", pkg)
        try:
            proc = subprocess.run(
                [py, "-m", "pip", "install", "--quiet", "--no-input", pkg],
                capture_output=True, text=True, timeout=120,
            )
            ok = proc.returncode == 0
            results[pkg] = ok
            if not ok:
                logger.warning(
                    "env_manager: pip install %r failed (rc=%d): %s",
                    pkg, proc.returncode, proc.stderr[:200],
                )
        except Exception as exc:
            logger.warning("env_manager: pip install %r raised %s", pkg, exc)
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


def preflight_install(
    generated_files: List[str],
    project_root: str,
    python: Optional[str] = None,
) -> Tuple[List[str], Dict[str, bool]]:
    """Scan generated files for imports, install any missing packages.

    Returns ``(missing_found, install_results)`` so the caller can decide
    whether to update requirements.txt after tests pass.
    """
    imports = scan_imports(generated_files)
    missing = find_missing_python_packages(imports, project_root)
    if not missing:
        return [], {}
    logger.info(
        "env_manager: %d missing package(s) detected: %s", len(missing), missing
    )
    results = install_packages(missing, python=python)
    return missing, results
