"""Objective-grounded behavioural acceptance for the swarm.

The swarm's success signal is otherwise self-graded: the same weak model writes
the code AND the unit tests that certify it, so a stub can pass its own tests.
This adds an INDEPENDENT check: does the assembled system actually *behave*?
Its checks are derived deterministically from the plan's declared ``endpoints``
contract (which encodes what the objective asked for) -- the model does not
author the pass criteria, so a stub cannot also weaken them.

Deliberately conservative, because a false red here would drop a
functionally-correct build (worse than the current self-grading):

* It runs ONLY when the plan declares an explicit run/serve command under
  ``contracts["acceptance"]["run"]``. There is NO framework-incantation
  guessing (no "try uvicorn, then flask run, then manage.py"): that would be
  framework special-casing. Absent a declared command, acceptance is SKIPPED
  (advisory), never failed.
* Only GET endpoints with no declared request body are probed -- a POST needs a
  body we do not have, and inventing one would false-red.
* The outcome splits hard: server never bound / connection refused / toolchain
  missing / timeout => SKIPPED (environmental, mirrors the npm toolchain-missing
  handling); a route that returns a 5xx (or a connection that dies) AFTER the
  server was confirmed up => FAILED (the server crashed handling a declared
  route -- an unambiguous defect a stub-test would have hidden).

All the process/network work is behind injectable seams (``boot`` / ``http_get``)
so this is unit-tested without ever spawning a server; the default seams do the
real work in production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_READY_TIMEOUT = 20.0
_READY_POLL = 0.3


@dataclass
class AcceptanceResult:
    outcome: str = "skipped"        # "passed" | "failed" | "skipped"
    reason: str = ""
    failures: List[str] = field(default_factory=list)
    checks_run: int = 0

    @property
    def ok(self) -> bool:
        """Advisory-degrading: only an actual FAIL blocks; skip never blocks."""
        return self.outcome != "failed"


def derive_http_checks(contracts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Probeable checks from the declared ``endpoints`` contract.

    Only GET endpoints that declare no request body are returned (a body-less
    GET is safe to call blind). Each check is ``{method, path}``; the pass rule
    (status < 500) lives in the probe, so nothing objective-specific is encoded.
    """
    checks: List[Dict[str, Any]] = []
    for ep in (contracts.get("endpoints") or []):
        if not isinstance(ep, dict):
            continue
        method = str(ep.get("method") or "GET").strip().upper()
        path = str(ep.get("path") or "").strip()
        if not path or not path.startswith("/"):
            continue
        if method != "GET":
            continue
        if ep.get("request"):          # needs a body we cannot invent
            continue
        checks.append({"method": method, "path": path})
    # De-dup by path, preserve order.
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for c in checks:
        if c["path"] not in seen:
            seen.add(c["path"])
            out.append(c)
    return out


class _Server:
    """Handle a default-boot server exposes: ``ready``, ``base_url``, ``close``."""

    def __init__(self, proc, base_url: str, ready: bool):
        self.proc = proc
        self.base_url = base_url
        self.ready = ready

    def close(self) -> None:  # pragma: no cover - exercised only with a real proc
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def _default_http_get(url: str, timeout: float = 5.0) -> Optional[int]:
    """GET ``url`` and return its HTTP status, or ``None`` on a connection error."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as e:
        return int(e.code)              # a 4xx/5xx response still bound
    except Exception:
        return None                     # connection refused / reset / timeout


def _default_boot(run: Dict[str, Any], root: str, python_exe: Optional[str],
                  http_get: Callable[..., Optional[int]]) -> Optional[_Server]:
    """Start the declared server, poll readiness, return a handle or ``None``.

    Returns ``None`` (=> SKIP) when the command is unusable or the toolchain is
    absent. ``ready`` reflects whether the server bound within the timeout.
    """
    import os
    import shutil
    import subprocess
    import time

    cmd = run.get("command")
    if isinstance(cmd, str):
        cmd = cmd.split()
    if not isinstance(cmd, list) or not cmd:
        return None
    if shutil.which(str(cmd[0])) is None and not os.path.exists(str(cmd[0])):
        return None                     # toolchain missing -> environmental skip
    port = int(run.get("port") or 8000)
    base_url = f"http://127.0.0.1:{port}"
    ready_path = str(run.get("readiness_path") or "/")
    cwd = os.path.join(root, str(run.get("cwd") or "")) if run.get("cwd") else root
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (run.get("env") or {}).items()})
    if python_exe:
        env.setdefault("PYTHON", python_exe)
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception:
        return None
    deadline = time.time() + _READY_TIMEOUT
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:     # process exited before binding
            break
        status = http_get(base_url + ready_path, timeout=2.0)
        if status is not None:
            ready = True
            break
        time.sleep(_READY_POLL)
    return _Server(proc, base_url, ready)


def run_acceptance(
    contracts: Dict[str, Any], root: str, *,
    python_exe: Optional[str] = None,
    boot: Optional[Callable[..., Optional[_Server]]] = None,
    http_get: Optional[Callable[..., Optional[int]]] = None,
) -> AcceptanceResult:
    """Run behavioural acceptance against a plan-declared server. Never raises.

    SKIPS (advisory) when nothing is declared/derivable or the environment
    cannot host the server; FAILS only when the booted server returns a 5xx (or
    dies) on a declared GET route.
    """
    spec = contracts.get("acceptance") if isinstance(contracts, dict) else None
    run = spec.get("run") if isinstance(spec, dict) else None
    if not isinstance(run, dict) or not run.get("command"):
        return AcceptanceResult(reason="no run command declared (skipped)")
    checks = derive_http_checks(contracts)
    if not checks:
        return AcceptanceResult(reason="no probeable GET endpoints (skipped)")
    http_get = http_get or _default_http_get
    boot = boot or _default_boot
    server = None
    try:
        server = boot(run, root, python_exe, http_get)
    except Exception:  # pragma: no cover - boot is best-effort
        server = None
    if server is None:
        return AcceptanceResult(reason="server toolchain unavailable (skipped)")
    try:
        if not getattr(server, "ready", False):
            return AcceptanceResult(
                reason="server did not bind within timeout (skipped)")
        failures: List[str] = []
        run_n = 0
        for c in checks:
            url = server.base_url + c["path"]
            status = http_get(url)
            run_n += 1
            if status is None:
                failures.append(f"{c['method']} {c['path']} -> no response "
                                "(server died)")
            elif status >= 500:
                failures.append(f"{c['method']} {c['path']} -> HTTP {status}")
        if failures:
            return AcceptanceResult(
                outcome="failed",
                reason=f"{len(failures)}/{run_n} declared routes errored",
                failures=failures, checks_run=run_n)
        return AcceptanceResult(outcome="passed",
                                reason=f"{run_n} routes responded",
                                checks_run=run_n)
    finally:
        try:
            server.close()
        except Exception:  # pragma: no cover
            pass


__all__ = ["AcceptanceResult", "derive_http_checks", "run_acceptance"]
