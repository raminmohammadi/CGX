"""Objective-grounded behavioural acceptance (cgx.session.tasks.acceptance).

Uses injected boot/http seams so no real server is ever spawned. The contract
under test: checks come from declared endpoints (not the model); it FAILS only
on a 5xx / dead connection after the server was up; every environmental case
(no run command, no endpoints, no bind, toolchain missing) is an advisory SKIP.
"""

from cgx.session.tasks import acceptance as ac


# ---------------------- check derivation ----------------------

def test_derive_http_checks_only_bodyless_get():
    contracts = {"endpoints": [
        {"method": "GET", "path": "/"},
        {"method": "GET", "path": "/health"},
        {"method": "POST", "path": "/chat"},               # not GET
        {"method": "GET", "path": "/echo", "request": {"msg": "str"}},  # has body
        {"method": "GET", "path": "relative"},             # not absolute
        {"method": "GET", "path": "/"},                    # dup
    ]}
    checks = ac.derive_http_checks(contracts)
    assert [c["path"] for c in checks] == ["/", "/health"]


# ---------------------- run_acceptance skips ----------------------

_RUN = {"acceptance": {"run": {"command": ["python", "app.py"]}},
        "endpoints": [{"method": "GET", "path": "/"}]}


class _FakeServer:
    def __init__(self, ready=True, base_url="http://127.0.0.1:8000"):
        self.ready = ready
        self.base_url = base_url
        self.closed = False

    def close(self):
        self.closed = True


def test_skips_without_run_command():
    r = ac.run_acceptance({"endpoints": [{"method": "GET", "path": "/"}]}, ".",
                          boot=lambda *a, **k: _FakeServer(),
                          http_get=lambda *a, **k: 200)
    assert r.outcome == "skipped" and r.ok


def test_skips_without_probeable_endpoints():
    r = ac.run_acceptance({"acceptance": {"run": {"command": ["x"]}},
                           "endpoints": [{"method": "POST", "path": "/x"}]}, ".",
                          boot=lambda *a, **k: _FakeServer(),
                          http_get=lambda *a, **k: 200)
    assert r.outcome == "skipped"


def test_skips_when_toolchain_missing():
    r = ac.run_acceptance(_RUN, ".", boot=lambda *a, **k: None,
                          http_get=lambda *a, **k: 200)
    assert r.outcome == "skipped" and "toolchain" in r.reason


def test_skips_when_server_never_binds():
    srv = _FakeServer(ready=False)
    r = ac.run_acceptance(_RUN, ".", boot=lambda *a, **k: srv,
                          http_get=lambda *a, **k: 200)
    assert r.outcome == "skipped" and "bind" in r.reason
    assert srv.closed          # server still torn down


# ---------------------- run_acceptance verdicts ----------------------

def test_passes_when_routes_respond():
    srv = _FakeServer()
    r = ac.run_acceptance(_RUN, ".", boot=lambda *a, **k: srv,
                          http_get=lambda url, timeout=5.0: 200)
    assert r.outcome == "passed" and r.ok and r.checks_run == 1
    assert srv.closed


def test_fails_on_server_error_status():
    r = ac.run_acceptance(_RUN, ".", boot=lambda *a, **k: _FakeServer(),
                          http_get=lambda url, timeout=5.0: 500)
    assert r.outcome == "failed" and not r.ok
    assert any("500" in f for f in r.failures)


def test_4xx_does_not_fail():
    # A 4xx (e.g. auth/validation) is not a server crash -> not a failure.
    r = ac.run_acceptance(_RUN, ".", boot=lambda *a, **k: _FakeServer(),
                          http_get=lambda url, timeout=5.0: 404)
    assert r.outcome == "passed"


def test_dead_connection_after_ready_is_failure():
    # Ready probe used a different call; the check itself gets no response.
    calls = {"n": 0}

    def http_get(url, timeout=5.0):
        calls["n"] += 1
        return None            # every call: no response
    srv = _FakeServer(ready=True)   # readiness asserted by the fake, not http_get
    r = ac.run_acceptance(_RUN, ".", boot=lambda *a, **k: srv, http_get=http_get)
    assert r.outcome == "failed"
    assert any("no response" in f for f in r.failures)


def test_never_raises_on_bad_boot():
    def boom(*a, **k):
        raise RuntimeError("boot exploded")
    r = ac.run_acceptance(_RUN, ".", boot=boom, http_get=lambda *a, **k: 200)
    assert r.outcome == "skipped"        # swallowed -> advisory skip
