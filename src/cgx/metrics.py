"""In-process metrics registry with a Prometheus text exporter.

Stdlib-only (no ``prometheus_client`` dependency) so it works in the core
install. Collection is always-on and cheap -- a counter increment is a
dict lookup under a lock -- following the same zero-config philosophy as
the rest of CGX. The exposition format is the Prometheus text format
(``version=0.0.4``) rendered by :func:`render_prometheus`, scraped via the
``/api/metrics`` endpoint.

Three metric types are supported: counters (monotonic), gauges
(set/inc/dec), and histograms (bucketed observations with ``_sum`` /
``_count`` aggregates). Labels are passed as keyword arguments and keyed
by their sorted ``(name, value)`` tuple.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

# Default latency buckets in milliseconds -- covers sub-ms cache hits up to
# minute-long LLM generations without exploding cardinality.
DEFAULT_BUCKETS_MS: Tuple[float, ...] = (
    5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000,
)

LabelKey = Tuple[Tuple[str, str], ...]


def _labelkey(labels: Optional[Dict[str, object]]) -> LabelKey:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _fmt_labels(key: LabelKey, extra: Optional[Tuple[str, str]] = None) -> str:
    pairs = list(key)
    if extra is not None:
        pairs = pairs + [extra]
    if not pairs:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in pairs)
    return "{" + inner + "}"


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricsRegistry:
    """Thread-safe holder for counters, gauges, and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._help: Dict[str, str] = {}
        self._counters: Dict[str, Dict[LabelKey, float]] = {}
        self._gauges: Dict[str, Dict[LabelKey, float]] = {}
        # name -> {labelkey: {"buckets": [...], "counts": [...], "sum", "count"}}
        self._hist: Dict[str, Dict[LabelKey, Dict[str, object]]] = {}

    def inc(self, name: str, amount: float = 1.0, *,
            help: Optional[str] = None, **labels: object) -> None:
        key = _labelkey(labels)
        with self._lock:
            if help and name not in self._help:
                self._help[name] = help
            fam = self._counters.setdefault(name, {})
            fam[key] = fam.get(key, 0.0) + float(amount)

    def set_gauge(self, name: str, value: float, *,
                  help: Optional[str] = None, **labels: object) -> None:
        key = _labelkey(labels)
        with self._lock:
            if help and name not in self._help:
                self._help[name] = help
            self._gauges.setdefault(name, {})[key] = float(value)

    def observe(self, name: str, value: float, *,
                help: Optional[str] = None,
                buckets: Optional[Tuple[float, ...]] = None,
                **labels: object) -> None:
        key = _labelkey(labels)
        bkts = tuple(buckets) if buckets else DEFAULT_BUCKETS_MS
        v = float(value)
        with self._lock:
            if help and name not in self._help:
                self._help[name] = help
            fam = self._hist.setdefault(name, {})
            cell = fam.get(key)
            if cell is None:
                cell = {"buckets": bkts,
                        "counts": [0 for _ in bkts],
                        "sum": 0.0, "count": 0}
                fam[key] = cell
            counts: List[int] = cell["counts"]  # type: ignore[assignment]
            for i, b in enumerate(cell["buckets"]):  # type: ignore[arg-type]
                if v <= b:
                    counts[i] += 1
            cell["sum"] = float(cell["sum"]) + v  # type: ignore[arg-type]
            cell["count"] = int(cell["count"]) + 1  # type: ignore[arg-type]

    def render(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: List[str] = []
        with self._lock:
            for name in sorted(set(self._counters) | set(self._gauges) | set(self._hist)):
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                if name in self._counters:
                    lines.append(f"# TYPE {name} counter")
                    for key, val in sorted(self._counters[name].items()):
                        lines.append(f"{name}{_fmt_labels(key)} {_num(val)}")
                elif name in self._gauges:
                    lines.append(f"# TYPE {name} gauge")
                    for key, val in sorted(self._gauges[name].items()):
                        lines.append(f"{name}{_fmt_labels(key)} {_num(val)}")
                elif name in self._hist:
                    lines.append(f"# TYPE {name} histogram")
                    for key, cell in sorted(self._hist[name].items()):
                        # counts[i] already holds the cumulative number of
                        # observations <= buckets[i] (see ``observe``), so emit
                        # them directly -- do not re-accumulate.
                        for b, c in zip(cell["buckets"], cell["counts"], strict=False):  # type: ignore[arg-type]
                            le = "+Inf" if b == float("inf") else _num(float(b))
                            lines.append(
                                f"{name}_bucket{_fmt_labels(key, ('le', le))} {int(c)}")
                        lines.append(
                            f"{name}_bucket{_fmt_labels(key, ('le', '+Inf'))} "
                            f"{int(cell['count'])}")
                        lines.append(f"{name}_sum{_fmt_labels(key)} {_num(float(cell['sum']))}")
                        lines.append(f"{name}_count{_fmt_labels(key)} {int(cell['count'])}")
        return "\n".join(lines) + ("\n" if lines else "")

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._hist.clear()


def _num(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return repr(v)


# --- module-level default registry + convenience helpers ------------------
_REGISTRY = MetricsRegistry()


def registry() -> MetricsRegistry:
    return _REGISTRY


def inc(name: str, amount: float = 1.0, *, help: Optional[str] = None, **labels: object) -> None:
    _REGISTRY.inc(name, amount, help=help, **labels)


def set_gauge(name: str, value: float, *, help: Optional[str] = None, **labels: object) -> None:
    _REGISTRY.set_gauge(name, value, help=help, **labels)


def observe(name: str, value: float, *, help: Optional[str] = None,
            buckets: Optional[Tuple[float, ...]] = None, **labels: object) -> None:
    _REGISTRY.observe(name, value, help=help, buckets=buckets, **labels)


def render_prometheus() -> str:
    return _REGISTRY.render()


def reset_for_tests() -> None:
    _REGISTRY.reset()
