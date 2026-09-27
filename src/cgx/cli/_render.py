"""Shared output + error helpers for the parity subcommands.

Every ``cgx`` subcommand added for UI parity emits **JSON by default**
(scriptable, matches the existing ``index``/``query`` commands) and can opt
into a compact human table with ``--table``. Errors map to a nonzero exit
code via :func:`die` so the CLI composes cleanly in shell pipelines.
"""

from __future__ import annotations

import json
import sys
from typing import Any, NoReturn, Sequence


def print_json(obj: Any) -> None:
    """Print ``obj`` as indented JSON (``default=str`` for non-serializables)."""
    print(json.dumps(obj, indent=2, default=str))


def print_table(rows: Sequence[dict], columns: Sequence[tuple[str, str]]) -> None:
    """Print ``rows`` as an aligned text table.

    ``columns`` is a sequence of ``(header, key)`` pairs; each row is a dict.
    Falls back to an empty-state line when there are no rows.
    """
    if not rows:
        print("(none)")
        return
    headers = [h for h, _ in columns]
    keys = [k for _, k in columns]
    cells = [[_fmt(r.get(k, "")) for k in keys] for r in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in cells:
        print(fmt.format(*row))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def emit(obj: Any, *, table: bool = False,
         columns: Sequence[tuple[str, str]] | None = None) -> None:
    """Emit ``obj`` as a table (when ``table`` and ``columns`` given) or JSON."""
    if table and columns is not None and isinstance(obj, list):
        print_table(obj, columns)
    else:
        print_json(obj)


def die(msg: str, code: int = 1) -> NoReturn:
    """Print an error to stderr and exit with ``code`` (default 1)."""
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def read_text_input(path: str | None) -> str:
    """Read source text from ``path`` or, when ``path`` is ``-``/None, stdin."""
    if path and path != "-":
        try:
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        except OSError as exc:
            die(f"cannot read {path!r}: {exc}")
    data = sys.stdin.read()
    if not data.strip():
        die("no source provided (pass --file PATH or pipe text on stdin)")
    return data
