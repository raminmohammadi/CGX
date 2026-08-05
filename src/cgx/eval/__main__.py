

"""CLI entry point for the CGX evaluation gate.

Usage::

    python -m cgx.eval [--evals-dir evals] [--json]

Exits non-zero when any threshold that actually ran is not met, so it can be
dropped straight into CI as a release gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from cgx.eval.harness import run_gate


def _default_evals_dir() -> str:
    # Repo layout: <root>/src/cgx/eval/__main__.py -> <root>/evals
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "evals")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cgx.eval", description=__doc__)
    parser.add_argument("--evals-dir", default=None, help="golden dataset directory")
    parser.add_argument("--json", action="store_true", help="emit the full JSON report")
    args = parser.parse_args(argv)

    evals_dir = args.evals_dir or _default_evals_dir()
    report, ok = run_gate(evals_dir)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for name, section in report["sections"].items():
            agg = section.get("aggregate") if isinstance(section, dict) else None
            if agg:
                pretty = ", ".join(f"{k}={v:.3f}" for k, v in sorted(agg.items()))
                print(f"[{name}] {pretty}")
            else:
                print(f"[{name}] {section}")
        for f in report["failures"]:
            print(f"FAIL: {f}")
        print("RESULT:", "PASS" if ok else "FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
