import json
import sqlite3
import sys

db = sys.argv[1]
con = sqlite3.connect(db)
con.row_factory = sqlite3.Row

tabs = [r[0] for r in con.execute(
    "select name from sqlite_master where type='table'")]

rows = list(con.execute("select * from artifacts order by rowid"))
print("KINDS:", [str(dict(r).get("kind")) for r in rows])

# task status overview
tcols = [r[1] for r in con.execute("pragma table_info(tasks)")]
print("TASK COLS:", tcols)
for r in con.execute("select * from tasks order by rowid"):
    d = dict(r)
    raw = d.get("data_json")
    try:
        t = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        t = {}
    print("TASK", t.get("kind"), "|", t.get("status"), "|", t.get("name"),
          "|", (t.get("failure") or "")[:80])

for r in rows:
    d = dict(r)
    kind = str(d.get("kind", "")).lower()
    raw = d.get("data_json")
    try:
        outer = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        outer = {}
    c = outer.get("content", outer) if isinstance(outer, dict) else {}
    print("\n==== ", kind, " ====")
    if "work_plan" in kind:
        for lay in (c.get("layers") or []):
            files = [f.get("path") for f in (lay.get("files") or [])]
            print("  layer", lay.get("name"), "->", files)
    elif "scaffold_patches" in kind:
        print("  generated:",
              [g.get("file") for g in (c.get("generated") or [])])
        print("  failed:", c.get("failed"))
    elif "applied_changes" in kind:
        print("  applied:", c.get("applied_files"))
        print("  failed:", c.get("failed_files"))
    elif "verify_report" in kind:
        print("  outcome:", c.get("outcome"),
              "skipped_reason:", c.get("skipped_reason"),
              "returncode:", c.get("returncode"),
              "tests_selected:", c.get("tests_selected"))
    elif "repair_plan" in kind:
        print("  classification:", c.get("classification"),
              "strategy:", c.get("strategy"),
              "can_apply diffs:", len(c.get("diffs") or []))
        print("  rationale:", str(c.get("rationale"))[:200])
    elif "api_check_report" in kind or "smoke_report" in kind:
        print("  outcome:", c.get("outcome"),
              "missing_modules:", c.get("missing_modules"),
              "failed_modules:", c.get("failed_modules"))
    elif "build_report" in kind:
        print("  outcome:", c.get("outcome"),
              "venv_path:", c.get("venv_path"))
