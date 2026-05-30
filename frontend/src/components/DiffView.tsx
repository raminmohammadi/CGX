// Render a unified diff either as a side-by-side "before / after" view
// (default) or as a classic unified-diff list. Multi-file diffs are split
// on the "diff --git" or "--- a/" header so each gets its own framed block.

import { Fragment, useState } from "react";

interface DiffBlock {
  filename: string;
  lines: string[];
  isNew?: boolean;
  isDeleted?: boolean;
}

function splitDiff(diff: string): DiffBlock[] {
  const blocks: DiffBlock[] = [];
  const lines = diff.split("\n");
  let current: DiffBlock | null = null;
  let pendingHeader: string[] = [];
  const flush = () => { if (current) blocks.push(current); };
  for (const ln of lines) {
    if (ln.startsWith("diff --git ")) {
      flush();
      const parts = ln.split(" ");
      const file = (parts.at(-1) || "diff").replace(/^b\//, "");
      current = { filename: file, lines: [ln] };
      pendingHeader = [];
    } else if (ln.startsWith("--- ") || ln.startsWith("+++ ")) {
      if (!current) {
        pendingHeader.push(ln);
        if (ln.startsWith("+++ ")) {
          const file = ln.replace(/^\+\+\+ b?\//, "").trim();
          current = { filename: file || "diff", lines: [...pendingHeader] };
          pendingHeader = [];
        }
      } else {
        current.lines.push(ln);
      }
      if (current) {
        if (ln.startsWith("--- ") && ln.includes("/dev/null")) current.isNew = true;
        if (ln.startsWith("+++ ") && ln.includes("/dev/null")) current.isDeleted = true;
      }
    } else {
      if (!current) current = { filename: "diff", lines: [] };
      current.lines.push(ln);
    }
  }
  flush();
  return blocks;
}

interface SplitRow {
  leftNum?: number; leftText?: string; leftKind?: "ctx" | "del" | "hunk";
  rightNum?: number; rightText?: string; rightKind?: "ctx" | "add" | "hunk";
}

function buildSplitRows(blockLines: string[]): SplitRow[] {
  const rows: SplitRow[] = [];
  let oldNo = 1, newNo = 1;
  let i = 0;
  while (i < blockLines.length) {
    const ln = blockLines[i];
    const m = ln.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (m) {
      oldNo = parseInt(m[1], 10) || 1;
      newNo = parseInt(m[2], 10) || 1;
      rows.push({ leftKind: "hunk", rightKind: "hunk", leftText: ln, rightText: ln });
      i++;
      let delBuf: string[] = [];
      let addBuf: string[] = [];
      const flushPairs = () => {
        const max = Math.max(delBuf.length, addBuf.length);
        for (let k = 0; k < max; k++) {
          const dt = k < delBuf.length ? delBuf[k] : undefined;
          const at = k < addBuf.length ? addBuf[k] : undefined;
          rows.push({
            leftNum: dt !== undefined ? oldNo + k : undefined,
            leftText: dt, leftKind: dt !== undefined ? "del" : undefined,
            rightNum: at !== undefined ? newNo + k : undefined,
            rightText: at, rightKind: at !== undefined ? "add" : undefined,
          });
        }
        oldNo += delBuf.length; newNo += addBuf.length;
        delBuf = []; addBuf = [];
      };
      while (i < blockLines.length
             && !blockLines[i].startsWith("@@")
             && !blockLines[i].startsWith("diff --git ")) {
        const l = blockLines[i];
        if (l.startsWith("+") && !l.startsWith("+++")) addBuf.push(l.slice(1));
        else if (l.startsWith("-") && !l.startsWith("---")) delBuf.push(l.slice(1));
        else if (l.startsWith("\\")) { /* "\ No newline at end of file" */ }
        else {
          flushPairs();
          const text = l.startsWith(" ") ? l.slice(1) : l;
          rows.push({
            leftNum: oldNo, leftText: text, leftKind: "ctx",
            rightNum: newNo, rightText: text, rightKind: "ctx",
          });
          oldNo++; newNo++;
        }
        i++;
      }
      flushPairs();
    } else {
      i++;
    }
  }
  return rows;
}

const STRIPES: React.CSSProperties = {
  backgroundImage:
    "repeating-linear-gradient(135deg, rgba(148,163,184,0.06) 0 6px, transparent 6px 12px)",
};

function toneForKind(k?: string): string {
  if (k === "del") return "bg-red-950/30 text-red-200";
  if (k === "add") return "bg-emerald-950/30 text-emerald-200";
  if (k === "ctx") return "text-slate-300";
  if (k === "hunk") return "bg-purple-500/10 text-purple-300";
  return "";
}

function SplitPane({ block }: { block: DiffBlock }) {
  const rows = buildSplitRows(block.lines);
  if (!rows.length) {
    return <div className="px-3 py-2 text-[11px] font-mono text-slate-500 italic">(no hunks)</div>;
  }
  return (
    <div className="grid grid-cols-2 font-mono text-[11px] leading-relaxed">
      {rows.map((r, i) => (
        <Fragment key={i}>
          <div
            className={`flex border-r border-white/5 ${toneForKind(r.leftKind)}`}
            style={r.leftKind ? undefined : STRIPES}
          >
            <span className="w-10 shrink-0 text-right pr-2 py-0.5 select-none text-slate-600 border-r border-white/5">
              {r.leftNum ?? ""}
            </span>
            <span className="flex-1 px-2 py-0.5 whitespace-pre overflow-x-auto">
              {r.leftText ?? "\u00A0"}
            </span>
          </div>
          <div
            className={`flex ${toneForKind(r.rightKind)}`}
            style={r.rightKind ? undefined : STRIPES}
          >
            <span className="w-10 shrink-0 text-right pr-2 py-0.5 select-none text-slate-600 border-r border-white/5">
              {r.rightNum ?? ""}
            </span>
            <span className="flex-1 px-2 py-0.5 whitespace-pre overflow-x-auto">
              {r.rightText ?? "\u00A0"}
            </span>
          </div>
        </Fragment>
      ))}
    </div>
  );
}

function colorForUnifiedLine(ln: string): string {
  if (ln.startsWith("+++") || ln.startsWith("---")) return "text-slate-500";
  if (ln.startsWith("@@")) return "text-purple-300 bg-purple-500/5";
  if (ln.startsWith("+")) return "text-emerald-400 bg-emerald-950/20";
  if (ln.startsWith("-")) return "text-red-400 bg-red-950/20";
  if (ln.startsWith("diff --git")) return "text-slate-500";
  return "text-slate-400";
}

function UnifiedPane({ block }: { block: DiffBlock }) {
  return (
    <div className="font-mono text-[11px] leading-relaxed">
      {block.lines.map((ln, j) => (
        <div key={j} className={"px-3 py-0.5 " + colorForUnifiedLine(ln)}>
          {ln || "\u00A0"}
        </div>
      ))}
    </div>
  );
}

export function DiffView({
  diff,
  defaultMode = "split",
}: {
  diff: string;
  defaultMode?: "split" | "unified";
}) {
  const [mode, setMode] = useState<"split" | "unified">(defaultMode);
  const blocks = splitDiff(diff || "");
  if (!blocks.length || !diff.trim()) {
    return (
      <div className="text-xs text-slate-500 font-mono italic">
        (no diff produced)
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {blocks.map((b, i) => (
        <div
          key={i}
          className="rounded-xl border border-muted bg-slate-950 overflow-hidden"
        >
          <div className="bg-slate-900 px-4 py-2 flex items-center justify-between gap-3 text-[10px] text-slate-400 border-b border-white/5 font-mono">
            <span className="text-slate-300 truncate">
              {b.filename}
              {b.isNew && (
                <span className="ml-2 text-[9px] uppercase tracking-wider text-emerald-400">
                  new file
                </span>
              )}
              {b.isDeleted && (
                <span className="ml-2 text-[9px] uppercase tracking-wider text-red-400">
                  deleted
                </span>
              )}
            </span>
            <div className="flex items-center gap-1 shrink-0">
              <button
                type="button"
                onClick={() => setMode("split")}
                className={`px-2 py-0.5 rounded text-[9px] uppercase tracking-wider ${
                  mode === "split"
                    ? "bg-slate-700 text-slate-100"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                Split
              </button>
              <button
                type="button"
                onClick={() => setMode("unified")}
                className={`px-2 py-0.5 rounded text-[9px] uppercase tracking-wider ${
                  mode === "unified"
                    ? "bg-slate-700 text-slate-100"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                Unified
              </button>
            </div>
          </div>
          {mode === "split" ? <SplitPane block={b} /> : <UnifiedPane block={b} />}
        </div>
      ))}
    </div>
  );
}

