import { useState } from "react";
import { FileText, FolderOpen, Save, Trash2 } from "lucide-react";
import { CardHeader } from "../Card";
import { Field, TextArea } from "../Input";
import { Pill } from "../Pill";
import { api } from "../../lib/api";

const PLACEHOLDER = `# Project context (CGX.md)

Anything here is given to the chatbot and the agents on every turn for this
repo — no need to repeat it in each question.

## Conventions
- ...

## Domain glossary
- ...

## Always / never
- Never ...
`;

// Editor for the repo-level CGX.md context file (the CLAUDE.md analogue).
// Keyed by a project root the user types in; loads whichever candidate file
// exists (CGX.md / AGENT.md / .cgx/CGX.md) and writes back to it.
export function ContextFileTab({ defaultProjectRoot = "" }: { defaultProjectRoot?: string }) {
  const [projectRoot, setProjectRoot] = useState(defaultProjectRoot);
  const [content, setContent] = useState("");
  const [meta, setMeta] = useState<{ exists: boolean; path: string; filename: string } | null>(
    null,
  );
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    if (!projectRoot.trim()) {
      setError("Enter a project path first.");
      return;
    }
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const cf = await api.getContextFile(projectRoot.trim());
      setContent(cf.content);
      setMeta({ exists: cf.exists, path: cf.path, filename: cf.filename });
      setStatus(cf.exists ? `Loaded ${cf.filename}` : "No context file yet — start writing one.");
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    if (!projectRoot.trim()) {
      setError("Enter a project path first.");
      return;
    }
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const r = await api.writeContextFile(projectRoot.trim(), content);
      setMeta({ exists: true, path: r.path, filename: r.filename });
      setStatus(`Saved ${r.filename} (${r.bytes} bytes)`);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!projectRoot.trim() || !meta?.exists) return;
    if (!confirm(`Delete ${meta.filename}?`)) return;
    setBusy(true);
    setError(null);
    try {
      await api.deleteContextFile(projectRoot.trim());
      setContent("");
      setMeta({ exists: false, path: meta.path, filename: meta.filename });
      setStatus("Deleted.");
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="p-6 space-y-4 overflow-y-auto h-full max-w-4xl">
      <CardHeader
        title="Context File (CGX.md)"
        description="A repo-level instruction file — CGX's CLAUDE.md. Its contents are injected into the chatbot and agents on every turn for this project. Commit it to share house rules with your team."
      />

      <div className="flex items-end gap-2">
        <div className="flex-1">
          <Field label="Project path">
            <input
              value={projectRoot}
              onChange={(e) => setProjectRoot(e.target.value)}
              placeholder="/path/to/your/repo"
              spellCheck={false}
              className="av-input w-full font-mono text-xs"
            />
          </Field>
        </div>
        <button className="av-btn-ghost" onClick={load} disabled={busy}>
          <FolderOpen className="h-3.5 w-3.5" /> Load
        </button>
      </div>

      {meta && (
        <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
          <FileText className="h-3.5 w-3.5" />
          <span className="break-all">{meta.path}</span>
          <Pill tone={meta.exists ? "neon" : "slate"}>{meta.exists ? "exists" : "new"}</Pill>
        </div>
      )}

      <Field label="Content">
        <TextArea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder={PLACEHOLDER}
          spellCheck={false}
          rows={22}
          className="font-mono text-[11px] leading-relaxed"
        />
      </Field>

      {error && <p className="text-xs text-red-300 font-mono whitespace-pre-wrap">{error}</p>}
      {status && <p className="text-xs text-emerald-300 font-mono">{status}</p>}

      <div className="flex justify-end gap-2">
        {meta?.exists && (
          <button
            className="av-btn py-1.5 px-3 text-xs bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20"
            onClick={remove}
            disabled={busy}
          >
            <Trash2 className="h-3.5 w-3.5" /> Delete
          </button>
        )}
        <button className="av-btn-primary" onClick={save} disabled={busy}>
          <Save className="h-3.5 w-3.5" /> {busy ? "Working…" : "Save"}
        </button>
      </div>
    </div>
  );
}
