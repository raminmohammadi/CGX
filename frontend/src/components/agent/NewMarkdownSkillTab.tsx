import { useEffect, useState } from "react";
import { Save, ShieldCheck } from "lucide-react";
import { CardHeader } from "../Card";
import { Field, TextArea } from "../Input";
import { api, ApiError, type SkillSummary, type SkillValidationError } from "../../lib/api";

const DEFAULT_TEMPLATE = `---
name: my-skill
description: One line about when this applies.
# Words/phrases that turn the skill on (deterministic, case-insensitive).
triggers: [keyword, another phrase]
# Where the body is injected: chat, plan, scaffold, or all. Default: chat.
surfaces: [chat]
# Set true to always apply it (regardless of triggers).
always_on: false
---
Write your guidance here in plain Markdown. This text is added to the
model's instructions whenever the skill is active — describe domain rules,
house style, glossaries, "always do X / never do Y", etc.
`;

// Markdown (SKILL.md) authoring surface -- the friendly, code-free skill
// format. Unlike the Python editor, nothing here is executed: the document
// is parsed and validated statically (frontmatter fields, size, uniqueness).
export function NewMarkdownSkillTab({
  editing,
  onCreated,
}: {
  editing?: { name: string } | null;
  onCreated?: (skill: SkillSummary) => void;
}) {
  const [content, setContent] = useState(DEFAULT_TEMPLATE);
  const [loadingSource, setLoadingSource] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<SkillValidationError | null>(null);

  useEffect(() => {
    if (!editing) {
      setContent(DEFAULT_TEMPLATE);
      return;
    }
    setLoadingSource(true);
    setError(null);
    api
      .getSkillSource(editing.name)
      .then((r) => setContent(r.source))
      .catch((e) =>
        setError({ error_kind: "load_failed", error_detail: String(e?.message || e) }),
      )
      .finally(() => setLoadingSource(false));
  }, [editing?.name]);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const skill = editing
        ? await api.updateMarkdownSkill(editing.name, content)
        : await api.createMarkdownSkill(content);
      onCreated?.(skill);
    } catch (e) {
      if (e instanceof ApiError) {
        try {
          const parsed = JSON.parse(e.body);
          setError(parsed.detail ?? parsed);
        } catch {
          setError({ error_kind: "unknown", error_detail: e.message });
        }
      } else {
        setError({ error_kind: "unknown", error_detail: String((e as Error)?.message || e) });
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-6 space-y-4 overflow-y-auto h-full max-w-4xl">
      <CardHeader
        title={editing ? `Edit skill: ${editing.name}` : "New Skill (Markdown)"}
        description="A SKILL.md document: YAML frontmatter (name, triggers, surfaces) plus a Markdown instruction body. The body is injected into the chatbot and agents when the skill is active — no code, nothing executed."
      />

      <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-2 text-[11px] text-emerald-300 font-mono flex items-start gap-2">
        <ShieldCheck className="h-3.5 w-3.5 shrink-0 mt-0.5" />
        Markdown skills are safe: they are parsed as text and never run code.
        Activation is deterministic (triggers / always_on / pin), so even small
        local models use them reliably.
      </div>

      <Field label="SKILL.md">
        <TextArea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          disabled={loadingSource}
          spellCheck={false}
          rows={24}
          className="font-mono text-[11px] leading-relaxed"
        />
      </Field>

      {error && (
        <div className="rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2 text-xs font-mono text-red-300">
          <p className="font-semibold uppercase tracking-wider text-[10px] mb-1">
            {error.error_kind.replace(/_/g, " ")}
          </p>
          <p className="whitespace-pre-wrap break-words">{error.error_detail}</p>
        </div>
      )}

      <div className="flex justify-end">
        <button className="av-btn-primary" onClick={save} disabled={saving || loadingSource}>
          <Save className="h-3.5 w-3.5" />{" "}
          {saving ? "Saving…" : editing ? "Save changes" : "Create skill"}
        </button>
      </div>
    </div>
  );
}
