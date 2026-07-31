import { useEffect, useState } from "react";
import { Save, TriangleAlert } from "lucide-react";
import { CardHeader } from "../Card";
import { Field, TextArea } from "../Input";
import { api, ApiError, type SkillSummary, type SkillValidationError } from "../../lib/api";

const DEFAULT_TEMPLATE = `from skills.base import Skill


class MySkill(Skill):
    # Stable, lower-snake identifier used in logs / task inputs.
    name = "my_skill"
    # One of: frontend, backend, fullstack, data, cli, style, infra.
    role = "infra"
    # Optional surface-form aliases (display name + common misspellings).
    aliases = ()
    # One-line summary shown on this skill's card.
    description = ""

    def detect(self, goal: str) -> float:
        """Return a confidence in [0.0, 1.0] that goal involves this skill."""
        return 0.0

    def scaffold_system_prompt(self) -> str:
        """Prompt fragment added when scaffolding a new project. Optional."""
        return ""

    def plan_system_prompt(self) -> str:
        """Prompt fragment added when planning a code change. Optional."""
        return ""
`;

export function NewSkillTab({
  editing,
  onCreated,
}: {
  editing?: { name: string } | null;
  onCreated?: (skill: SkillSummary) => void;
}) {
  const [source, setSource] = useState(DEFAULT_TEMPLATE);
  const [loadingSource, setLoadingSource] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<SkillValidationError | null>(null);

  useEffect(() => {
    if (!editing) {
      setSource(DEFAULT_TEMPLATE);
      return;
    }
    setLoadingSource(true);
    setError(null);
    api
      .getSkillSource(editing.name)
      .then((r) => setSource(r.source))
      .catch((e) => setError({ error_kind: "load_failed", error_detail: String(e?.message || e) }))
      .finally(() => setLoadingSource(false));
  }, [editing?.name]);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const skill = editing
        ? await api.updateSkill(editing.name, source)
        : await api.createSkill(source);
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
        title={editing ? `Edit skill: ${editing.name}` : "New Skill"}
        description="Define one Python class inheriting from skills.base.Skill. It's validated (syntax, a single Skill subclass, a working detect() call) before it's saved."
      />

      <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-[11px] text-amber-300 font-mono flex items-start gap-2">
        <TriangleAlert className="h-3.5 w-3.5 shrink-0 mt-0.5" />
        Custom skill code runs with full local privileges when a session executes it — review it like you would a VS Code extension before saving.
      </div>

      <Field label="Skill source">
        <TextArea
          value={source}
          onChange={(e) => setSource(e.target.value)}
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
          <Save className="h-3.5 w-3.5" /> {saving ? "Saving…" : editing ? "Save changes" : "Create skill"}
        </button>
      </div>
    </div>
  );
}
