import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight, Play, Plus, Save, Trash2, X } from "lucide-react";
import { Card, CardHeader } from "../Card";
import { Field, Select, TextArea, TextInput } from "../Input";
import { Pill } from "../Pill";
import { SkillPicker } from "./SkillPicker";
import { cn } from "../../lib/utils";
import { api, type AgentProfileSummary } from "../../lib/api";
import type { AgentProfileLaunch } from "./RunTab";
import type { SessionModeValue } from "../../lib/api";

type ModeChoice = "auto" | SessionModeValue;

const emptyForm = { name: "", objective: "", project_root: "", mode: "auto" as ModeChoice, skills: [] as string[] };

let launchCounter = 0;

export function ProfilesTab({
  onLaunch,
}: {
  onLaunch: (launch: AgentProfileLaunch) => void;
}) {
  const [profiles, setProfiles] = useState<AgentProfileSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<typeof emptyForm | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = () => {
    api
      .listAgentProfiles()
      .then(setProfiles)
      .catch((e) => setError(String(e?.message || e)));
  };

  useEffect(load, []);

  const startNew = () => setEditing({ ...emptyForm });
  const startEdit = (p: AgentProfileSummary) =>
    setEditing({
      name: p.name, objective: p.objective, project_root: p.project_root,
      mode: (p.mode || "auto") as ModeChoice, skills: p.skills,
    });

  const save = async () => {
    if (!editing) return;
    if (!editing.name.trim() || !editing.objective.trim()) {
      setError("Name and objective are required.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await api.upsertAgentProfile(editing.name.trim(), {
        objective: editing.objective.trim(),
        project_root: editing.project_root.trim(),
        mode: editing.mode === "auto" ? "" : editing.mode,
        skills: editing.skills,
      });
      setEditing(null);
      load();
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setSaving(false);
    }
  };

  const remove = async (name: string) => {
    if (!confirm(`Delete agent profile "${name}"?`)) return;
    try {
      await api.deleteAgentProfile(name);
      load();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  };

  const launch = (p: AgentProfileSummary) => {
    launchCounter += 1;
    onLaunch({
      id: launchCounter,
      objective: p.objective,
      projectRoot: p.project_root,
      mode: (p.mode || null) as SessionModeValue | null,
      skills: p.skills,
    });
  };

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-4xl">
      <CardHeader
        title="Agent Profiles"
        description="Save a task + skill bundle and launch it into a new session without re-typing it."
        right={
          <button onClick={startNew} className="av-btn-primary">
            <Plus className="h-3 w-3" /> New profile
          </button>
        }
      />

      {error && <p className="text-xs text-red-300 font-mono">{error}</p>}

      {profiles.length === 0 ? (
        <p className="text-xs text-slate-500 text-center py-6">
          No agent profiles yet -- click <strong>New profile</strong> above to create one.
        </p>
      ) : (
        <div className="space-y-2">
          {profiles.map((p) => {
            const isOpen = expanded === p.name;
            return (
              <Card key={p.name} padded={false} className="overflow-hidden">
                <button
                  onClick={() => setExpanded(isOpen ? null : p.name)}
                  className="w-full flex items-center gap-3 px-4 py-3 text-left"
                >
                  {isOpen ? (
                    <ChevronDown className="h-3 w-3 text-slate-500 shrink-0" />
                  ) : (
                    <ChevronRight className="h-3 w-3 text-slate-500 shrink-0" />
                  )}
                  <span className="text-xs font-semibold text-white truncate">{p.name}</span>
                  <span className="text-[10px] text-slate-500 font-mono truncate flex-1">
                    {p.objective}
                  </span>
                  {p.mode && <Pill tone="slate">{p.mode}</Pill>}
                </button>
                {isOpen && (
                  <div className="px-4 pb-4 space-y-3">
                    <p className="text-[11px] text-slate-400">{p.objective}</p>
                    {p.project_root && (
                      <p className="text-[10px] font-mono text-slate-500">root: {p.project_root}</p>
                    )}
                    {p.skills.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {p.skills.map((s) => (
                          <span
                            key={s}
                            className="text-[9px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded px-1.5 py-0.5"
                          >
                            {s}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="flex items-center justify-end gap-1.5">
                      <button className="av-btn-primary py-1 px-2 text-[10px]" onClick={() => launch(p)}>
                        <Play className="h-3 w-3" /> Launch
                      </button>
                      <button className="av-btn-ghost py-1 px-2 text-[10px]" onClick={() => startEdit(p)}>
                        Edit
                      </button>
                      <button
                        className="av-btn py-1 px-2 text-[10px] bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20"
                        onClick={() => remove(p.name)}
                      >
                        <Trash2 className="h-3 w-3" /> Delete
                      </button>
                    </div>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}

      {editing && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setEditing(null);
          }}
        >
          <div className="bg-[#0d1117] border border-white/10 rounded-xl w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto shadow-2xl shadow-black/60">
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
              <h2 className="text-sm font-semibold text-white font-mono">
                {editing.name ? "Edit agent profile" : "New agent profile"}
              </h2>
              <div className="flex items-center gap-2">
                <button className="av-btn-ghost" onClick={() => setEditing(null)}>
                  <X className="h-3 w-3" /> Cancel
                </button>
                <button className="av-btn-primary" onClick={save} disabled={saving}>
                  <Save className="h-3 w-3" /> {saving ? "Saving…" : "Save"}
                </button>
              </div>
            </div>
            <div className="p-6 space-y-3">
              <Field label="Name">
                <TextInput
                  value={editing.name}
                  onChange={(e) => setEditing({ ...editing, name: e.target.value })}
                />
              </Field>
              <Field label="Objective">
                <TextArea
                  rows={3}
                  value={editing.objective}
                  onChange={(e) => setEditing({ ...editing, objective: e.target.value })}
                  placeholder="e.g. Add a GraphQL API for the todos resource"
                />
              </Field>
              <Field label="Project root" hint="Absolute path on disk; optional.">
                <TextInput
                  value={editing.project_root}
                  onChange={(e) => setEditing({ ...editing, project_root: e.target.value })}
                  placeholder="/path/to/repo"
                />
              </Field>
              <Field label="Mode">
                <Select
                  value={editing.mode}
                  onChange={(e) => setEditing({ ...editing, mode: (e.target as any).value })}
                >
                  <option value="auto">auto</option>
                  <option value="explore">explore</option>
                  <option value="greenfield">greenfield</option>
                </Select>
              </Field>
              <Field label="Skills" hint="Optional; leave empty to auto-detect from the objective.">
                <SkillPicker
                  selected={editing.skills}
                  onChange={(skills) => setEditing({ ...editing, skills })}
                />
              </Field>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
