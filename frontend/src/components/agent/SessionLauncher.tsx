import { useState } from "react";
import { Rocket } from "lucide-react";
import { Field, TextArea, TextInput } from "../Input";
import { SkillPicker } from "./SkillPicker";
import { cn } from "../../lib/utils";
import type { SessionModeValue } from "../../lib/api";

type ModeChoice = "auto" | SessionModeValue;

// Empty-state form: collect an objective + project root and kick off a
// new session via ``api.agentSessionCreate``. The parent handles the
// actual create call so the page can swap to the live view on success.
export function SessionLauncher({
  defaultProjectRoot,
  onCreate,
  pending,
  error,
}: {
  defaultProjectRoot: string;
  onCreate: (opts: {
    objective: string; projectRoot: string;
    mode: SessionModeValue | null;
    skills: string[];
  }) => Promise<void> | void;
  pending: boolean;
  error: string | null;
}) {
  const [objective, setObjective] = useState("");
  const [projectRoot, setProjectRoot] = useState(defaultProjectRoot);
  const [mode, setMode] = useState<ModeChoice>("auto");
  const [skills, setSkills] = useState<string[]>([]);
  const canSubmit = objective.trim().length > 0 && !pending;
  return (
    <div className="max-w-2xl mx-auto w-full mt-8 px-6 space-y-5">
      <header className="space-y-1">
        <p className="av-section-eyebrow">Stateful Agent Loop</p>
        <h1 className="text-xl font-bold text-white">Start a new session</h1>
        <p className="text-[12px] text-slate-400">
          State the change you want and the agent will branch into
          investigation, recommend next steps, and gate the write loop on
          your explicit approval.
        </p>
      </header>
      <Field label="Objective" hint="One sentence; the agent will iterate from here.">
        <TextArea
          rows={3}
          value={objective}
          onChange={(e) => setObjective(e.target.value)}
          placeholder="e.g. Add rate limiting to the public webhook endpoint"
          disabled={pending}
        />
      </Field>
      <Field label="Project root" hint="Absolute path on disk; leave blank to use index defaults.">
        <TextInput
          value={projectRoot}
          onChange={(e) => setProjectRoot(e.target.value)}
          placeholder="/path/to/repo"
          disabled={pending}
        />
      </Field>
      <Field label="Mode" hint="Auto picks greenfield for empty / unindexed projects; explore otherwise.">
        <div className="flex gap-1.5">
          {(["auto", "explore", "greenfield"] as const).map((m) => (
            <button
              key={m} type="button" disabled={pending}
              onClick={() => setMode(m)}
              className={cn(
                "text-[11px] font-mono px-2 py-1 rounded border transition",
                mode === m
                  ? "border-emerald-500/40 bg-emerald-950/30 text-emerald-300"
                  : "border-white/10 bg-slate-950/40 text-slate-300 hover:border-white/20",
              )}
            >{m}</button>
          ))}
        </div>
      </Field>
      <Field label="Skills" hint="Optional; leave empty to auto-detect from the objective.">
        <SkillPicker selected={skills} onChange={setSkills} />
      </Field>
      {error && (
        <p className="text-[12px] font-mono text-red-300 bg-red-950/30 border border-red-500/20 rounded px-3 py-2">
          {error}
        </p>
      )}
      <div className="flex justify-end">
        <button
          type="button"
          disabled={!canSubmit}
          onClick={() => onCreate({
            objective: objective.trim(),
            projectRoot: projectRoot.trim(),
            mode: mode === "auto" ? null : mode,
            skills,
          })}
          className="av-btn-primary"
        >
          <Rocket className="h-3.5 w-3.5" /> Begin session
        </button>
      </div>
    </div>
  );
}
