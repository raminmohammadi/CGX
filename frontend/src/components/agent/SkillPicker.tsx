import { useEffect, useState } from "react";
import { Pill } from "../Pill";
import { api, type SkillSummary } from "../../lib/api";

// Checklist of every known skill (built-in + custom) with its description
// visible inline, so a user can read what a skill does while deciding
// whether to assign it -- not just a name to hover over.
export function SkillPicker({
  selected,
  onChange,
}: {
  selected: string[];
  onChange: (names: string[]) => void;
}) {
  const [skills, setSkills] = useState<SkillSummary[]>([]);

  useEffect(() => {
    api.listSkills().then(setSkills).catch(() => setSkills([]));
  }, []);

  const toggle = (name: string) => {
    onChange(
      selected.includes(name)
        ? selected.filter((n) => n !== name)
        : [...selected, name],
    );
  };

  if (skills.length === 0) {
    return <p className="text-[10px] text-slate-500 font-mono italic">No skills yet.</p>;
  }

  return (
    <div className="max-h-64 overflow-y-auto border border-white/5 rounded-lg divide-y divide-white/5">
      {skills.map((s) => {
        const checked = selected.includes(s.name);
        return (
          <label
            key={s.name}
            className="flex items-start gap-2.5 px-3 py-2 cursor-pointer hover:bg-white/5 transition"
          >
            <input
              type="checkbox"
              checked={checked}
              onChange={() => toggle(s.name)}
              className="mt-0.5 rounded shrink-0"
            />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs font-semibold text-slate-100">{s.name}</span>
                <Pill tone="slate" className="text-[9px]">{s.role}</Pill>
                {s.is_custom && <Pill tone="purple" className="text-[9px]">custom</Pill>}
              </div>
              {s.description && (
                <p className="text-[10px] text-slate-500 mt-0.5">{s.description}</p>
              )}
            </div>
          </label>
        );
      })}
    </div>
  );
}
