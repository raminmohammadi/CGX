import { useEffect, useState } from "react";
import { CardHeader } from "../Card";
import { StatCard } from "../StatCard";
import { TabBar } from "../TabBar";
import { SkillCard } from "./SkillCard";
import { api, type SkillSummary } from "../../lib/api";

const ROLE_TABS: { key: string; label: string }[] = [
  { key: "all", label: "All" },
  { key: "frontend", label: "Frontend" },
  { key: "backend", label: "Backend" },
  { key: "fullstack", label: "Fullstack" },
  { key: "data", label: "Data" },
  { key: "cli", label: "CLI" },
  { key: "style", label: "Style" },
  { key: "infra", label: "Infra" },
];

export function SkillsTab({
  onEdit,
}: {
  onEdit: (skill: SkillSummary) => void;
}) {
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [role, setRole] = useState("all");
  const [search, setSearch] = useState("");

  const load = () => {
    setLoading(true);
    setError(null);
    api
      .listSkills()
      .then(setSkills)
      .catch((e) => setError(String(e?.message || e)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const remove = async (name: string) => {
    if (!confirm(`Delete custom skill "${name}"?`)) return;
    try {
      await api.deleteSkill(name);
      load();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  };

  const filtered = skills.filter((s) => {
    if (role !== "all" && s.role !== role) return false;
    if (search && !`${s.name} ${s.description}`.toLowerCase().includes(search.toLowerCase())) {
      return false;
    }
    return true;
  });

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-6xl">
      <CardHeader
        title="Skills"
        description="Technology-specific knowledge the agent can be pointed at when planning or scaffolding."
      />

      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Total skills" value={skills.length} tone="slate" />
        <StatCard
          label="Custom"
          value={skills.filter((s) => s.is_custom).length}
          tone="neon"
        />
        <StatCard
          label="Built-in"
          value={skills.filter((s) => !s.is_custom).length}
          tone="slate"
        />
      </div>

      <div className="flex items-center justify-between gap-3 flex-wrap">
        <TabBar tabs={ROLE_TABS} active={role} onChange={setRole} />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search skills…"
          className="av-input text-xs w-48"
        />
      </div>

      {error && <p className="text-xs text-red-300 font-mono">{error}</p>}

      {loading ? (
        <p className="text-xs text-slate-500 font-mono">Loading…</p>
      ) : filtered.length === 0 ? (
        <p className="text-xs text-slate-500 text-center py-6">No skills match.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {filtered.map((s) => (
            <SkillCard
              key={s.name}
              skill={s}
              onEdit={s.is_custom ? () => onEdit(s) : undefined}
              onDelete={s.is_custom ? () => remove(s.name) : undefined}
            />
          ))}
        </div>
      )}
    </div>
  );
}
