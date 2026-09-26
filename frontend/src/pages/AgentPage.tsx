import { useState } from "react";
import { FilePlus2, FileText, Layers, PanelLeftClose, Play, Users } from "lucide-react";
import { RunTab, type AgentProfileLaunch } from "../components/agent/RunTab";
import { ProfilesTab } from "../components/agent/ProfilesTab";
import { SkillsTab } from "../components/agent/SkillsTab";
import { NewSkillTab } from "../components/agent/NewSkillTab";
import { NewMarkdownSkillTab } from "../components/agent/NewMarkdownSkillTab";
import { ContextFileTab } from "../components/agent/ContextFileTab";
import { CollapsedRail } from "../components/CollapsedRail";
import { useAgentSession } from "../store/agentSession";
import { cn } from "../lib/utils";
import type { SkillSummary } from "../lib/api";

type AgentTab = "run" | "profiles" | "skills" | "new-skill" | "context";
type CreateFormat = "markdown" | "python";

const CATEGORIES: { key: AgentTab; label: string; icon: typeof Play }[] = [
  { key: "run", label: "Run", icon: Play },
  { key: "profiles", label: "Agent Profiles", icon: Users },
  { key: "skills", label: "Skills", icon: Layers },
  { key: "new-skill", label: "New Skill", icon: FilePlus2 },
  { key: "context", label: "Context File", icon: FileText },
];

// Stateful, session-shaped Agent Loop (Phase 4), split into a left-nav
// category shell (matching Settings' split-pane pattern): Run (the
// original single view), Agent Profiles (save + launch a {task, skills}
// bundle), Skills (library of built-in + custom skills), and New Skill
// (create/edit a custom skill).
export default function AgentPage() {
  const [tab, setTab] = useState<AgentTab>("run");
  const [pendingLaunch, setPendingLaunch] = useState<AgentProfileLaunch | null>(null);
  const [editingSkill, setEditingSkill] = useState<
    { name: string; format: SkillSummary["format"] } | null
  >(null);
  // Format for a *fresh* skill; markdown is the friendly default. When
  // editing, the skill's own format wins (see the new-skill render below).
  const [createFormat, setCreateFormat] = useState<CreateFormat>("markdown");
  const { agentNavCollapsed, setAgentNavCollapsed } = useAgentSession();

  // Clicking the left nav directly always starts New Skill fresh; editing
  // a specific skill only happens via SkillsTab's "Edit" action, which
  // sets both editingSkill and tab together (bypassing this handler).
  const handleTabChange = (next: AgentTab) => {
    setEditingSkill(null);
    setTab(next);
  };

  return (
    <div className="flex h-full overflow-hidden">
      {agentNavCollapsed ? (
        <CollapsedRail side="left" label="Agent" onExpand={() => setAgentNavCollapsed(false)} />
      ) : (
        <div className="w-60 border-r border-muted bg-header flex flex-col shrink-0">
          <div className="p-4 border-b border-muted flex items-center justify-between">
            <h2 className="text-sm font-bold text-white">Agent</h2>
            <button
              onClick={() => setAgentNavCollapsed(true)}
              title="Collapse"
              className="av-btn-icon h-5 w-5"
            >
              <PanelLeftClose className="h-3 w-3" />
            </button>
          </div>
          <nav className="p-2 space-y-0.5 overflow-y-auto flex-1">
            {CATEGORIES.map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => handleTabChange(key)}
                className={cn("av-nav-item w-full", tab === key && "is-active font-medium")}
              >
                <Icon className={cn("h-3.5 w-3.5 shrink-0", tab === key ? "text-emerald-400" : "text-slate-500")} />
                <span className="truncate flex-1 text-left">{label}</span>
              </button>
            ))}
          </nav>
        </div>
      )}

      <div className="flex-1 overflow-hidden">
        {tab === "run" && (
          <RunTab
            pendingLaunch={pendingLaunch}
            onLaunchConsumed={() => setPendingLaunch(null)}
          />
        )}
        {tab === "profiles" && (
          <ProfilesTab
            onLaunch={(launch) => {
              setPendingLaunch(launch);
              setTab("run");
            }}
          />
        )}
        {tab === "skills" && (
          <SkillsTab
            onEdit={(skill: SkillSummary) => {
              setEditingSkill({ name: skill.name, format: skill.format });
              setTab("new-skill");
            }}
          />
        )}
        {tab === "new-skill" &&
          (() => {
            // Editing keeps the skill's own format; creating uses the toggle.
            const format: CreateFormat =
              editingSkill != null
                ? editingSkill.format === "python"
                  ? "python"
                  : "markdown"
                : createFormat;
            const onCreated = () => {
              setEditingSkill(null);
              setTab("skills");
            };
            return (
              <div className="h-full flex flex-col">
                {!editingSkill && (
                  <div className="px-6 pt-5 flex items-center gap-2">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500">
                      Format
                    </span>
                    {(["markdown", "python"] as CreateFormat[]).map((f) => (
                      <button
                        key={f}
                        onClick={() => setCreateFormat(f)}
                        className={cn(
                          "px-2.5 py-1 rounded-full text-[10px] font-mono uppercase tracking-wider border transition",
                          createFormat === f
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                            : "bg-slate-950 text-slate-500 border-white/5 hover:text-slate-300",
                        )}
                      >
                        {f}
                      </button>
                    ))}
                  </div>
                )}
                <div className="flex-1 overflow-hidden">
                  {format === "markdown" ? (
                    <NewMarkdownSkillTab editing={editingSkill} onCreated={onCreated} />
                  ) : (
                    <NewSkillTab editing={editingSkill} onCreated={onCreated} />
                  )}
                </div>
              </div>
            );
          })()}
        {tab === "context" && <ContextFileTab />}
      </div>
    </div>
  );
}
