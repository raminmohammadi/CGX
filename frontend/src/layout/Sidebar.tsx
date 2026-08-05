import { useEffect, useState } from "react";
import { NavLink, useNavigate } from "react-router";
import {
  Bot,
  ChevronDown,
  ChevronRight,
  CircleCheck,
  Database,
  LayoutDashboard,
  Loader2,
  MessageSquareDot,
  Plus,
  Settings,
  Trash2,
  Wand2,
  Activity,
  ShieldCheck,
} from "lucide-react";
import { api, type SessionSummary } from "../lib/api";
import { useWorkspace } from "../store/workspace";
import { useTasks } from "../store/tasks";
import { cn, formatRelative } from "../lib/utils";

type NavTab = {
  to: string;
  label: string;
  icon: typeof Bot;
  pageKey: "agent" | "ask" | "plan" | "index" | null;
};

const overviewTab: NavTab = { to: "/", label: "Overview", icon: LayoutDashboard, pageKey: null };

const navGroups: { eyebrow: string; tabs: NavTab[] }[] = [
  {
    eyebrow: "Converse",
    tabs: [{ to: "/ask", label: "Contextual Ask", icon: MessageSquareDot, pageKey: "ask" }],
  },
  {
    eyebrow: "Build",
    tabs: [
      { to: "/plan", label: "Self-Testing Plan", icon: Wand2, pageKey: "plan" },
      { to: "/agent", label: "Agent Loop", icon: Bot, pageKey: null },
    ],
  },
  {
    eyebrow: "Retrieval",
    tabs: [{ to: "/index", label: "Incremental Index", icon: Database, pageKey: "index" }],
  },
  {
    eyebrow: "Observability",
    tabs: [
      { to: "/activity", label: "User Activity", icon: Activity, pageKey: null },
      { to: "/admin", label: "Admin", icon: ShieldCheck, pageKey: null },
    ],
  },
  {
    eyebrow: "System",
    tabs: [{ to: "/settings", label: "Profiles & Setup", icon: Settings, pageKey: null }],
  },
];

function useRunningPages() {
  const { agent, ask, plan, index } = useTasks();
  return {
    agent: agent.busy,
    ask: ask.busy,
    plan: plan.busy,
    index: index.busy,
  };
}

export default function Sidebar() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [navCollapsed, setNavCollapsed] = useState(false);
  const selected = useWorkspace((s) => s.selectedSessionId);
  const setSelected = useWorkspace((s) => s.setSelectedSession);
  const navigate = useNavigate();
  const running = useRunningPages();

  const refresh = async () => {
    try {
      const list = await api.listSessions();
      setSessions(list);
    } catch {
      setSessions([]);
    }
  };
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 8000);
    return () => clearInterval(id);
  }, []);

  const createSession = async () => {
    try {
      const s = await api.createSession();
      await refresh();
      setSelected(s.id);
      navigate("/ask");
    } catch {
      /* surface in main page later */
    }
  };

  const removeSession = async (sid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await api.deleteSession(sid);
      if (selected === sid) setSelected(null);
      await refresh();
    } catch {
      /* ignore */
    }
  };

  return (
    <aside
      className="w-64 bg-header border-r flex flex-col justify-between flex-shrink-0"
      style={{ borderColor: "rgba(255,255,255,0.06)" }}
    >
      <div className="p-3 space-y-5 overflow-y-auto">
        <div>
          <NavLink
            to={overviewTab.to}
            end
            className={({ isActive }) => cn("av-nav-item mb-2", isActive && "is-active font-medium")}
          >
            {({ isActive }) => (
              <>
                <overviewTab.icon
                  className={cn("h-3.5 w-3.5 shrink-0", isActive ? "text-emerald-400" : "text-slate-500")}
                />
                <span className="truncate flex-1">{overviewTab.label}</span>
              </>
            )}
          </NavLink>

          <button
            onClick={() => setNavCollapsed((c) => !c)}
            className="flex items-center justify-between w-full px-2 mb-2 group"
          >
            <span className="av-section-eyebrow">Engine Core Layers</span>
            {navCollapsed
              ? <ChevronRight className="h-3 w-3 text-slate-600 group-hover:text-slate-400 transition" />
              : <ChevronDown className="h-3 w-3 text-slate-600 group-hover:text-slate-400 transition" />
            }
          </button>
          {!navCollapsed && (
            <div className="space-y-3">
              {navGroups.map((group) => (
                <div key={group.eyebrow}>
                  <span className="av-section-eyebrow block px-2 mb-1 text-[9px]">
                    {group.eyebrow}
                  </span>
                  <nav className="space-y-0.5">
                    {group.tabs.map(({ to, label, icon: Icon, pageKey }) => {
                      const isRunning = pageKey ? running[pageKey] : false;
                      return (
                        <NavLink
                          key={to}
                          to={to}
                          className={({ isActive }) =>
                            cn("av-nav-item", isActive && "is-active font-medium")
                          }
                        >
                          {({ isActive }) => (
                            <>
                              <Icon
                                className={cn(
                                  "h-3.5 w-3.5 shrink-0",
                                  isActive ? "text-emerald-400" : "text-slate-500",
                                )}
                              />
                              <span className="truncate flex-1">{label}</span>
                              {isRunning && (
                                <span title="Task running">
                                  <Loader2 className="h-3 w-3 text-emerald-400 animate-spin shrink-0" />
                                </span>
                              )}
                            </>
                          )}
                        </NavLink>
                      );
                    })}
                  </nav>
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          <div className="flex items-center justify-between px-2 mb-1.5">
            <span className="av-section-eyebrow">Active turn paths</span>
            <button
              onClick={createSession}
              title="New session"
              className="av-btn-icon h-5 w-5"
            >
              <Plus className="h-3 w-3" />
            </button>
          </div>
          <div className="space-y-1 max-h-56 overflow-y-auto">
            {sessions.length === 0 && (
              <p className="text-[10px] text-slate-500 px-2 font-mono italic">
                No sessions yet -- your Ask turns will land here.
              </p>
            )}
            {sessions.map((s) => (
              <div
                key={s.id}
                onClick={() => {
                  setSelected(s.id);
                  navigate("/ask");
                }}
                className={cn(
                  "group p-2 rounded text-xs cursor-pointer border transition",
                  selected === s.id
                    ? "bg-slate-950 border-emerald-500/30"
                    : "bg-slate-950/40 border-white/5 hover:border-white/10",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="text-slate-200 font-medium truncate flex-1">
                    {s.title || s.id.slice(0, 8)}
                  </p>
                  <button
                    onClick={(e) => removeSession(s.id, e)}
                    className="opacity-0 group-hover:opacity-100 transition text-slate-500 hover:text-red-400"
                    title="Delete session"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
                <span className="text-[9px] text-slate-500 font-mono">
                  {s.message_count} turns · {formatRelative(s.updated_at)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div
        className="p-3 border-t bg-slate-950/40 flex items-center justify-between"
        style={{ borderColor: "rgba(255,255,255,0.06)" }}
      >
        <span className="text-[10px] text-slate-500 font-mono">
          Storage context: ~/.cgx/
        </span>
        <CircleCheck
          className="text-emerald-500 h-3.5 w-3.5"
          aria-label="0700 owner-only permissions"
        />
      </div>
    </aside>
  );
}
