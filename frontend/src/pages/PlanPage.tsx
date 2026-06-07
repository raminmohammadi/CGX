import { useEffect, useState } from "react";
import { CircleCheck, FlaskConical, Play, Square, TriangleAlert } from "lucide-react";
import { streamSSE } from "../lib/sse";
import { abortConnection, getConnection, setConnection } from "../lib/connections";
import { useTasks } from "../store/tasks";
import { useWorkspace } from "../store/workspace";
import { Card, CardHeader } from "../components/Card";
import { Field, TextArea, TextInput, Toggle } from "../components/Input";
import { Markdown } from "../components/Markdown";
import { DiffView } from "../components/DiffView";
import { Pill } from "../components/Pill";

const PAGE_KEY = "plan";

export default function PlanPage() {
  const { provider, index, projectRoot, setProjectRoot } = useWorkspace();
  const { plan, setPlan, resetPlan } = useTasks();
  const { busy, thought, warning, planMd, diff, report, error } = plan;

  // Form fields use local state -- they're user input, not task output.
  const [task, setTask] = useState("");
  const [selfTest, setSelfTest] = useState(false);
  const [runTests, setRunTests] = useState(false);

  // On mount: if busy but no live connection, stream finished while away.
  useEffect(() => {
    if (busy && !getConnection(PAGE_KEY)) {
      setPlan({ busy: false });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const run = () => {
    if (!task.trim() || busy) return;

    setPlan({ busy: true, thought: "", warning: null, planMd: null, diff: null, report: null, error: null });

    abortConnection(PAGE_KEY);
    const conn = streamSSE(
      "/api/plan",
      {
        task,
        project_root: projectRoot || null,
        self_test: selfTest,
        run_tests: runTests,
        index,
        provider,
      },
      (ev, data) => {
        if (ev === "thought" && data?.delta) {
          useTasks.setState((s) => ({
            plan: { ...s.plan, thought: s.plan.thought + String(data.delta) },
          }));
        } else if (ev === "thought_warning") {
          setPlan({ warning: String(data?.message || "") });
        } else if (ev === "plan") {
          const diffsArr: Array<{ file: string; patch: string }> = data?.diffs || [];
          const combined = diffsArr.map((d) => (d.patch || "").trim()).filter(Boolean).join("\n");
          setPlan({
            planMd: String(data?.plan_md || ""),
            diff: combined,
            report: (data?.report as any) || {},
            busy: false,
          });
        } else if (ev === "cancelled") {
          setPlan({ busy: false, error: "Cancelled." });
        } else if (ev === "error") {
          setPlan({ error: String(data?.message || "error"), busy: false });
        }
      },
      (err) => {
        setPlan({ error: String((err as any)?.message || err), busy: false });
      },
    );

    setConnection(PAGE_KEY, conn);
    conn.done.finally(() => {
      setPlan({ busy: false });
      abortConnection(PAGE_KEY);
    });
  };

  const cancel = () => {
    abortConnection(PAGE_KEY);
    setPlan({ busy: false });
  };

  const reportPills = (() => {
    const out: { tone: "neon" | "amber" | "red" | "slate"; label: string }[] = [];
    if (report) {
      if (report.ast_ok) out.push({ tone: "neon", label: "AST Match" });
      else if (report.ast_ok === false) out.push({ tone: "red", label: "AST Failed" });
      if (report.tests_ok) out.push({ tone: "neon", label: "Pytest Success" });
      else if (report.tests_ok === false) out.push({ tone: "amber", label: "Pytest Failed" });
    }
    return out;
  })();

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-5xl">
      <CardHeader
        title="Self-Testing Verification Loop"
        description="Executes isolated dry-run evaluations, verifying code change validity inside safe AST parsing environments."
        right={
          busy ? (
            <button onClick={cancel} className="av-btn-ghost">
              <Square className="h-3 w-3" /> Cancel
            </button>
          ) : null
        }
      />

      <Card padded>
        <div className="grid grid-cols-3 gap-3">
          <Field className="col-span-3" label="Task description">
            <TextArea
              rows={4}
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder='e.g. "Replace fixed-length file-path guard in cgx/parser.py with a 1MB size cap"'
            />
          </Field>
          <Field label="Project root" className="col-span-2">
            <TextInput
              value={projectRoot}
              onChange={(e) => setProjectRoot(e.target.value)}
              placeholder="/abs/path/to/repo"
            />
          </Field>
          <div className="flex flex-col gap-3 justify-end">
            <Toggle checked={selfTest} onChange={setSelfTest} label="Self-test (1 retry)" />
            <Toggle checked={runTests} onChange={setRunTests} label="Run pytest" />
          </div>
        </div>
        <div className="flex justify-between items-center mt-4">
          <div className="flex gap-2">
            <button onClick={resetPlan} disabled={busy} className="av-btn-ghost text-[10px]">
              Clear
            </button>
            <p className="text-[10px] text-slate-500 font-mono self-center">
              Project root needed for AST/pytest checks.
            </p>
          </div>
          <button onClick={run} disabled={busy} className="av-btn-primary">
            <Play className="h-3 w-3" /> {busy ? "Generating…" : "Generate Plan"}
          </button>
        </div>
      </Card>

      {(thought || warning) && (
        <Card padded>
          <div className="flex items-center gap-2 mb-2 text-[10px] uppercase tracking-wider text-slate-500 font-mono">
            <span className="av-dot" /> Planner thinking
          </div>
          {thought && (
            <pre className="text-[11px] text-slate-400 font-mono whitespace-pre-wrap leading-relaxed">
              {thought}
            </pre>
          )}
          {warning && (
            <p className="text-[10px] text-amber-400/80 bg-amber-500/5 px-2 py-1 rounded border border-amber-500/10 font-mono mt-3">
              <TriangleAlert className="inline h-3 w-3 mr-1" />
              {warning}
            </p>
          )}
        </Card>
      )}

      {error && (
        <Card padded className="border-red-500/40">
          <p className="text-xs text-red-300 font-mono">{error}</p>
        </Card>
      )}

      {planMd !== null && (
        <div className="space-y-4">
          <Card padded>
            <CardHeader
              eyebrow="Planner output"
              title="Plan"
              right={reportPills.map((p, i) => (
                <Pill key={i} tone={p.tone}>
                  <CircleCheck className="h-3 w-3" /> {p.label}
                </Pill>
              ))}
            />
            {planMd ? (
              <Markdown text={planMd} />
            ) : (
              <p className="text-[11px] text-slate-500 font-mono italic">
                No plan text was produced. The model may need a larger context or a more specific task description.
              </p>
            )}
          </Card>

          <div className="bg-surface rounded-xl border border-muted overflow-hidden">
            <div className="bg-slate-950 px-4 py-3 border-b border-muted flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Pill tone="neon">Parsed Codegen Report</Pill>
                <span className="text-xs text-slate-300 font-mono">
                  {report?.summary || "diffs"}
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs font-mono">
                {report?.ast_ok && (
                  <span className="text-emerald-400 font-semibold flex items-center gap-1">
                    <CircleCheck className="h-3 w-3" /> AST Match
                  </span>
                )}
                {report?.tests_ok && (
                  <span className="text-emerald-400 font-semibold flex items-center gap-1">
                    <FlaskConical className="h-3 w-3" /> Pytest Success
                  </span>
                )}
              </div>
            </div>
            <div className="p-4">
              <DiffView diff={diff || ""} />
            </div>
            {report?.log_tail && (
              <details className="border-t border-muted">
                <summary className="px-4 py-2 text-[10px] font-mono text-slate-400 cursor-pointer hover:text-slate-200">
                  pytest log tail
                </summary>
                <pre className="px-4 pb-3 text-[10px] text-slate-400 font-mono whitespace-pre-wrap leading-relaxed">
                  {report.log_tail}
                </pre>
              </details>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
