import { useState } from "react";
import { ShieldAlert } from "lucide-react";
import { Card, CardHeader } from "../../components/Card";
import { Pill } from "../../components/Pill";
import { StatCard } from "../../components/StatCard";
import { api, type GovScanResult } from "../../lib/api";
import { ErrorLine, useAsync, type SectionProps } from "./common";

export default function GovernanceSection({ refreshKey }: SectionProps) {
  const { data: policy, error } = useAsync(() => api.govPolicy(), [refreshKey]);
  const [text, setText] = useState("");
  const [scan, setScan] = useState<GovScanResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [erase, setErase] = useState({ run_id: "", owner: "" });

  const run = async (key: string, fn: () => Promise<string>) => {
    setBusy(key);
    setMsg(null);
    try {
      setMsg(await fn());
    } catch (e: any) {
      setMsg(String(e?.message || e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6">
      <ErrorLine error={error} />
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Retention" value={policy ? `${policy.retention_days}d` : "--"} tone="neon" />
        <StatCard label="Store full text" value={policy ? (policy.store_full_text ? "yes" : "preview") : "--"} />
        <StatCard label="Scrub PII" value={policy ? (policy.scrub_pii ? "on" : "off") : "--"} tone={policy?.scrub_pii ? "neon" : "amber"} />
        <StatCard label="Preview cap" value={policy ? `${policy.preview_cap}` : "--"} />
      </div>

      <Card padded>
        <CardHeader
          eyebrow="Data governance (M)"
          title="PII scan"
          description="Non-destructive: counts PII in a snippet and returns a scrubbed preview."
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste text to audit for email / card / IPv4 / phone…"
          className="av-input font-mono h-24 resize-none"
        />
        <div className="flex items-center gap-2 mt-2">
          <button
            className="av-btn-primary"
            disabled={!text.trim() || busy === "scan"}
            onClick={() => run("scan", async () => {
              const r = await api.govScan(text);
              setScan(r);
              return `Found ${r.total} PII match(es).`;
            })}
          >
            <ShieldAlert className="h-3 w-3" /> Scan
          </button>
          {scan && (
            <div className="flex items-center gap-1.5">
              {scan.findings.map((f) => (
                <Pill key={f.type} tone={f.count ? "amber" : "slate"}>{f.type}: {f.count}</Pill>
              ))}
            </div>
          )}
        </div>
        {scan && (
          <pre className="mt-2 bg-slate-950 border border-white/5 rounded p-2 text-[11px] font-mono text-slate-300 whitespace-pre-wrap max-h-40 overflow-y-auto">
            {scan.scrubbed}
          </pre>
        )}
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Card padded>
          <CardHeader eyebrow="Retention" title="Purge expired" description="TTL sweep across every observation store." />
          <button
            className="av-btn-ghost"
            disabled={busy === "purge"}
            onClick={() => {
              if (!window.confirm("Delete rows older than the retention window across all stores?")) return;
              void run("purge", async () => {
                const r = await api.govPurge();
                return `Purged ${r.total} row(s): ${JSON.stringify(r.deleted)}`;
              });
            }}
          >
            Run TTL purge
          </button>
        </Card>
        <Card padded>
          <CardHeader eyebrow="Right to erasure" title="Erase by run / owner" description="Deletes every row for one run_id or owner." />
          <div className="space-y-2">
            <input
              className="av-input"
              placeholder="run_id"
              value={erase.run_id}
              onChange={(e) => setErase({ run_id: e.target.value, owner: "" })}
            />
            <input
              className="av-input"
              placeholder="owner"
              value={erase.owner}
              onChange={(e) => setErase({ run_id: "", owner: e.target.value })}
            />
            <button
              className="av-btn-ghost"
              disabled={busy === "erase" || (!erase.run_id.trim() && !erase.owner.trim())}
              onClick={() => {
                if (!window.confirm("Permanently erase all rows for this subject?")) return;
                void run("erase", async () => {
                  const r = await api.govErase(
                    erase.run_id.trim() ? { run_id: erase.run_id.trim() } : { owner: erase.owner.trim() },
                  );
                  return `Erased ${r.total} row(s): ${JSON.stringify(r.deleted)}`;
                });
              }}
            >
              Erase subject
            </button>
          </div>
        </Card>
      </div>

      {msg && <p className="text-xs font-mono text-emerald-400">{msg}</p>}
    </div>
  );
}
