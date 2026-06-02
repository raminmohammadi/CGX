import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle, Microchip, RefreshCcw, XCircle } from "lucide-react";
import { api, type HardwareMatrixResponse } from "../lib/api";
import { Card, CardHeader } from "../components/Card";

export default function HardwarePage() {
  const [data, setData] = useState<HardwareMatrixResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      const d = await api.hardwareMatrix();
      setData(d);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const ram = data?.hardware?.ram_gb;
  const vram = data?.hardware?.gpu_vram_gb;

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-6xl">
      <CardHeader
        title="Hardware-Aware Local Catalog"
        description="Cross-references localized system resources directly against 4-bit quantized GGUF inference thresholds."
        right={
          <button onClick={load} disabled={busy} className="av-btn-primary">
            <Microchip className="h-3 w-3" />
            {busy ? "Detecting…" : "Detect Hardware Budget"}
          </button>
        }
      />

      <div className="grid grid-cols-3 gap-3">
        <Stat label="System RAM" value={ram != null ? `${ram.toFixed(1)} GB` : "—"} tone="emerald" />
        <Stat
          label="GPU VRAM"
          value={vram != null ? `${vram.toFixed(1)} GB` : "—"}
          tone={vram != null ? "emerald" : "slate"}
        />
        <Stat
          label="Catalog rows"
          value={data ? `${data.rows.length}` : "—"}
          tone="slate"
        />
      </div>

      <div className="bg-surface rounded-xl border border-muted overflow-hidden">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-slate-950 text-slate-400 uppercase border-b border-white/5">
            <tr>
              <th className="p-3 text-[10px]">Model</th>
              <th className="p-3 text-[10px]">Params</th>
              <th className="p-3 text-[10px]">Min RAM</th>
              <th className="p-3 text-[10px]">Rec VRAM</th>
              <th className="p-3 text-[10px]">Family</th>
              <th className="p-3 text-[10px]">Fit</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5 text-slate-300">
            {data?.rows?.map((r) => (
              <tr key={r.model} title={r.reason}>
                <td className="p-3 font-semibold text-white">{r.model}</td>
                <td className="p-3">{r.params_b.toFixed(1)}B</td>
                <td className="p-3">{r.min_ram_gb.toFixed(1)} GB</td>
                <td className="p-3">{r.rec_vram_gb.toFixed(1)} GB</td>
                <td className="p-3 text-slate-400">{r.family}</td>
                <td className={`p-3 font-medium ${fitColor(r.fit)}`}>
                  <span className="flex items-center gap-1.5">
                    <FitIcon fit={r.fit} />
                    {r.fit}
                    {r.notes && (
                      <span className="text-slate-500 text-[10px]">({r.notes})</span>
                    )}
                  </span>
                </td>
              </tr>
            ))}
            {!data?.rows?.length && (
              <tr>
                <td colSpan={6} className="p-6 text-center text-slate-500">
                  {busy ? "Loading…" : error || "No catalog rows."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <Card padded>
        <CardHeader title="Local-First vs Cloud Trade-offs" eyebrow="Matrix" />
        <div className="grid grid-cols-2 gap-3 text-xs">
          {data?.tradeoffs?.map((t) => (
            <div
              key={t.dimension}
              className="bg-slate-950 p-3 rounded-lg border border-white/5 flex justify-between items-center gap-3"
            >
              <div className="min-w-0">
                <p className="text-slate-200 font-medium truncate">{t.dimension}</p>
                <p className="text-[10px] text-slate-500 font-mono mt-0.5 truncate">
                  Local: {t.local}
                </p>
                <p className="text-[10px] text-slate-500 font-mono truncate">
                  Cloud: {t.cloud}
                </p>
              </div>
              <span
                className={`uppercase text-[10px] font-bold px-2 py-0.5 rounded border font-mono whitespace-nowrap ${winnerClasses(t.winner)}`}
              >
                {t.winner}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {error && (
        <Card padded className="border-red-500/40">
          <p className="text-xs text-red-300 font-mono flex items-center gap-2">
            <RefreshCcw className="h-3 w-3" /> {error}
          </p>
        </Card>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "emerald" | "slate";
}) {
  return (
    <Card padded className={tone === "emerald" ? "border-bright" : undefined}>
      <p className="av-section-eyebrow mb-1">{label}</p>
      <p
        className={`text-xl font-bold font-mono ${tone === "emerald" ? "text-emerald-400" : "text-slate-200"}`}
      >
        {value}
      </p>
    </Card>
  );
}

function fitColor(fit: string): string {
  const f = fit.toLowerCase();
  if (f.includes("fits")) return "text-emerald-400";
  if (f.includes("tight")) return "text-amber-400";
  return "text-red-400";
}

function FitIcon({ fit }: { fit: string }) {
  const f = fit.toLowerCase();
  if (f.includes("fits"))
    return <CheckCircle className="h-3.5 w-3.5 text-emerald-400 shrink-0" />;
  if (f.includes("tight"))
    return <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0" />;
  return <XCircle className="h-3.5 w-3.5 text-red-400 shrink-0" />;
}

function winnerClasses(winner: string): string {
  const w = (winner || "").toLowerCase();
  if (w.includes("local"))
    return "text-emerald-400 bg-emerald-500/5 border-emerald-500/10";
  if (w.includes("cloud"))
    return "text-purple-400 bg-purple-500/5 border-purple-500/10";
  return "text-slate-400 bg-slate-500/5 border-white/5";
}
