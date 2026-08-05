import { useEffect, useState } from "react";
import type { PillTone } from "../../components/Pill";

// Shared props every Ops section receives from the hub. ``refreshKey`` bumps
// on manual/auto refresh; ``goto`` lets a section deep-link to another tab.
export type SectionProps = { refreshKey: number; goto?: (tab: string) => void };

export function fmtCost(v?: number | null) {
  return v == null ? "--" : `$${Number(v).toFixed(4)}`;
}
export function fmtNum(v?: number | null) {
  return v == null ? "--" : Number(v).toLocaleString();
}
export function fmtMs(v?: number | null) {
  return v == null ? "--" : `${Math.round(v)}ms`;
}
export function fmtPct(v?: number | null) {
  return v == null ? "--" : `${Math.round(v * 100)}%`;
}

export const SEV_TONE: Record<string, PillTone> = {
  critical: "red",
  warning: "amber",
  info: "slate",
};

// Small data-fetch hook: runs ``loader`` whenever ``deps`` change, tracking
// loading/error and guarding against setState after unmount.
export function useAsync<T>(
  loader: () => Promise<T>,
  deps: unknown[],
): { data: T | null; loading: boolean; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    loader()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e?.message || e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, loading, error };
}

export function ErrorLine({ error }: { error: string | null }) {
  if (!error) return null;
  return <p className="text-xs text-red-400 font-mono">{error}</p>;
}
