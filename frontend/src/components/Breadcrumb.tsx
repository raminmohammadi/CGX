import { ChevronRight } from "lucide-react";
import { Fragment } from "react";

export function Breadcrumb({ items }: { items: string[] }) {
  return (
    <div className="flex items-center gap-1.5 text-xs font-mono text-slate-500">
      {items.map((label, i) => (
        <Fragment key={i}>
          {i > 0 && <ChevronRight className="h-3 w-3 text-slate-700 shrink-0" />}
          <span className={i === items.length - 1 ? "text-slate-200 font-medium" : undefined}>
            {label}
          </span>
        </Fragment>
      ))}
    </div>
  );
}
