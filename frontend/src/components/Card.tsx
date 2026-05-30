import { type HTMLAttributes, type ReactNode } from "react";
import { cn } from "../lib/utils";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  active?: boolean;
  padded?: boolean;
}

export function Card({
  className,
  active,
  padded = true,
  children,
  ...rest
}: CardProps) {
  return (
    <div
      className={cn("av-card", padded && "p-5", active && "is-active", className)}
      {...rest}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  eyebrow,
  title,
  right,
  description,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  right?: ReactNode;
  description?: ReactNode;
}) {
  return (
    <div className="flex justify-between items-start gap-4 mb-4">
      <div>
        {eyebrow && <div className="av-section-eyebrow mb-1">{eyebrow}</div>}
        <h2 className="text-lg font-bold text-white tracking-tight">{title}</h2>
        {description && (
          <p className="text-xs text-slate-400 mt-0.5">{description}</p>
        )}
      </div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}
