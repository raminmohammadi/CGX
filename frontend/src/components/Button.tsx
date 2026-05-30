import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "../lib/utils";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost" | "danger";
  icon?: ReactNode;
}

export function Button({
  variant = "ghost",
  icon,
  children,
  className,
  ...rest
}: ButtonProps) {
  const base =
    variant === "primary"
      ? "av-btn-primary"
      : variant === "danger"
        ? "av-btn bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20"
        : "av-btn-ghost";
  return (
    <button className={cn(base, className)} {...rest}>
      {icon}
      {children}
    </button>
  );
}
