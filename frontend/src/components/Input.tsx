import type { InputHTMLAttributes, TextareaHTMLAttributes } from "react";
import { cn } from "../lib/utils";

export interface FieldProps {
  label?: string;
  hint?: string;
  className?: string;
}

export function Field({
  label,
  hint,
  className,
  children,
}: FieldProps & { children: React.ReactNode }) {
  return (
    <div className={className}>
      {label && <label className="av-label">{label}</label>}
      {children}
      {hint && <p className="mt-1 text-[10px] text-slate-500 font-mono">{hint}</p>}
    </div>
  );
}

export function TextInput({
  className,
  ...rest
}: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn("av-input", className)} {...rest} />;
}

export function TextArea({
  className,
  ...rest
}: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn("av-input resize-y min-h-[80px] leading-snug", className)}
      {...rest}
    />
  );
}

export function NumberInput({
  className,
  ...rest
}: InputHTMLAttributes<HTMLInputElement>) {
  return <input type="number" className={cn("av-input", className)} {...rest} />;
}

export function Select({
  className,
  children,
  ...rest
}: InputHTMLAttributes<HTMLSelectElement> & { children: React.ReactNode }) {
  return (
    <select className={cn("av-input appearance-none pr-8", className)} {...rest as any}>
      {children}
    </select>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex items-center gap-2 text-xs text-slate-300 hover:text-white transition"
    >
      <span
        className={cn(
          "relative h-4 w-7 rounded-full transition",
          checked ? "bg-emerald-500" : "bg-slate-700",
        )}
      >
        <span
          className={cn(
            "absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all",
            checked ? "left-3.5" : "left-0.5",
          )}
        />
      </span>
      <span className="font-mono text-[11px]">{label}</span>
    </button>
  );
}
