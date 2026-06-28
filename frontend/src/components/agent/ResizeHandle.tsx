import { useCallback, useRef } from "react";
import { cn } from "../../lib/utils";

// Thin vertical drag handle for resizing a sibling panel.
//
// ``direction`` says which side of the handle the resized panel sits on:
//   "left"  -> the panel is to the left of this handle; dragging right
//              increases its width.
//   "right" -> the panel is to the right of this handle; dragging right
//              decreases its width.
//
// ``getCurrent`` returns the width at pointer-down so the delta is added
// to a stable baseline (avoids drift when the parent re-renders during
// the drag).
export function ResizeHandle({
  direction,
  getCurrent,
  onResize,
  ariaLabel,
}: {
  direction: "left" | "right";
  getCurrent: () => number;
  onResize: (next: number) => void;
  ariaLabel?: string;
}) {
  const baseRef = useRef(0);
  const startRef = useRef(0);

  const onPointerMove = useCallback((ev: PointerEvent) => {
    const delta = ev.clientX - startRef.current;
    const signed = direction === "left" ? delta : -delta;
    onResize(baseRef.current + signed);
  }, [direction, onResize]);

  const onPointerUp = useCallback(() => {
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, [onPointerMove]);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    baseRef.current = getCurrent();
    startRef.current = e.clientX;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  }, [getCurrent, onPointerMove, onPointerUp]);

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={ariaLabel}
      onPointerDown={onPointerDown}
      className={cn(
        "shrink-0 w-1 cursor-col-resize select-none",
        "bg-transparent hover:bg-emerald-500/40 active:bg-emerald-500/60",
        "transition-colors",
      )}
    />
  );
}
