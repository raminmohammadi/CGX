import { Component, type ReactNode } from "react";

interface Props {
  /** Rendered when a child throws during render or in a lifecycle. */
  fallback?: (error: Error, reset: () => void) => ReactNode;
  /** Human-readable label rendered in the default fallback. */
  label?: string;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

// Local error boundary. Wraps a subtree so a failing renderer surfaces
// the error message inline instead of unmounting the whole page (the
// React default behaviour for uncaught exceptions during render). Each
// instance owns its own ``reset`` so the user can retry without a full
// reload.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: { componentStack?: string }): void {
    // Keep a single console line so the original stack is still
    // discoverable through DevTools; the inline fallback handles the UX.
    // eslint-disable-next-line no-console
    console.error("[ErrorBoundary]", this.props.label || "render error",
      error, info.componentStack);
  }

  reset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    if (error === null) return this.props.children;
    if (this.props.fallback) return this.props.fallback(error, this.reset);
    return (
      <div className="rounded-lg border border-red-500/30 bg-red-950/30 p-3 my-2 text-[11px] font-mono text-red-200 space-y-1">
        <p className="text-red-300 uppercase tracking-wider text-[10px]">
          render error{this.props.label ? ` · ${this.props.label}` : ""}
        </p>
        <pre className="whitespace-pre-wrap break-words">
          {error.message || String(error)}
        </pre>
        <button
          type="button" onClick={this.reset}
          className="text-emerald-300 hover:text-emerald-200 underline underline-offset-2"
        >retry</button>
      </div>
    );
  }
}
