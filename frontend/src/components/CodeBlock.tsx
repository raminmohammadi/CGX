import { useEffect, useRef } from "react";
import hljs from "highlight.js/lib/common";
import "highlight.js/styles/atom-one-dark.css";

// Lightweight code block with hljs autodetect. We avoid pulling in the
// full markdown pipeline here since some pages need just one snippet.

export function CodeBlock({
  code,
  language,
  filename,
  className,
}: {
  code: string;
  language?: string;
  filename?: string;
  className?: string;
}) {
  const ref = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    ref.current.removeAttribute("data-highlighted");
    delete (ref.current.dataset as any).highlighted;
    try {
      hljs.highlightElement(ref.current);
    } catch {
      /* ignore */
    }
  }, [code, language]);
  return (
    <div
      className={
        "bg-slate-950 rounded-lg border border-muted overflow-hidden " +
        (className || "")
      }
    >
      {filename && (
        <div className="bg-slate-900 px-3 py-1.5 flex items-center justify-between text-[10px] text-slate-400 border-b border-white/5 font-mono">
          <span>{filename}</span>
          {language && <span className="text-slate-600">{language}</span>}
        </div>
      )}
      <pre className="p-3 overflow-x-auto text-[11px] leading-relaxed font-mono">
        <code ref={ref} className={language ? `language-${language}` : undefined}>
          {code}
        </code>
      </pre>
    </div>
  );
}
