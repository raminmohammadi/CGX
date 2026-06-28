import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/atom-one-dark.css";

// Shared markdown renderer for streamed LLM output. Code fences pick up
// hljs highlighting; tables/strikethrough/etc. come from GFM. ``[[chunk_id]]``
// citation tokens emitted by the answer engine are rewritten into compact
// emerald chips so the prose reads cleanly instead of leaking raw ids.

const CITE_PREFIX = "cite:";

// Pull a short, human-readable label out of a chunk id. Chunk ids look like
// ``src/cgx/foo.py::function::name`` after the project-root prefix has been
// stripped server-side; fall back to ``basename`` for free-form refs like
// ``README [lead]``.
function citationLabel(id: string): string {
  const trimmed = id.trim();
  const parts = trimmed.split("::");
  if (parts.length >= 3) return parts[parts.length - 1];
  if (parts.length === 2) return parts[1] || parts[0];
  const bracket = trimmed.indexOf(" [");
  const head = bracket >= 0 ? trimmed.slice(0, bracket) : trimmed;
  const slash = head.lastIndexOf("/");
  return slash >= 0 ? head.slice(slash + 1) : head;
}

// Rewrite citation tokens into markdown links with a ``cite:`` URL scheme.
// We pick this up below in the ``a`` component renderer to swap in a chip.
// Two shapes are handled: the documented ``[[id]]`` form and the single-
// bracket ``[path::kind::name]`` shape some models emit when they forget the
// double brackets. The single-bracket pass requires ``::`` and refuses to
// match real markdown links (those have ``(`` immediately after).
function preprocessCitations(text: string): string {
  if (!text) return "";
  let out = text.replace(/\[\[([^\[\]]+)\]\]/g, (_m, id) => {
    const label = citationLabel(String(id));
    return `[${label}](${CITE_PREFIX}${encodeURIComponent(String(id))})`;
  });
  out = out.replace(/\[([^\[\]]*::[^\[\]]+)\](?!\()/g, (_m, id) => {
    const label = citationLabel(String(id));
    return `[${label}](${CITE_PREFIX}${encodeURIComponent(String(id))})`;
  });
  return out;
}

function CitationChip({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <span
      title={id}
      className="inline-flex items-baseline gap-1 align-baseline mx-0.5 px-1.5 py-0.5 rounded
                 font-mono text-[10.5px] leading-none
                 bg-emerald-500/10 text-emerald-300 border border-emerald-500/25
                 hover:bg-emerald-500/20 hover:text-emerald-200
                 transition cursor-help no-underline"
    >
      <span className="text-emerald-500/70">§</span>
      <span className="truncate max-w-[18ch]">{children}</span>
    </span>
  );
}

export function Markdown({ text }: { text: string }) {
  const md = preprocessCitations(text || "");
  return (
    <div className="text-[14px] text-slate-200 leading-[1.7] space-y-3.5
                    [&>*:first-child]:mt-0 [&>*:last-child]:mb-0
                    [&_p]:my-0
                    [&_h1]:text-lg [&_h1]:font-bold [&_h1]:text-white [&_h1]:mt-5 [&_h1]:mb-2
                    [&_h2]:text-[15px] [&_h2]:font-semibold [&_h2]:text-white [&_h2]:mt-5 [&_h2]:mb-2
                          [&_h2]:pb-1 [&_h2]:border-b [&_h2]:border-white/5
                    [&_h3]:text-[11px] [&_h3]:font-semibold [&_h3]:text-emerald-300
                          [&_h3]:uppercase [&_h3]:tracking-[0.12em] [&_h3]:mt-4 [&_h3]:mb-1.5
                    [&_a:not(.cite)]:text-emerald-400 [&_a:not(.cite)]:underline [&_a:not(.cite)]:decoration-emerald-500/30
                          [&_a:not(.cite)]:underline-offset-2 [&_a:not(.cite):hover]:decoration-emerald-400
                    [&_strong]:text-white [&_strong]:font-semibold
                    [&_em]:text-slate-300
                    [&_code]:font-mono [&_code]:text-[12.5px] [&_code]:text-pink-300
                          [&_code]:bg-slate-950/80 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded
                          [&_code]:border [&_code]:border-white/5
                    [&_pre]:bg-slate-950 [&_pre]:border [&_pre]:border-white/5
                          [&_pre]:rounded-lg [&_pre]:p-4 [&_pre]:text-[12px]
                          [&_pre]:overflow-x-auto [&_pre]:font-mono [&_pre]:my-3
                          [&_pre]:shadow-inner
                    [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-slate-200
                          [&_pre_code]:border-0
                    [&_ul]:list-disc [&_ul]:pl-6 [&_ul]:space-y-1.5 [&_ul]:my-2
                    [&_ol]:list-decimal [&_ol]:pl-6 [&_ol]:space-y-1.5 [&_ol]:my-2
                    [&_li]:marker:text-emerald-500/60 [&_li]:pl-1
                    [&_li>p]:my-0
                    [&_blockquote]:border-l-2 [&_blockquote]:border-emerald-500/40
                          [&_blockquote]:pl-3 [&_blockquote]:text-slate-400 [&_blockquote]:italic
                    [&_hr]:border-white/5 [&_hr]:my-4
                    [&_table]:w-full [&_table]:text-[12.5px] [&_table]:border-collapse [&_table]:my-3
                    [&_th]:text-left [&_th]:p-2 [&_th]:border-b [&_th]:border-white/10 [&_th]:text-slate-400
                          [&_th]:font-mono [&_th]:text-[11px] [&_th]:uppercase [&_th]:tracking-wider
                    [&_td]:p-2 [&_td]:border-b [&_td]:border-white/5">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          a({ href, children, ...rest }) {
            if (typeof href === "string" && href.startsWith(CITE_PREFIX)) {
              const id = decodeURIComponent(href.slice(CITE_PREFIX.length));
              return <CitationChip id={id}>{children}</CitationChip>;
            }
            return (
              <a href={href} target="_blank" rel="noreferrer noopener" {...rest}>
                {children}
              </a>
            );
          },
        }}
      >
        {md}
      </ReactMarkdown>
    </div>
  );
}
