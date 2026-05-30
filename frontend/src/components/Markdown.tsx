import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/atom-one-dark.css";

// Shared markdown renderer for streamed LLM output. Code fences pick up
// hljs highlighting; tables/strikethrough/etc. come from GFM.

export function Markdown({ text }: { text: string }) {
  return (
    <div className="text-sm text-slate-300 leading-relaxed space-y-3
                    [&_h1]:text-base [&_h1]:font-bold [&_h1]:text-white
                    [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-white [&_h2]:mt-4
                    [&_h3]:text-xs [&_h3]:font-semibold [&_h3]:text-emerald-300 [&_h3]:uppercase [&_h3]:tracking-wider
                    [&_a]:text-emerald-400 [&_a:hover]:underline
                    [&_strong]:text-white
                    [&_code]:font-mono [&_code]:text-[12px] [&_code]:text-pink-300
                          [&_code]:bg-slate-950 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded
                    [&_pre]:bg-slate-950 [&_pre]:border [&_pre]:border-muted
                          [&_pre]:rounded-lg [&_pre]:p-3 [&_pre]:text-[11px]
                          [&_pre]:overflow-x-auto [&_pre]:font-mono
                    [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-slate-300
                    [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:space-y-1
                    [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:space-y-1
                    [&_blockquote]:border-l-2 [&_blockquote]:border-emerald-500/40
                          [&_blockquote]:pl-3 [&_blockquote]:text-slate-400 [&_blockquote]:italic
                    [&_table]:w-full [&_table]:text-xs [&_table]:font-mono [&_table]:border-collapse
                    [&_th]:text-left [&_th]:p-2 [&_th]:border-b [&_th]:border-muted [&_th]:text-slate-400
                    [&_td]:p-2 [&_td]:border-b [&_td]:border-muted/50">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
