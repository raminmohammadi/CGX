import{i as e,n as t,r as n,t as r}from"./markdown-Bd97_IbF.js";var i=e(),a=`cite:`;function o(e){let t=e.trim(),n=t.split(`::`);if(n.length>=3)return n[n.length-1];if(n.length===2)return n[1]||n[0];let r=t.indexOf(` [`),i=r>=0?t.slice(0,r):t,a=i.lastIndexOf(`/`);return a>=0?i.slice(a+1):i}function s(e){if(!e)return``;let t=e.replace(/\[\[([^\[\]]+)\]\]/g,(e,t)=>`[${o(String(t))}](${a}${encodeURIComponent(String(t))})`);return t=t.replace(/\[([^\[\]]*::[^\[\]]+)\](?!\()/g,(e,t)=>`[${o(String(t))}](${a}${encodeURIComponent(String(t))})`),t}function c({id:e,children:t}){return(0,i.jsxs)(`span`,{title:e,className:`inline-flex items-baseline gap-1 align-baseline mx-0.5 px-1.5 py-0.5 rounded
                 font-mono text-[10.5px] leading-none
                 bg-emerald-500/10 text-emerald-300 border border-emerald-500/25
                 hover:bg-emerald-500/20 hover:text-emerald-200
                 transition cursor-help no-underline`,children:[(0,i.jsx)(`span`,{className:`text-emerald-500/70`,children:`§`}),(0,i.jsx)(`span`,{className:`truncate max-w-[18ch]`,children:t})]})}function l({text:e}){let o=s(e||``);return(0,i.jsx)(`div`,{className:`text-[14px] text-slate-200 leading-[1.7] space-y-3.5
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
                    [&_td]:p-2 [&_td]:border-b [&_td]:border-white/5`,children:(0,i.jsx)(n,{remarkPlugins:[t],rehypePlugins:[r],components:{a({href:e,children:t,...n}){if(typeof e==`string`&&e.startsWith(a)){let n=decodeURIComponent(e.slice(5));return(0,i.jsx)(c,{id:n,children:t})}return(0,i.jsx)(`a`,{href:e,target:`_blank`,rel:`noreferrer noopener`,...n,children:t})}},children:o})})}export{l as t};