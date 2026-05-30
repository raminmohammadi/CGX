function g(t,o,a,s){const i=new AbortController,c=(async()=>{try{const e=await fetch(t,{method:"POST",headers:{"content-type":"application/json",accept:"text/event-stream"},body:JSON.stringify(o),signal:i.signal});if(!e.ok||!e.body)throw new Error(`SSE ${t} → ${e.status}`);const n=e.body.getReader(),u=new TextDecoder;let r="";for(;;){const{value:h,done:p}=await n.read();if(p)break;r+=u.decode(h,{stream:!0}).replace(/\r\n/g,`
`);let d;for(;(d=r.indexOf(`

`))>=0;){const b=r.slice(0,d);r=r.slice(d+2),f(b,a)}}r.trim()&&f(r,a)}catch(e){if(e?.name==="AbortError")return;s?.(e)}})();return{abort:()=>i.abort(),done:c}}function f(t,o){let a="message";const s=[];for(const e of t.split(`
`)){const n=e.replace(/\r$/,"");!n||n.startsWith(":")||(n.startsWith("event:")?a=n.slice(6).trim():n.startsWith("data:")&&s.push(n.slice(5).trimStart()))}if(!s.length)return;const i=s.join(`
`);let c=i;try{c=JSON.parse(i)}catch{}o(a,c)}const l=new Map;function m(t){return l.get(t)??null}function w(t,o){l.set(t,o)}function y(t){l.get(t)?.abort(),l.delete(t)}export{y as a,g as b,m as g,w as s};
