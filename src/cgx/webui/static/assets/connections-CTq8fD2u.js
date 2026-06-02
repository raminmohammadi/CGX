const t=new Map;function e(n){return t.get(n)??null}function c(n,o){t.set(n,o)}function s(n){t.get(n)?.abort(),t.delete(n)}export{s as a,e as g,c as s};
