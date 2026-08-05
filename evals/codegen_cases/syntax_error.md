## Plan

Add a helper module (this intentionally contains a Python syntax error).

```diff path=pkg/broken.py
--- /dev/null
+++ b/pkg/broken.py
@@ -0,0 +1,2 @@
+def broken(:
+    return
```
