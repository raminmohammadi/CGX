// Minimal VS Code extension that hosts the CGX web UI inside a
// webview panel. The extension does NOT spawn the server; the user is
// expected to run `cgx-ui` (or `python app.py`) separately. The URL
// is read from the `cgx.ui.url` setting (default http://localhost:8765).

import * as vscode from "vscode";

let currentPanel: vscode.WebviewPanel | undefined;

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("cgx.openUI", () => openOrFocus(context)),
    vscode.commands.registerCommand("cgx.refreshUI", () => {
      if (currentPanel) {
        currentPanel.webview.html = renderHtml(currentUrl());
      }
    })
  );
}

export function deactivate(): void {
  currentPanel?.dispose();
  currentPanel = undefined;
}

function currentUrl(): string {
  const cfg = vscode.workspace.getConfiguration("cgx");
  return (cfg.get<string>("ui.url") || "http://localhost:8765").trim();
}

function openOrFocus(context: vscode.ExtensionContext): void {
  if (currentPanel) {
    // Reload the framed UI, don't just reveal it: the served bundle may have
    // changed (a rebuild) or the server may have restarted since the panel was
    // opened, and the webview never refreshes on its own -- retainContextWhenHidden
    // keeps the old document alive. Re-rendering re-fetches a fresh page.
    currentPanel.webview.html = renderHtml(currentUrl());
    currentPanel.reveal(vscode.ViewColumn.Active);
    return;
  }
  const url = currentUrl();
  currentPanel = vscode.window.createWebviewPanel(
    "cgxUI",
    "CGX",
    vscode.ViewColumn.Active,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
      // The CGX server is reached over plain HTTP on localhost; we
      // do not load any local resources, so no localResourceRoots are
      // declared (an empty list would block the iframe outright).
    }
  );
  currentPanel.webview.html = renderHtml(url);
  currentPanel.onDidDispose(() => {
    currentPanel = undefined;
  }, null, context.subscriptions);
}

function renderHtml(url: string): string {
  // The CGX UI is served as a full HTML document, so we frame it as
  // an iframe filling the panel. A cache-busting query param forces the
  // iframe to load a fresh document on every render (open / refresh), so a
  // rebuilt bundle or a restarted server is never masked by a retained
  // webview. We escape the URL so a malicious setting value can't break out
  // of the attribute.
  const busted = url + (url.includes("?") ? "&" : "?") + "_cgx=" + Date.now();
  const safe = busted.replace(/"/g, "&quot;");
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy"
          content="default-src 'none'; frame-src http://localhost:* http://127.0.0.1:*; style-src 'unsafe-inline';" />
    <style>
      html, body { margin: 0; padding: 0; height: 100%; background: #1e1e1e; color: #ddd; font-family: system-ui, sans-serif; }
      iframe { border: 0; width: 100%; height: 100%; display: block; }
      .err { padding: 1rem; }
      .err code { background: #2a2a2a; padding: 0.1rem 0.4rem; border-radius: 3px; }
    </style>
  </head>
  <body>
    <iframe src="${safe}"
            sandbox="allow-scripts allow-same-origin allow-forms allow-downloads"
            referrerpolicy="no-referrer"></iframe>
    <noscript>
      <div class="err">
        CGX needs scripts enabled. Configure the web UI URL via the
        <code>cgx.ui.url</code> setting and run
        <code>cgx-ui</code> in a terminal first.
      </div>
    </noscript>
  </body>
</html>`;
}
