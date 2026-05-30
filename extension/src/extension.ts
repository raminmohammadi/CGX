// Minimal VS Code extension that hosts the Averix Gradio UI inside a
// webview panel. The extension does NOT spawn the server; the user is
// expected to run `averix-ui` (or `python app.py`) separately. The URL
// is read from the `averix.ui.url` setting (default http://localhost:7860).

import * as vscode from "vscode";

let currentPanel: vscode.WebviewPanel | undefined;

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("averix.openUI", () => openOrFocus(context)),
    vscode.commands.registerCommand("averix.refreshUI", () => {
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
  const cfg = vscode.workspace.getConfiguration("averix");
  return (cfg.get<string>("ui.url") || "http://localhost:7860").trim();
}

function openOrFocus(context: vscode.ExtensionContext): void {
  if (currentPanel) {
    currentPanel.reveal(vscode.ViewColumn.Active);
    return;
  }
  const url = currentUrl();
  currentPanel = vscode.window.createWebviewPanel(
    "averixUI",
    "Averix",
    vscode.ViewColumn.Active,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
      // The Gradio server is reached over plain HTTP on localhost; we
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
  // The Gradio UI is served as a full HTML document, so we frame it as
  // an iframe filling the panel. We escape the URL so a malicious
  // setting value can't break out of the attribute.
  const safe = url.replace(/"/g, "&quot;");
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
        Averix needs scripts enabled. Configure the Gradio URL via the
        <code>averix.ui.url</code> setting and run
        <code>averix-ui</code> in a terminal first.
      </div>
    </noscript>
  </body>
</html>`;
}
