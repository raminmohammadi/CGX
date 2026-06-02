# CGX — VS Code extension scaffold

Hosts the CGX web UI inside a VS Code webview so you can ask
questions about the open workspace without leaving the editor.

This directory is a **scaffold only** — it is not packaged into a
`.vsix` from the repo. Build it locally with the steps below.

## What it does

- Registers two commands:
  - **CGX: Open UI** — opens a webview panel pointing at the running
    CGX server.
  - **CGX: Reload UI** — re-renders the iframe (useful after
    restarting the CGX server).
- Reads the server URL from `cgx.ui.url` (default
  `http://localhost:8765`).
- Does **not** spawn the server. Start it first with `cgx-ui` (or
  `python app.py`) from the repo root. `cgx-ui` resolves to
  `cgx.webui.launch:launch` and serves the React frontend + FastAPI
  backend (Ask / Plan / Agent / Setup / Index pages, the
  `/api/tasks` registry, and the `/api/rollback` undo endpoint) on
  port 8765.

## Layout

```
extension/
├── package.json        # manifest, commands, settings
├── tsconfig.json       # strict TypeScript config, ES2021 / CommonJS
├── src/extension.ts    # activate / deactivate + webview rendering
└── README.md           # this file
```

## Local build

Requires Node ≥ 18 and the VS Code extension generator toolchain.

```bash
cd extension
npm install
npm run compile          # emits out/extension.js
```

Open the `extension/` folder in VS Code and press **F5** to launch an
Extension Development Host with the extension activated. Run the
**CGX: Open UI** command from the command palette.

## Packaging into a .vsix (optional)

```bash
npm install -g @vscode/vsce
vsce package
```

This produces `cgx-0.0.1.vsix` which can be side-loaded via
**Extensions: Install from VSIX…** in VS Code.

## Privacy

The extension only loads the URL configured in `cgx.ui.url`. No
telemetry is emitted by this scaffold. The CGX server itself
processes everything locally; see the top-level project README for the
full privacy posture.
