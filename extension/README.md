# Averix — VS Code extension scaffold

Hosts the Averix Gradio UI inside a VS Code webview so you can ask
questions about the open workspace without leaving the editor.

This directory is a **scaffold only** — it is not packaged into a
`.vsix` from the repo. Build it locally with the steps below.

## What it does

- Registers two commands:
  - **Averix: Open UI** — opens a webview panel pointing at the running
    Averix server.
  - **Averix: Reload UI** — re-renders the iframe (useful after
    restarting the Gradio server).
- Reads the server URL from `averix.ui.url` (default
  `http://localhost:7860`).
- Does **not** spawn the server. Start it first with `averix-ui` (or
  `python app.py`) from the repo root.

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
**Averix: Open UI** command from the command palette.

## Packaging into a .vsix (optional)

```bash
npm install -g @vscode/vsce
vsce package
```

This produces `averix-0.0.1.vsix` which can be side-loaded via
**Extensions: Install from VSIX…** in VS Code.

## Privacy

The extension only loads the URL configured in `averix.ui.url`. No
telemetry is emitted by this scaffold. The Gradio server itself
processes everything locally; see the top-level project README for the
full privacy posture.
