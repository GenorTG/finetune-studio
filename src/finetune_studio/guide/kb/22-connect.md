---
id: connect
title: Connect an exported Smart Brain
page: connect
keywords: connect mcp stdio http claude cursor copilot windsurf gemini lm studio open webui ollama chatgpt keyfile passphrase bearer health
---
## Purpose
Connect the exported knowledge package to an MCP host. The package retrieves from documents; it is not an LLM.
Connect is a local recipe builder, not a deployment or a client-connection status dashboard.
New exported packages also host their own /connect page, linked from the package chat page.
It includes the same recipes, a same-server health/MCP/tools/rag_info check and a Linux MCPB bridge download.
The check's bearer token stays in memory and is sent only to this package, never to the editable recipe address.
It checks browser-to-package transport, not the selected vendor client. HTTP access tokens require HTTPS off-loopback.

## Workflow
1. Build and download a package on a project's Package page; extract it and run bash install.sh.
2. Open Connect from the navigation or Package page. Choose the AI tool and edit the absolute package and passphrase-file paths.
3. Copy setup, merge it into the client's configuration and enable the server. VS Code uses servers; the other supplied stdio JSON recipes use mcpServers.
4. HTTP clients use run-http.sh and POST /mcp. Remote binding requires RAG_AUTH_TOKEN and HTTPS; the token is separate from the passphrase.
5. Copy terminal checks and run them from the client machine. GET /health must have ok true; MCP initialize must have result.serverInfo.
6. In the client, call rag_info to confirm the package, then rag_search with a known fact. Only this proves that client's tool access.

## Key controls
- `#connect-builder` — client and local-only configuration fields.
- `#connect-client` — client choice.
- `#connect-path` — extracted package folder on the client machine.
- `#connect-keyfile` — passphrase file path, never the passphrase.
- `#connect-url` — package HTTP address, not studio port 7860.
- `#connect-copy` — copies setup; reports clipboard success or manual-copy fallback.
- `#connect-checks` — startup and terminal connection checks.
- `#connect-copy-test` — copies terminal checks, without executing them.

## What to check
Edits update recipes immediately and stay in the browser; no keys or addresses are sent to the studio.
The studio makes no probes. The package rejects foreign browser Origins; its own Connect page can test the same server.
There is no arbitrary server-side URL proxy or test against an address entered into the recipe builder.
rag_get_document requires packaged originals; rag_answer requires answer generation and an operator-configured LLM.

## Common mistakes
There is no package /v1/chat/completions endpoint. Ollama needs an MCP host such as Open WebUI.
ChatGPT is not a ready-made Actions/OAuth/tunnel integration here.
The unsigned smart-brain.mcpb bridge selects an already installed Linux package and private keyfile; it contains no knowledge or keys.
It requires Node >=18 and an MCPB-capable Linux host; it does not install Python dependencies.
Claude Desktop's official MCPB hosts are macOS/Windows, outside the current Linux-only scope: one-click Claude Desktop installation remains unverified and unsupported.
Installed Claude Code 2.1.325 passed an isolated real MCP health check on 2026-11-18 with a synthetic package, not model-driven tool use.
Gemini CLI 0.65.0 accepted isolated settings but requires client authentication to check the connection.
Cursor, VS Code Copilot, Windsurf, LM Studio and Open WebUI have no actual-client certification yet.
Windsurf's current version-specific setup is explicitly unverified; consult its installed-version documentation.
127.0.0.1 in Open WebUI's container means that container, not the host. 401 on /health means missing or wrong package bearer token.
