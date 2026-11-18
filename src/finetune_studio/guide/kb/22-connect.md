---
id: connect
title: Connect an exported Smart Brain
page: connect
keywords: connect mcp stdio http claude cursor copilot windsurf gemini lm studio open webui ollama chatgpt keyfile passphrase bearer health
---
## Purpose
Connect the exported knowledge package to an MCP host. The package retrieves from documents; it is not an LLM.
Connect is a local recipe builder, not a deployment or a client-connection status dashboard.

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
The package rejects foreign browser Origins, so there is deliberately no misleading browser reachability test or arbitrary server-side URL proxy.
rag_get_document requires packaged originals; rag_answer requires answer generation and an operator-configured LLM.

## Common mistakes
There is no package /v1/chat/completions endpoint. Ollama needs an MCP host such as Open WebUI.
ChatGPT is not a ready-made Actions/OAuth/tunnel integration here. One-click .mcpb installation is not shipped.
Windsurf's current version-specific setup is explicitly unverified; consult its installed-version documentation.
127.0.0.1 in Open WebUI's container means that container, not the host. 401 on /health means missing or wrong package bearer token.
