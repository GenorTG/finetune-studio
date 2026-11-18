---
id: connect
title: Connect an exported Smart Brain
page: connect
keywords: connect mcp stdio http claude cursor copilot windsurf gemini lm studio open webui ollama chatgpt keyfile passphrase bearer health openai compatible retrieve completions
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
4. HTTP clients use run-http.sh and POST /mcp. The startup command passes --port explicitly: blank listener port follows a direct localhost HTTP URL (including a nondefault port). For HTTPS, public URLs or proxy paths, enter the package's internal listener port (usually 8899); it cannot be inferred from a public proxy URL. If localhost is itself a proxy, enter the internal port explicitly. This changes startup only, never client URLs/checks. Remote binding requires RAG_AUTH_TOKEN and HTTPS; the token is separate from the passphrase.
5. Copy terminal checks and run them from the client machine. GET /health must have ok true; MCP initialize must have result.serverInfo.
6. In the client, call rag_info to confirm the package, then rag_search with a known fact. Only this proves that client's tool access.

## Key controls
- `#connect-builder` — client and local-only configuration fields.
- `#connect-client` — client choice.
- `#connect-path` — extracted package folder on the client machine.
- `#connect-keyfile` — passphrase file path, never the passphrase.
- `#connect-url` — package HTTP address, not studio port 7860.
- `#connect-port` — internal listener port; blank follows direct local HTTP, not public/proxy URLs.
- `#connect-copy` — copies setup; reports clipboard success or manual-copy fallback.
- `#connect-checks` — startup and terminal connection checks.
- `#connect-copy-test` — copies terminal checks, without executing them.

## What to check
Edits update recipes immediately and stay in the browser; no keys or addresses are sent to the studio.
The studio makes no probes. The package rejects foreign browser Origins; its own Connect page can test the same server.
There is no arbitrary server-side URL proxy or test against an address entered into the recipe builder.
rag_get_document requires packaged originals; rag_answer requires answer generation and an operator-configured LLM.

## Common mistakes
New exports expose an OpenAI-compatible text-chat subset: choose knowledge-api in the client list.
GET /v1/models advertises smart-brain; POST /v1/retrieve accepts query, top_k (1–100), images (boolean).
It returns scoped passages with scores and optional pictures without an LLM. No caller may widen the package scope.
POST /v1/chat/completions accepts model smart-brain, text messages, stream (boolean), n=1,
temperature (0–2), top_p (0–1), and max_tokens OR max_completion_tokens (bounded by operator answer_max_tokens).
The last turn must be a user question; at most 32 messages, each 1–16000 characters. System/developer text is
client guidance subordinate to package identity, answer rules and evidence. Tools, attachments, usage accounting,
stream_options and unsupported parameters fail with 400; this is not the entire OpenAI API.
JSON responses attach knowledge.sources, images, weak, sources_hidden and generated. generated=false means a
retrieval miss, not an LLM answer. Stream=true uses buffered SSE: answer chunk, final chunk with knowledge, [DONE].
There is no token-by-token upstream stream and no invented token usage. File-name hiding cannot be overridden by clients.
The operator sets RAG_LLM_PROVIDER (openai or anthropic), RAG_LLM_BASE_URL, RAG_LLM_MODEL, optionally
RAG_LLM_API_KEY, then starts run-http.sh --answer. Existing OpenAI-compatible/Claude adapters are reused;
the incoming client never chooses a provider URL, model, key, body or headers. No provider returns 503;
upstream failure returns a sanitized 502. Keys stay on the package host, never in the exported recipe or client setup.
Clients use the package address with /v1 and its RAG_AUTH_TOKEN, not the provider API key or passphrase.
The package Connect page has Test retrieval API and Test answer API with immediate status and displayed JSON.
Only these same-server buttons make requests; the studio recipe builder still makes no network probes.
Ollama can be the operator's upstream via its OpenAI-compatible /v1 API, or use an MCP host such as Open WebUI.
ChatGPT is not a ready-made Actions/OAuth/tunnel integration here.
The unsigned smart-brain.mcpb bridge selects an already installed Linux package and private keyfile; it contains no knowledge or keys.
It requires Node >=18 and an MCPB-capable Linux host; it does not install Python dependencies.
Claude Desktop's official MCPB hosts are macOS/Windows, outside the current Linux-only scope: one-click Claude Desktop installation remains unverified and unsupported.
Installed Claude Code 2.1.325 (claude-sonnet-5-5) passed three actual model-driven synthetic MCP sessions on 2026-11-18: plaintext direct stdio, encrypted direct stdio/keyfile, and encrypted exported MCPB Node bridge/keyfile. Each called rag_info and rag_search and answered the Highmere price with its returned source. This certifies those three text paths on one Linux host, not HTTP/images, clean-machine installation, Claude Desktop or all vendor clients.
Gemini CLI 0.65.0 accepted isolated settings but requires client authentication to check the connection.
Cursor, VS Code Copilot, Windsurf, LM Studio and Open WebUI have no actual-client certification yet.
Windsurf's current version-specific setup is explicitly unverified; consult its installed-version documentation.
127.0.0.1 in Open WebUI's container means that container, not the host. 401 on /health means missing or wrong package bearer token.
