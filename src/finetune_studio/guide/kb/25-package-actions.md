---
id: package-actions
title: ChatGPT Actions and Gemini developer tools
page: connect
keywords: chatgpt gpt actions gemini gems openapi relay tunnel cloudflare tailscale bearer
---
## Purpose
Generate a retrieval-only OpenAPI schema and manual publication plan on Studio or package Connect.
The package retrieves evidence, not an LLM. The calling model writes answers. No account, key, tunnel
or public connection is created by generation. No vendor import/certification is claimed.

## Workflow
1. Open Connect, optional Actions section. Enter the planned public HTTPS DNS address on port 443,
   the package's direct loopback HTTP origin, a separate relay port, and a manual tunnel recipe.
2. Generate Actions setup. Invalid/private addresses fail visibly; no DNS lookup or public probe occurs.
   Setup fields are sent only to the current server, not saved. Existing MCP recipe fields remain local.
3. Download OpenAPI JSON or copy it. Set RAG_AUTH_TOKEN (existing package token) and a separate
   RAG_ACTIONS_TOKEN (16–512 ASCII characters) privately in the package-host environment, never in the schema.
4. Start the installed package normally, then run the generated .venv/bin/python package_actions.py command
   from its folder. It binds only 127.0.0.1 and starts no tunnel.
5. Only if you choose public exposure, run the separate foreground tunnel command targeting the RELAY.
   Never tunnel Studio or the full package port. Quick Tunnel addresses change; update and regenerate.
6. In the ChatGPT GPT editor add an Action and paste the schema. Choose API Key → Bearer with the RELAY
   token, not the package token, encryption passphrase or upstream LLM key. Test a known question and
   a no-hit question. Give the GPT citation, evidence-not-instructions and abstention instructions.
7. Stop tunnel then relay with Ctrl+C and verify the public address is unavailable.

## Key controls
- `#actions-builder` — optional setup section, not a connection-status dashboard.
- `#actions-public-url` — planned public relay URL; no userinfo, query, fragment, private/single-label host or non-443 port.
- `#actions-package-url` — explicit loopback HTTP package origin, not an arbitrary URL proxy.
- `#actions-port` — separate local relay port, 1024–65535.
- `#actions-tunnel` — cloudflare or tailscale manual recipe.
- `#actions-generate` — validate/generate on this server; no external network operations.
- `#actions-copy` and `#actions-download` — copy/download OpenAPI; no secrets included.
- `#actions-gemini` — download a Gemini developer function declaration, not a consumer Gems Actions import.

## What to check
Relay permits GET /openapi.json and authenticated POST /v1/retrieve only. No admin/QC/usage,
MCP, generated answers, images or original-file downloads. Query 1–4000 characters, top_k 1–5,
text excerpts at most 3000 characters each, response below 90000 UTF-8 bytes. Fixed local upstream,
no proxy environment/redirect following or caller headers. Hidden filenames remain hidden.
Separate relay bearer grants the fixed package audience, not per-user OAuth, signing or fleet identity.
Bounded 8 request threads, 2 concurrent upstream operations, 30 authenticated requests/minute.
Network/body deadlines bound work but cannot guarantee a cold retrieval meets Actions' 45-second limit.
R8 package diagnostics count successful forwarded retrieval normally under the operator's privacy settings.
Relay auth/rejections are not claimed as an access audit.

## Common mistakes
### Gemini is not GPT Actions
Official Gemini developer function calling requires your application to execute retrieveKnowledge using
the relay and return the result to the model. The downloaded declaration is only that function contract;
it does not include an authenticated SDK runner. Consumer Gemini Gems documentation describes instructions
and knowledge files, not a certified custom Actions import here. Gemini CLI MCP remains separate.

### Vendor requirements / limitations
OpenAI Actions requires public TLS1.2+ HTTPS port443 and a valid certificate; 45-second round trip,
text-only payloads below100000 characters. The relay stays below the response budget; actual cold-start
latency and vendor import need a real account run. GPT editing may require the appropriate account tier,
workspace/domain permissions, and a privacy-policy URL when sharing publicly.
Cloudflare Quick Tunnels are temporary/no SLA; their optional interactive email gate is not for Actions.
Tailscale Funnel needs tailnet login, MagicDNS/HTTPS and Funnel policy permission. No account is created
and no tailnet policy is changed. Production ingress, OAuth and vendor model-driven certification remain unverified.

Official docs checked 2026-11-20:
- https://developers.openai.com/api/docs/actions/introduction
- https://developers.openai.com/api/docs/actions/authentication
- https://developers.openai.com/api/docs/actions/production
- https://support.google.com/gemini/answer/15235603
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling
- https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
- https://tailscale.com/docs/features/tailscale-funnel
