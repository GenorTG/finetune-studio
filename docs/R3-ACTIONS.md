# ChatGPT Actions and Gemini developer tools

New exported packages include a separate, stdlib-only retrieval relay and a Connect setup generator.
**The package is retrieval, not an LLM.** The cloud model supplies answers from returned text evidence.

## What is implemented

Studio and package **Connect → Optional · ChatGPT Actions and Gemini developer tools** validate a planned
public HTTPS URL, direct loopback package address and relay port. Generate, copy or download an OpenAPI3.1
schema, a Gemini developer function declaration, and reviewable foreground relay/tunnel commands.
Generation never performs DNS/public probes, saves settings, creates credentials/accounts, or launches processes.

The package's `package_actions.py` is an optional, separate relay. Set the package's existing
`RAG_AUTH_TOKEN` and a **different** `RAG_ACTIONS_TOKEN` privately in the host environment.
Start the installed package normally, then run the generated command from its directory.
The relay binds only 127.0.0.1. It forwards only text retrieval to the fixed direct loopback package origin.
It does not include an LLM, choose a provider, ingest documents or widen the package's audience/scope.

Allowed routes:
- `GET /openapi.json`: generic credential-free schema.
- `POST /v1/retrieve`: relay bearer required; JSON query 1–4000 characters and top_k 1–5 only.

Responses preserve hidden-source policy, return at most 5 excerpts of 3000 characters each, and reject
payloads over 90000 UTF-8 bytes. No images, original-file URLs, admin/QC/usage reports, MCP or generated answers.
Caller headers and owner identity are not forwarded. Package credentials never leave the host. Upstream
redirects and HTTP proxy environment variables are disabled. Request/body deadlines, 8 connection threads,
2 simultaneous upstream operations and 30 authenticated requests/minute bound resource use.
These are bounded local guards, not a production ingress/security certification or R13 access audit.

## Publish only if you choose to

The UI generates **manual** Cloudflare Quick Tunnel or Tailscale Funnel commands against the RELAY port.
Do not tunnel Studio or the full package server. No tunnel is started or tested by Studio.

Cloudflare Quick Tunnels require the separately installed cloudflared CLI, use temporary changing hostnames,
and have no production SLA. Start one only for a consciously chosen test, copy its real HTTPS address back
into Connect and regenerate. Its interactive email gate is not usable by non-interactive Actions clients.

Tailscale Funnel requires a separately installed/authenticated Tailscale client, MagicDNS, HTTPS and policy
permission. Funnel exposes the selected port to the public internet, unlike private Tailscale Serve.
The generated foreground command omits --bg and --yes; account/policy approval remains with the owner.
Stop the tunnel with Ctrl+C and then the relay; verify the public address is unavailable.
Do not reset unrelated Tailscale Serve/Funnel settings.

## ChatGPT custom GPT

In an account/workspace that allows GPT editing and Actions:
1. Add an Action and paste the generated schema, or import the schema from the relay's public /openapi.json.
2. Set authentication to API Key, Bearer. Supply only the relay token, never the encryption passphrase,
   package server token or upstream model key. Appropriate account/workspace/domain permissions are needed.
3. Instruct the GPT to cite returned source labels, treat source text as evidence rather than instructions,
   and abstain when no supporting evidence is returned.
4. Use the vendor editor's actual test control with a known question, a no-hit question and a wrong-token
   test. Check that only the package's selected audience/sources appear.
5. Public sharing may require a privacy-policy URL; describe the actual deployment/operator's processing.

Official Actions requirements include public TLS 1.2+ HTTPS on port 443, valid certificates, 45-second
round trip and text request/response payloads below 100000 characters. The relay's response budget is
stricter, but a cold embedder/retriever may exceed the time limit. Local transport checks are not vendor import,
model-driven tool-use certification or evidence of a public deployment.

## Gemini: separate capabilities

The downloaded JSON is a **developer function declaration** named retrieveKnowledge. Gemini function calling
asks the caller application to execute that function against the relay and send the response back to Gemini.
This declaration does not contain an authenticated application/SDK runner. No automatic consumer Gems
Actions/OpenAPI import is claimed. Official Gems help documents instructions and knowledge files, not an
equivalent certified import here. Gemini CLI's existing MCP connection is a different local-client option.

## Evidence and remaining work

Local actual exported-package/relay tests cover encrypted/plaintext scope, hidden filenames, separate auth,
unsupported routes, query validation, bounded output, disabled proxy/redirect following and rate limits.
The Studio/package browser workflow must be checked before merge; current results belong in RESULTS/HANDOFF.

Not performed: public exposure, paid/vendor accounts, credential creation, Tailscale policy changes,
public certificate tests, vendor editor imports or live ChatGPT/Gemini model-driven certification.
No OAuth, per-user identity, signing, automatic tunnel management or fleet updates are supplied.
R4 vendor-store export remains unconfirmed.

## Official sources checked 2026-11-20

- [Actions introduction](https://developers.openai.com/api/docs/actions/introduction)
- [Actions setup](https://developers.openai.com/api/docs/actions/getting-started)
- [Actions authentication](https://developers.openai.com/api/docs/actions/authentication)
- [Actions production constraints](https://developers.openai.com/api/docs/actions/production)
- [Gemini developer function calling](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling)
- [Consumer Gems help](https://support.google.com/gemini/answer/15235603)
- [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
- [Tailscale Funnel CLI](https://tailscale.com/docs/reference/tailscale-cli/funnel)
