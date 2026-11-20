---
id: package-bots
title: Slack, Teams and Discord bot bridges
page: connect
keywords: slack teams discord bots webhook signature ephemeral audience privacy replay
---
## Purpose
Connect → optional bot section generates config and registration templates. New Linux exports ship
package_bots.py, a loopback-only bridge to the package's R2 knowledge + operator LLM API.
Generation creates no account, key, registration, ingress or message. Local HTTP testing is not vendor certification.

## Workflow
1. Export one scoped package for the intended audience; configure its operator LLM and bearer token.
2. Select Slack /brain, Teams outgoing webhook, or Discord /brain. Enter the package's direct literal-loopback
   HTTP origin and a distinct bridge port. Generate, copy or download templates. Setup values are sent only
   to this server for validation, never probed or stored. Editing invalidates stale downloads.
3. Save the download's config object as bot-config.json (chmod 600). Replace team_id (guild for Discord),
   channel_id, user_ids (explicit allowlist), application_id (Slack app ID, Discord app ID, Teams tenant ID).
   No wildcard or REPLACE values can start a listener. No roles or package scopes come from command arguments.
4. Set RAG_AUTH_TOKEN privately in the package-host environment. Also set SLACK_SIGNING_SECRET,
   TEAMS_SIGNING_SECRET (base64 HMAC key), or DISCORD_PUBLIC_KEY (hex Ed25519 verification public key).
   Never put secrets in JSON, URLs, templates, browser inputs or source control.
5. Run the generated command inside the installed package directory. Only 127.0.0.1 is bound. Keep replay_file
   in a private persistent directory (0600 file); it stores only expiring hashed IDs, not questions or tokens.
   Preserve it across restarts. Use a separate file/config/bridge/scoped package for a different audience.
6. Separately review public HTTPS ingress targeting ONLY the bridge and register the vendor interface.
   Slack: app manifest slash-command URL /slack, commands permission, workspace installation.
   Discord: register the downloaded guild command, install with application commands permission, set the
   interaction endpoint /discord; Discord validates a signed PING before accepting the URL.
   Teams: team settings → Apps → create outgoing webhook; callback /teams; copy its HMAC key privately.
   Teams policy/tenant permissions may disallow this. This is not an Azure Bot Framework/OAuth application.
7. Verify actual vendor known/no-hit questions, wrong audience, latency, private/public reply behavior and shutdown.
   Stop ingress, then bridge (Ctrl+C) and package. Generation performs none of those operations.

## Key controls
- `#bots-builder` — optional local setup section.
- `#bots-vendor` — Slack slash command, Teams outgoing webhook, Discord HTTP interaction.
- `#bots-package-url` — package literal-loopback origin, never Studio or an arbitrary proxy URL.
- `#bots-port` — distinct loopback bridge port, 1024–65535.
- `#bots-generate` — pure template validation/generation; does not test a connection.
- `#bots-copy` and `#bots-download` — credential-free setup, not an installed/registered bot.

## What to check
Raw-body Slack HMAC-SHA256, Teams base64-key HMAC-SHA256, Discord Ed25519 verification fail closed.
Slack/Discord signed timestamps allow at most five minutes; Teams authenticated activity timestamp
must also be recent. Stable event IDs are persisted as SHA256 hashes for 24 hours, maximum 10000 entries.
Duplicate events return409 without rerunning retrieval/LLM, including after restart; full/unavailable storage
fails closed. Replay protection is not a signed identity/access audit (R13).

Only the selected vendor path accepts POST; no MCP, original files, QC or usage admin. Fixed loopback package
target, no caller headers, redirects or environment proxies. Body ≤24000 bytes, question ≤4000 characters,
response ≤200000 bytes, output about2000 characters, 8 request threads, 2 answer slots, 30 authenticated
commands/minute. Original files, image blobs and citation preview text are not forwarded. Names hidden by the
package stay hidden. Plain-text output neutralizes mentions/URLs and appends bounded package citations.
The operator LLM sees the scoped evidence; the vendor sees the question and reply/citations. R8 package query
privacy and retention settings apply normally. No extra plaintext query store is created by this bridge.

Slack acknowledges immediately, then posts an ephemeral reply via the signed payload's hooks.slack.com
commands response_url. Discord returns deferred ephemeral type5/flag64, then PATCHes the original interaction
using a fixed discord.com webhook URL constructed from authenticated app/token fields; allowed_mentions empty.
Neither forwards the package bearer to callback endpoints. No proactive messages or broad chat-history reading.
Async callbacks have no automatic retry. Failed delivery emits a sanitized generic local log. Pending replies are
in memory and can be lost on restart; duplicates remain blocked. Submit a new question rather than replaying.

Teams outgoing webhooks work only through @mention in public channels. The reply is visible to the ENTIRE
configured channel, not just allowed invoking users. Export for that channel audience. Package call timeout3.5s
leaves time within Teams' five-second deadline, but cold retrieval/slow LLM can fail. Errors produce an honest
unavailable message, not a fake answer or an unimplemented deferred Teams reply. A full Azure Bot with durable
conversation delivery is not shipped. Provider/config errors are sanitized; no-hit replies come from R2 without
claiming generation. Counts/identity audit, durable queues, service management and vendor certification are separate.

## Common mistakes
Do not confuse a Teams outgoing webhook with a full Azure Bot, ephemeral invocation permission with a channel
document ACL, or local synthetic HTTP tests with actual vendor delivery. Do not discard replay state at restart
or expect an in-memory pending reply to survive a crash. Re-submit a new question after a delivery failure.

Official sources checked 2026-11-20:
- https://docs.slack.dev/authentication/verifying-requests-from-slack/
- https://docs.slack.dev/interactivity/implementing-slash-commands/
- https://docs.discord.com/developers/interactions/receiving-and-responding
- https://docs.discord.com/developers/interactions/application-commands
- https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-outgoing-webhook
