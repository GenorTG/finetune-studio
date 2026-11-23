# Local bot bridges (Linux)

New Smart Brain server exports include package_bots.py. Studio and installed-package Connect pages
generate Slack slash-command, Teams outgoing-webhook and Discord HTTP-interaction templates.

The bridges call only their fixed package's /v1/chat/completions endpoint using its existing private bearer.
The package retrieves only its exported audience and answers using its operator-configured LLM.
No model or vendor credentials are bundled. This is local-tested software, not vendor certification.

## Install and configure
1. Export a package for one audience, install it with bash install.sh, configure its LLM and start run-http.sh
   with bearer authentication. Do not publish this full server.
2. On Connect select the bot interface, literal-loopback package address and separate bridge port.
   Generate/download the template JSON. Save its config object in bot-config.json (chmod 600).
3. Replace every REPLACE audience ID: team_id (guild for Discord), channel_id, user_ids explicit allowlist,
   application_id (Slack app / Discord app / Teams tenant). Set replay_file in a private persistent directory.
4. Privately set RAG_AUTH_TOKEN and the vendor verifier named in secret_environment.
   Slack uses SLACK_SIGNING_SECRET; Teams TEAMS_SIGNING_SECRET is its base64 HMAC key;
   Discord DISCORD_PUBLIC_KEY is the app's hex Ed25519 PUBLIC key, not its bot token.
5. In the installed package folder run:
   .venv/bin/python package_bots.py --config bot-config.json

Nothing creates an account, registers a bot, starts ingress or sends a message until a signed command is
received by a running bridge. The listener is loopback only. Configure public HTTPS ingress separately,
targeting ONLY the bridge. Do not expose Studio/full package or put credentials in a public URL/repository.

## Vendor registration gates
- Slack: create/install a workspace app, use the downloaded registration manifest with a real HTTPS /slack
  command URL, grant commands permission. /brain replies are ephemeral. Workspace policy may require approval.
- Discord: create an application, register the downloaded guild-only /brain command via the application-commands
  API (POST /api/v10/applications/APP_ID/guilds/GUILD_ID/commands), install it with applications.commands permission,
  set a real HTTPS /discord interaction endpoint. The template omits global-command-only context fields.
  A correctly signed PING is required. Replies are deferred ephemeral; no gateway bot or message-content scope.
- Teams: a permitted team owner creates an outgoing webhook in team Apps, callback HTTPS /teams.
  The generated object is a setup worksheet, NOT an importable Azure app manifest. The webhook behaves as
  an @mention bot in PUBLIC channels only; replies appear in that channel and are visible to all its members.
  application_id binds the tenant ID; team/channel/user IDs come from the authenticated activity.
  The five-second response deadline requires a warm fast package/LLM; slow calls return unavailable.
  UTC Z timestamps are normalized for the exported runtime's Python 3.10 parser compatibility.
  This does not implement a full Azure Bot with deferred conversation replies.

These account actions and public deployment were NOT performed during implementation.

## Security, privacy and operations
One config/bridge/scoped package per audience; explicit users are invocation permission, not document ACL
translation. Source scope remains fixed at export. Hidden source names remain hidden. Answers append
package-projected source names, never original downloads/images. Package/operator LLM sees queries/evidence;
chat vendors see queries and answers. Configure R8 package retention/counts-only privacy accordingly.
No OAuth, cross-tenant auto-discovery, channel-history access or signed R13 audit is claimed.

Vendor signatures are checked over raw bytes. Slack/Discord timestamp and Teams activity freshness is ±5min.
Replay IDs are hashed in a private0600 SQLite file, retained24h, capped10000; duplicates fail409 and never call
the model twice. Keep the file across restarts. Replay storage errors fail closed. Requests≤24kB,
queries≤4000chars,2 answer slots,8 HTTP threads,30 authenticated commands/minute.
No redirects, proxy environment, arbitrary upstream URL or forwarded vendor headers.
Callbacks go only to Slack commands hooks or fixed Discord interaction webhooks; package bearer never follows.
Slack/Discord acknowledge promptly and deliver async. Pending work is in memory, not durable; callback failures
are logged generically without tokens/questions and are not retried. A new command is required after failure.
Stop ingress first, then bridge with Ctrl+C; verify it is unavailable. A service manager is not installed.

## Verification status
Automated local tests use real HTTP with synthetic HMAC/Ed25519 signatures, exported scoped/encrypted packages
and a synthetic operator LLM. Vendor callback transport is redirected by tests to a loopback collector only.
Tests cover wrong signatures/stale requests/audience, ephemeral/deferred shape, restart replay, privacy, source
projection, no provider, rate/path/callback restrictions and setup controls. Browser checks cover Studio and
package Connect at desktop/mobile sizes. This does not prove vendor editor import, installation, production
TLS/ingress, cold-start SLA, actual vendor delivery or independent answer quality.

Official docs:
- [Slack signing](https://docs.slack.dev/authentication/verifying-requests-from-slack/)
- [Slack commands](https://docs.slack.dev/interactivity/implementing-slash-commands/)
- [Discord interactions](https://docs.discord.com/developers/interactions/receiving-and-responding)
- [Discord commands](https://docs.discord.com/developers/interactions/application-commands)
- [Teams outgoing webhooks](https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-outgoing-webhook)
