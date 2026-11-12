# Deploy a knowledge package

This guide is for the **standalone package** downloaded from a project's Package page. It does not install or update
Finetune Studio itself; see [INSTALL](INSTALL.md) and [DEPLOYMENT](DEPLOYMENT.md) for the Studio.

Choose the connection before handing the package over:

| Use | Recommended connection | Who can access it |
|---|---|---|
| Your own assistant on this computer | MCP over stdio (`run-mcp.sh`) | The local MCP client and user account |
| A trusted assistant on another computer | MCP over HTTPS (`/mcp`) | Clients that hold the service bearer token |
| A public-facing chatbot | Your own web app/API gateway in front of the package | Public users see only the routes your gateway exposes |

The package's HTTP server is a small trusted-service API, not a multi-user account system. A bearer token grants the
same access to every holder. Do not put it in a public webpage or give it to anonymous visitors.

## 1. Decide what the package contains

Build separate exports for separate audiences. On **Package**, choose only the folders, tags or documents that belong
in that export. Documents left out are not in the package and cannot be searched through it.

- **Public chatbot:** select only publishable documents (for example `public/`), choose the customer audience, leave
  **Include original files** off, and turn on **Hide file names in answers**. Check the selected contents themselves:
  this option does not redact names, personal information or secrets written inside a document.
- **Internal assistant:** include the material its authorized users need, and keep the export and its passphrase in
  private storage. Include original files only when users need to download them; anyone who can use the package server
  can fetch originals that were included.

**Important limit:** hide-file-names applies to generated answers. It does not hide names or passage text from raw
`/search` results or the MCP `rag_search` tool. Anyone with access to those routes can inspect the package's selected
contents. Do not expose raw search, MCP, document-list or original-file routes to public visitors. The audience choice
is about package contents and answer behavior, not per-user authorization.

The export is encrypted at rest by default. Save its passphrase in a password manager; the Studio shows a generated
passphrase only once. Encryption protects the archive and on-disk corpus, not a running server from people who can
query it. The service needs the passphrase while it runs.

## 2. Run it beside a local MCP client

Download and extract the archive. Open a terminal in the extracted folder (the one containing `setup.sh`). For an
encrypted package, create a passphrase file using your password manager or a secure editor; do not put the passphrase
directly in a command or commit it. Restrict it to your account:

```bash
chmod 600 "$HOME/.config/fts/brain-passphrase.txt"
bash setup.sh --keyfile "$HOME/.config/fts/brain-passphrase.txt"
```

`setup.sh` installs the package's private Python environment and writes `MCP-ENTRY.txt` with the absolute package and
keyfile paths. It can also start the HTTP server or install a persistent systemd user service. For a local MCP client,
copy the generated stdio entry into that client's MCP configuration, then restart/reload the client. The command in
the entry runs `run-mcp.sh`; the encrypted package is opened locally and its passphrase is read from the keyfile.

No network listener is needed for stdio. Keep the extracted folder and keyfile available to the OS account running the
MCP client. An unencrypted package does not need `--keyfile`, but its archive and corpus are readable by anyone who can
read those files.

## 3. Serve HTTP to trusted remote clients over HTTPS

Keep the package server bound to loopback when a reverse proxy runs on the same machine. Give the service a long random
bearer token and store it in a private file so you can configure trusted clients later:

```bash
mkdir -p "$HOME/.config/fts"
chmod 700 "$HOME/.config/fts"
openssl rand -hex 32 > "$HOME/.config/fts/brain-http-token"
chmod 600 "$HOME/.config/fts/brain-http-token"
```

For an encrypted package, use the same private passphrase file from the previous step. Start a persistent service with
the token in its environment; `setup.sh` writes it into a user-only systemd unit:

```bash
RAG_AUTH_TOKEN="$(cat "$HOME/.config/fts/brain-http-token")" \
  bash setup.sh --yes --service --host 127.0.0.1 --port 8899 \
  --keyfile "$HOME/.config/fts/brain-passphrase.txt"
```

For an unencrypted package, omit `--keyfile`. On a host without systemd, start `bash run-http.sh --host 127.0.0.1
--port 8899` under a supervisor that supplies `RAG_AUTH_TOKEN` and restarts the process. Do not expose the port
directly to the internet. If the proxy is on another machine, use a private network and bind/firewall the service only
to that network; non-loopback binds require `RAG_AUTH_TOKEN`.

Example Caddy site, with DNS for `brain.example.com` pointing to this host and ports 80/443 reachable by Caddy:

```caddyfile
brain.example.com {
    reverse_proxy 127.0.0.1:8899
}
```

Caddy obtains and renews HTTPS certificates. The proxy forwards the client's `Authorization` header by default. Keep
the server token private; distribute it only to trusted MCP clients over a secure channel. Configure an HTTP MCP client
to use `https://brain.example.com/mcp` with `Authorization: Bearer <token>`; for example, Claude Code supports:

```bash
claude mcp add --transport http team-brain https://brain.example.com/mcp \
  --header "Authorization: Bearer <token>"
```

Replace the placeholder locally; never save a real token in this guide, a public repository or a shared command log.
HTTP clients need the token on every request. The package's built-in chat page can also use a token, but anyone you give
the token to gets the same access as the other trusted clients. A model-provider API key is separate from this service
token; keys typed into the chat page are sent with the request and are not saved by the package server.

## 4. Put a public chatbot in front of it

Do **not** give a browser visitor the package's bearer token. Instead, keep the token on a server-side gateway and build
a customer-facing frontend that calls that gateway. The gateway should:

1. expose only the answer operation needed by the bot (normally `POST /ask`), not `/search`, `/mcp`, `/profile`,
   `/files/*` or document-list operations;
2. attach the package token server-side, reject requests outside its contract, and avoid returning source filenames;
3. rate-limit requests, apply your normal abuse controls, and avoid logging bearer tokens or sensitive questions;
4. use HTTPS from the browser to your gateway and from the gateway to the package server.

Use the public-scoped package—not the internal package—as the gateway's data source. A customer-audience export with
file names hidden makes generated answers avoid naming their source files, but the package still contains the selected
passages and may contain identifying text. Review the exported documents and test the public bot as an untrusted user
before launch. If you do not have a gateway that can restrict routes and keep credentials server-side, keep the package
private and use a trusted MCP client instead.

## 5. Publish an update

When the source documents change, update the Smart Brain, run the quiz again, then use **Package → New version of this**
to export a newer version of the same package. Keep its package identity and passphrase so the installed keyfile still
opens it. Keep the downloaded archive as a rollback copy until the update is verified.

On the package host, from its installed directory:

```bash
bash update.sh /path/to/new-version.tar.gz \
  --keyfile "$HOME/.config/fts/brain-passphrase.txt"
```

The updater checks that this is the same package and a newer version, opens the new encrypted corpus before replacing
anything, and keeps the runtime configuration and keyfile. A systemd user service is restarted automatically; otherwise
restart the server or MCP client. The simplest update keeps the same passphrase. If you change it, update the contents
of the configured keyfile before restarting the service; keep a secure backup of the old keyfile until rollback is no
longer needed. If you change the keyfile's path, update the service/MCP command too. For `setup.sh` installations,
`bash setup.sh --update /path/to/new-version.tar.gz --keyfile ...` is equivalent.

After an HTTP update, check `https://brain.example.com/health` with the bearer token and confirm the package version
and document counts; then ask a question whose answer changed. For stdio, restart the MCP client and check
`rag_info`/`rag_search`. Keep the previous archive and passphrase available until these checks pass.

## Troubleshooting

| Symptom | Check |
|---|---|
| `401 unauthorized` | Send `Authorization: Bearer <token>`; check that the client and service use the same token. |
| Service refuses a public bind | Set `RAG_AUTH_TOKEN`; prefer loopback plus a same-host HTTPS proxy. |
| Encrypted package will not start | Confirm the keyfile contains the exact export passphrase and is readable by the service account. |
| HTTPS works but MCP does not | Use the `/mcp` path, preserve the `Authorization` header, and verify the client supports Streamable HTTP. |
| New package is refused | It must be a newer version of the same package; re-export with **New version of this**. |
| Search works but answers do not | Answer generation is optional; configure the package's answer connection or let the MCP client write answers from `rag_search` passages. |

For the generated package's complete configuration and API reference, read its `README.md`; it describes the exact
search layers, included models, tools and settings shipped by that export.
