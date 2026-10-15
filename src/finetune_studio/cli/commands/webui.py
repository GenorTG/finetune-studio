"""`fts webui` — launch the WebUI server (uvicorn)."""
from __future__ import annotations


def cmd_webui(args) -> None:
    import os

    import uvicorn

    from finetune_studio.access import server
    addr = server.resolve(args.host, args.port)
    # The app reads these to know who can reach it (the "no password on the network" warning, host checks).
    os.environ["FTS_LISTEN_HOST"], os.environ["FTS_LISTEN_PORT"] = addr.host, str(addr.port)
    if addr.extra_hosts():
        print(f"note: fts webui listens on {addr.host} only; the Tailscale address {addr.tailscale_ip} is served by "
              "the service (fts up / systemd)")
    uvicorn.run("finetune_studio.webui.app:app", host=addr.host, port=addr.port, reload=args.reload)
