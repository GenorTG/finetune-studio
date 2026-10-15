"""`fts webui` — launch the WebUI server (uvicorn)."""
from __future__ import annotations


def cmd_webui(args) -> None:
    import os

    import uvicorn

    from finetune_studio.access import server
    addr = server.resolve(args.host, args.port)
    binds = addr.binds()
    # One uvicorn socket here: 0.0.0.0 when the switches need more than one address; the middleware then refuses
    # whatever is switched off (the service binds each address on its own instead).
    host = binds[0] if len(binds) == 1 else "0.0.0.0"
    os.environ["FTS_LISTEN_ADDRS"] = os.environ["FTS_LISTEN_HOST"] = host
    os.environ["FTS_LISTEN_PORT"] = str(addr.port)
    if len(binds) > 1:
        print(f"note: fts webui listens on 0.0.0.0:{addr.port} and refuses what is switched off "
              f"({server.describe(addr)} is on); the service binds {', '.join(binds)} only")
    uvicorn.run("finetune_studio.webui.app:app", host=host, port=addr.port, reload=args.reload)
