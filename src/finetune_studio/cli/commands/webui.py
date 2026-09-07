"""`fts webui` — launch the WebUI server (uvicorn)."""
from __future__ import annotations


def cmd_webui(args) -> None:
    import uvicorn
    uvicorn.run(
        "finetune_studio.webui.app:app",
        host=args.host, port=args.port, reload=args.reload,
    )
