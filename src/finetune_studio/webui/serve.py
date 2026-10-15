"""Run the web app on several listening sockets the supervisor already bound (``--fd`` repeated).

uvicorn's own CLI takes a single ``--fd``; this is only used when the web UI also listens on a second address
(the Tailscale IP, ``access.server.ServerAddress.extra_hosts``).
"""
from __future__ import annotations

import argparse
import socket
import sys

import uvicorn

WEB_APP = "finetune_studio.webui.app:app"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m finetune_studio.webui.serve")
    p.add_argument("--fd", type=int, action="append", required=True, help="an inherited listening socket")
    p.add_argument("--timeout-graceful-shutdown", type=int, default=3)
    p.add_argument("--app", default=WEB_APP, help=argparse.SUPPRESS)        # tests serve a tiny app
    args = p.parse_args(argv)
    socks = [socket.socket(fileno=fd) for fd in args.fd]
    config = uvicorn.Config(args.app, access_log=False, timeout_graceful_shutdown=args.timeout_graceful_shutdown)
    uvicorn.Server(config).run(sockets=socks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
