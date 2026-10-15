"""``python -m finetune_studio.supervisor``: the process systemd runs."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import signal
import sys
from pathlib import Path

from finetune_studio.supervisor import control
from finetune_studio.supervisor.core import Supervisor
from finetune_studio.supervisor.events import EventLog
from finetune_studio.supervisor.inference_component import inference_spec
from finetune_studio.supervisor.launcher import detect as detect_launcher
from finetune_studio.supervisor.paths import (
    events_path,
    inference_socket_path,
    lock_path,
    socket_path,
)
from finetune_studio.supervisor.web import web_spec


def take_lock(path: Path):
    """One supervisor per FTS_ROOT: returns the held lock file, or None if another one runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def listen_plan(host: str | None, port: int | None):
    """Re-read before every web start: a changed Settings -> Access & network applies with ``fts restart web``."""
    from finetune_studio.access import server

    last: dict[str, tuple[tuple[str, ...], int]] = {}

    def plan() -> tuple[tuple[str, ...], int]:
        try:
            addr = server.refresh_tailscale(server.resolve(host, port))
            last["plan"] = (addr.binds(), addr.port)
        except ValueError as e:
            if "plan" not in last:
                raise
            print(f"[supervisor] {e}; keeping the previous listen addresses", file=sys.stderr, flush=True)
        return last["plan"]

    return plan


async def run(host: str | None, port: int | None) -> None:
    from finetune_studio.supervisor.jobs import JobTable
    events = EventLog(events_path())
    plan = listen_plan(host, port)
    binds, bound_port = plan()
    listen = ", ".join(f"{h}:{bound_port}" for h in binds)
    sup = Supervisor([web_spec(plan=plan, cwd=os.getcwd()),
                      inference_spec(inference_socket_path(), cwd=os.getcwd())], events,
                     meta={"launcher": detect_launcher(), "listen": listen},
                     jobs=JobTable(events, cwd=os.getcwd(), inference_socket=inference_socket_path()))
    server = await control.serve(sup, socket_path())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(sup.shutdown_gracefully()))
    print(f"[supervisor] pid {os.getpid()} serving {listen}, control socket {socket_path()}", flush=True)
    try:
        await sup.run()
    finally:
        await server.stop()
        socket_path().unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m finetune_studio.supervisor")
    parser.add_argument("--host", default=None, help="default: $FTS_ROOT/server.json (127.0.0.1 = this computer only)")
    parser.add_argument("--port", type=int, default=None, help="default: $FTS_ROOT/server.json, else 7860")
    args = parser.parse_args(argv)
    from finetune_studio import logsetup
    logsetup.setup("supervisor")
    from finetune_studio.access import server
    try:
        server.resolve(args.host, args.port)              # fail fast on a broken server.json
    except ValueError as e:
        print(f"[supervisor] {e}", file=sys.stderr)
        return 2
    lock = take_lock(lock_path())
    if lock is None:
        print(f"[supervisor] another supervisor already runs for this FTS_ROOT ({lock_path()})", file=sys.stderr)
        return 1
    asyncio.run(run(args.host, args.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
