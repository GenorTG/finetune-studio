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
from finetune_studio.supervisor.paths import events_path, lock_path, socket_path
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


async def run(host: str, port: int) -> None:
    sup = Supervisor([web_spec(host, port, cwd=os.getcwd())], EventLog(events_path()))
    server = await control.serve(sup, socket_path())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, sup.shutdown)
    print(f"[supervisor] pid {os.getpid()} serving {host}:{port}, control socket {socket_path()}", flush=True)
    try:
        await sup.run()
    finally:
        await server.stop()
        socket_path().unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m finetune_studio.supervisor")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    lock = take_lock(lock_path())
    if lock is None:
        print(f"[supervisor] another supervisor already runs for this FTS_ROOT ({lock_path()})", file=sys.stderr)
        return 1
    asyncio.run(run(args.host, args.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
