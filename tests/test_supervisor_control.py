"""Control API over the unix socket, the fd hand-off of the listening socket, and the CLI table."""
from __future__ import annotations

import asyncio
import socket
import sys

import pytest

from finetune_studio.cli.commands import supervisor as cli_sup
from finetune_studio.supervisor import control
from finetune_studio.supervisor.client import (
    SupervisorClient,
    SupervisorError,
    SupervisorUnavailable,
)
from finetune_studio.supervisor.core import Supervisor
from finetune_studio.supervisor.events import EventLog
from finetune_studio.supervisor.paths import socket_path
from finetune_studio.supervisor.spec import ComponentSpec
from finetune_studio.supervisor.web import ListenSocket


@pytest.fixture(autouse=True)
def short_root(monkeypatch):
    import tempfile
    d = tempfile.mkdtemp(prefix="fts")  # AF_UNIX paths are limited to ~100 chars
    monkeypatch.setenv("FTS_ROOT", d)
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def sleeper(name: str = "web", **kw) -> ComponentSpec:
    return ComponentSpec(name=name, argv=lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
                         ready_after=0.2, grace=2.0, backoff_base=0.2, **kw)


def serve_and(scenario, specs):
    async def main():
        sup = Supervisor(specs, EventLog())
        server = await control.serve(sup, socket_path())
        runner = asyncio.create_task(sup.run())
        try:
            await scenario(sup)
        finally:
            sup.shutdown()
            await runner
            await server.stop()
    asyncio.run(main())


def test_status_actions_events_logs_and_errors_over_the_socket():
    async def scenario(sup):
        client = SupervisorClient()
        for _ in range(100):
            if (await asyncio.to_thread(client.status))["components"]["web"]["state"] == "ready":
                break
            await asyncio.sleep(0.1)
        snap = await asyncio.to_thread(client.status)
        assert snap["components"]["web"]["state"] == "ready" and snap["supervisor"]["pid"]
        assert (await asyncio.to_thread(client.act, "web", "restart"))["accepted"] is True
        for _ in range(100):
            if (await asyncio.to_thread(client.status))["components"]["web"]["spawns"] == 2:
                break
            await asyncio.sleep(0.1)
        kinds = [e["kind"] for e in await asyncio.to_thread(client.events)]
        assert "restart_requested" in kinds and kinds.count("spawned") == 2
        assert await asyncio.to_thread(client.logs, "web") == []
        with pytest.raises(SupervisorError) as nope:
            await asyncio.to_thread(client.act, "ghost", "restart")
        assert nope.value.status == 404
        with pytest.raises(SupervisorError) as bad:
            await asyncio.to_thread(client.act, "web", "explode")
        assert bad.value.status == 404
        assert socket_path().stat().st_mode & 0o077 == 0  # owner only

    serve_and(scenario, [sleeper()])


def test_client_without_a_supervisor_raises_unavailable():
    with pytest.raises(SupervisorUnavailable):
        SupervisorClient().status()


def test_listen_socket_survives_child_death_then_closes_on_release():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    listen = ListenSocket("127.0.0.1", port)
    code = (
        "import socket, sys\n"
        "fd = int(sys.argv[1])\n"
        "s = socket.socket(fileno=fd)\n"
        "while True:\n"
        "    c, _ = s.accept(); c.sendall(b'pong'); c.close()\n"
    )
    spec = ComponentSpec(name="web", argv=lambda: [sys.executable, "-c", code, str(listen.fileno())],
                         pass_fds=lambda: (listen.fileno(),), prepare=listen.open, release=listen.close,
                         ready_after=0.3, grace=1.0, backoff_base=0.3, backoff_max=0.3)

    def ping() -> bytes:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
            return c.recv(4)

    async def scenario(sup):
        for _ in range(100):
            if sup.snapshot()["components"]["web"]["state"] == "ready":
                break
            await asyncio.sleep(0.05)
        assert await asyncio.to_thread(ping) == b"pong"
        pid = sup.snapshot()["components"]["web"]["pid"]
        import os
        import signal
        os.kill(pid, signal.SIGKILL)
        # While the child is down the port still accepts (kernel backlog) and is served after the respawn.
        assert await asyncio.to_thread(ping) == b"pong"
        assert sup.snapshot()["components"]["web"]["spawns"] >= 2
        sup.stop("web")
        for _ in range(100):
            if sup.snapshot()["components"]["web"]["state"] == "stopped":
                break
            await asyncio.sleep(0.05)
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(("127.0.0.1", port), timeout=2)

    async def main():
        sup = Supervisor([spec], EventLog())
        runner = asyncio.create_task(sup.run())
        try:
            await scenario(sup)
        finally:
            sup.shutdown()
            await runner

    asyncio.run(main())


def test_status_table_shows_state_exit_signal_and_backoff():
    snap = {"supervisor": {"pid": 1, "uptime_s": 125}, "components": {
        "web": {"state": "backoff", "pid": None, "uptime_s": None, "restarts": 2, "retry_in_s": 4.0,
                "last_exit": {"signal": "SIGABRT", "code": None}, "health": {}},
        "inference": {"state": "ready", "pid": 99, "uptime_s": 61, "restarts": 0, "last_exit": None,
                      "health": {"detail": "HTTP 200"}}}}
    text = "\n".join(cli_sup.format_status(snap))
    assert "retry in 4.0s (signal SIGABRT)" in text and "HTTP 200" in text and "2m05s" in text


def test_cli_status_exits_3_when_supervisor_is_down(monkeypatch, capsys):
    import types
    with pytest.raises(SystemExit) as exc:
        cli_sup.cmd_status(types.SimpleNamespace(json=False))
    assert exc.value.code == cli_sup.EXIT_DOWN
    assert "fts up" in capsys.readouterr().err


def test_doctor_runs_with_everything_down(monkeypatch):
    monkeypatch.setattr(cli_sup, "_unit_installed", lambda: False)
    rows = cli_sup.doctor_checks(port=1)
    by = {check: status for status, check, _ in rows}
    assert by["supervisor"] == "fail" and by["port 1"] == "fail"
