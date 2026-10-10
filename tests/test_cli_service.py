"""`fts up|down|service`: manual and systemd are separate, explicit options."""
from __future__ import annotations

import asyncio
import sys
import tempfile
import types
from typing import ClassVar

import pytest

from finetune_studio.cli.commands import service as svc
from finetune_studio.supervisor import control
from finetune_studio.supervisor.core import Supervisor
from finetune_studio.supervisor.events import EventLog
from finetune_studio.supervisor.paths import socket_path
from finetune_studio.supervisor.spec import ComponentSpec


@pytest.fixture(autouse=True)
def short_root(monkeypatch):
    import shutil
    d = tempfile.mkdtemp(prefix="fts")
    monkeypatch.setenv("FTS_ROOT", d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def args(**kw):
    base = {"manual": False, "systemd": False, "host": "127.0.0.1", "port": 7999}
    base.update(kw)
    return types.SimpleNamespace(**base)


class Popen:
    calls: ClassVar[list] = []

    def __init__(self, argv, **kw):
        Popen.calls.append((argv, kw))


def fake_ready(monkeypatch):
    monkeypatch.setattr(svc, "_wait_ready", lambda s: {"supervisor": {"pid": 7}, "components": {"web": {"state": "ready"}}})


def test_manual_up_spawns_a_detached_supervisor_with_a_log(monkeypatch, capsys, short_root):
    Popen.calls = []
    monkeypatch.setattr(svc.subprocess, "Popen", Popen)
    monkeypatch.setattr(svc, "_running", lambda: None)
    monkeypatch.setattr(svc, "_unit_active", lambda: False)
    fake_ready(monkeypatch)
    svc.cmd_up(args(manual=True))
    argv, kw = Popen.calls[0]
    assert argv[:3] == [sys.executable, "-m", "finetune_studio.supervisor"] and "--port" in argv and "7999" in argv
    assert kw["start_new_session"] is True and kw["stdin"] is svc.subprocess.DEVNULL
    assert "manual" in capsys.readouterr().out


def test_manual_up_refuses_while_the_systemd_unit_is_active(monkeypatch, capsys):
    monkeypatch.setattr(svc, "_running", lambda: None)
    monkeypatch.setattr(svc, "_unit_active", lambda: True)
    with pytest.raises(SystemExit) as exc:
        svc.cmd_up(args(manual=True))
    assert exc.value.code == 1 and "systemctl" in capsys.readouterr().err


def test_default_up_uses_systemd_when_the_unit_is_installed_else_manual(monkeypatch):
    seen = []
    monkeypatch.setattr(svc, "_running", lambda: None)
    monkeypatch.setattr(svc, "_unit_installed", lambda: True)
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/systemctl")
    monkeypatch.setattr(svc, "_systemctl", lambda *a: seen.append(a) or 0)
    fake_ready(monkeypatch)
    svc.cmd_up(args())
    assert seen == [("start", svc.UNIT)]
    seen.clear()
    Popen.calls = []
    monkeypatch.setattr(svc, "_unit_installed", lambda: False)
    monkeypatch.setattr(svc, "_unit_active", lambda: False)
    monkeypatch.setattr(svc.subprocess, "Popen", Popen)
    svc.cmd_up(args())
    assert seen == [] and len(Popen.calls) == 1


def test_systemd_up_without_a_unit_says_how_to_install(monkeypatch, capsys):
    monkeypatch.setattr(svc, "_running", lambda: None)
    monkeypatch.setattr(svc, "_unit_installed", lambda: False)
    with pytest.raises(SystemExit):
        svc.cmd_up(args(systemd=True))
    assert "fts service install" in capsys.readouterr().err


def test_up_is_a_noop_when_already_running(monkeypatch, capsys):
    monkeypatch.setattr(svc, "_running", lambda: {"supervisor": {"pid": 5, "launcher": "systemd"}})
    svc.cmd_up(args())
    assert "already running (systemd" in capsys.readouterr().out


def test_down_stops_a_manual_supervisor_over_the_socket_and_waits_until_gone():
    async def main():
        spec = ComponentSpec(name="web", argv=lambda: [sys.executable, "-c", "import time; time.sleep(60)"], ready_after=0.2, grace=2.0)
        sup = Supervisor([spec], EventLog(), meta={"launcher": "manual"})
        server = await control.serve(sup, socket_path())
        runner = asyncio.create_task(sup.run())
        await asyncio.sleep(0.5)
        down = asyncio.create_task(asyncio.to_thread(svc.cmd_down, args()))
        await asyncio.wait_for(runner, 10)  # the supervisor loop ends on the shutdown request
        await server.stop()  # what __main__.run does after the loop ends
        socket_path().unlink(missing_ok=True)
        await asyncio.wait_for(down, 20)
        assert any(e["kind"] == "shutdown_requested" for e in sup.events.since(0))

    asyncio.run(main())


def test_down_when_nothing_runs_is_not_an_error(monkeypatch, capsys):
    monkeypatch.setattr(svc, "_running", lambda: None)
    monkeypatch.setattr(svc, "_unit_active", lambda: False)
    svc.cmd_down(args())
    assert "not running" in capsys.readouterr().out


def test_service_command_delegates_to_the_installer_with_the_right_flags(monkeypatch):
    calls = []
    monkeypatch.setattr(svc.subprocess, "run", lambda argv, **kw: calls.append(argv) or types.SimpleNamespace(returncode=0))
    for action, no_start, flag in (("install", False, None), ("install", True, "--no-start"), ("uninstall", False, "--uninstall"),
                                   ("status", False, "--status"), ("restart", False, "--restart")):
        with pytest.raises(SystemExit) as exc:
            svc.cmd_service(types.SimpleNamespace(action=action, no_start=no_start))
        assert exc.value.code == 0
        assert calls[-1][0] == "bash" and calls[-1][1].endswith("install-service.sh")
        assert (flag in calls[-1]) if flag else len(calls[-1]) == 2
