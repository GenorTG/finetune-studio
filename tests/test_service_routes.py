"""``/api/service/*``: honest statuses, plain-language problems, restart through the supervisor."""
from __future__ import annotations

import asyncio
import shutil
import socket
import sys
import tempfile
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from finetune_studio.supervisor import control
from finetune_studio.supervisor.core import Supervisor
from finetune_studio.supervisor.events import EventLog
from finetune_studio.supervisor.paths import socket_path
from finetune_studio.supervisor.spec import ComponentSpec
from finetune_studio.webui.restart_cause import describe
from finetune_studio.webui.routes import service
from finetune_studio.webui.routes.system import router as system_router


@pytest.fixture
def client(monkeypatch):
    d = tempfile.mkdtemp(prefix="fts")
    monkeypatch.setenv("FTS_ROOT", d)
    app = FastAPI()
    app.include_router(service.router)
    app.include_router(system_router)
    yield TestClient(app)
    shutil.rmtree(d, ignore_errors=True)


def comp(state="ready", **kw):
    base = {"desired": "running", "state": state, "pid": 1, "uptime_s": 5.0, "spawns": 1, "restarts": 0,
            "consecutive_crashes": 0, "last_exit": None, "health": {}, "retry_in_s": None}
    base.update(kw)
    return base


def test_health_is_a_plain_200(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_status_without_a_supervisor_is_200_unmanaged_with_a_hint(client):
    body = client.get("/api/service/status").json()
    assert body["managed"] is False and body["problems"] == [] and "fts up" in body["hint"]


def test_actions_without_a_supervisor_are_409_not_a_silent_ok(client):
    r = client.post("/api/service/components/web/restart")
    assert r.status_code == 409 and "supervisor" in r.json()["error"]
    assert client.get("/api/service/events").status_code == 409
    assert client.get("/api/service/logs/web").status_code == 409


def test_dead_socket_file_is_503_with_an_error_problem(client):
    path = socket_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX) as s:
        s.bind(str(path))  # a socket file nobody listens on: a supervisor that died uncleanly
    r = client.get("/api/service/status")
    assert r.status_code == 503
    body = r.json()
    assert body["managed"] is True and body["reachable"] is False
    assert body["problems"][0]["severity"] == "error"


def test_problems_are_plain_language_and_worst_first():
    now = time.time()
    crashed = comp("ready", spawns=2, restarts=1, last_exit={"signal": "SIGSEGV", "code": None, "at": now - 30, "reason": "exited"})
    probs = service.find_problems({"web": crashed}, now)
    assert probs[0]["severity"] == "notice" and "restarted unexpectedly" in probs[0]["title"]
    assert "SIGSEGV" in probs[0]["detail"] and "marked failed" in probs[0]["detail"]
    failed = comp("failed", consecutive_crashes=8, last_exit={"signal": "SIGABRT", "code": None, "at": now, "reason": "exited"})
    backoff = comp("backoff", retry_in_s=4.0, last_exit={"signal": None, "code": 3, "at": now, "reason": "exited"})
    probs = service.find_problems({"a": backoff, "b": failed, "c": crashed}, now)
    assert [p["severity"] for p in probs] == ["error", "warning", "notice"]
    assert "8 early exits" in probs[0]["detail"] and "SIGABRT" in probs[0]["detail"]
    assert "code 3" in probs[1]["detail"] and "4.0" in probs[1]["detail"]


def test_requested_restart_and_old_exits_are_not_problems():
    now = time.time()
    requested = comp("ready", last_exit={"signal": "SIGTERM", "code": None, "at": now - 5, "reason": "requested"})
    old = comp("ready", last_exit={"signal": "SIGSEGV", "code": None, "at": now - 3600, "reason": "exited"})
    assert service.find_problems({"web": requested, "x": old}, now) == []


def test_restart_cause_text():
    now = time.time()
    assert describe(None) == ""
    assert "SIGSEGV" in describe({"signal": "SIGSEGV", "at": now - 5, "reason": "exited"})
    assert "on request" in describe({"signal": "SIGTERM", "at": now - 5, "reason": "requested"})
    assert "health checks" in describe({"signal": "SIGTERM", "at": now - 5, "reason": "unhealthy"})
    assert "code 3" in describe({"code": 3, "signal": None, "at": now - 5, "reason": "exited"})
    assert describe({"signal": "SIGSEGV", "at": now - 99999, "reason": "exited"}) == ""


def test_status_events_logs_and_restart_through_a_real_supervisor(client):
    async def main():
        spec = ComponentSpec(name="web", argv=lambda: [sys.executable, "-c", "print('hello from web', flush=True); import time; time.sleep(30)"],
                             ready_after=0.2, grace=2.0, backoff_base=0.2)
        sup = Supervisor([spec], EventLog(), meta={"launcher": "manual", "listen": "127.0.0.1:1"})
        server = await control.serve(sup, socket_path())
        runner = asyncio.create_task(sup.run())

        def get(url):
            return client.get(url)

        try:
            for _ in range(100):
                body = (await asyncio.to_thread(get, "/api/service/status")).json()
                if body.get("components", {}).get("web", {}).get("state") == "ready":
                    break
                await asyncio.sleep(0.1)
            assert body["managed"] is True and body["reachable"] is True
            assert body["supervisor"]["launcher"] == "manual" and body["problems"] == []
            logs = (await asyncio.to_thread(get, "/api/service/logs/web")).json()
            assert "hello from web" in logs["lines"]
            r = await asyncio.to_thread(client.post, "/api/service/components/web/restart")
            assert r.status_code == 202
            assert (await asyncio.to_thread(client.post, "/api/service/components/ghost/restart")).status_code == 404
            assert (await asyncio.to_thread(client.post, "/api/service/components/web/explode")).status_code == 404
            assert (await asyncio.to_thread(get, "/api/service/logs/ghost")).status_code == 404
            for _ in range(100):
                ev = (await asyncio.to_thread(get, "/api/service/events")).json()["events"]
                if [e["kind"] for e in ev].count("spawned") == 2:
                    break
                await asyncio.sleep(0.1)
            assert "restart_requested" in [e["kind"] for e in ev]
        finally:
            sup.shutdown()
            await runner
            await server.stop()

    asyncio.run(main())
