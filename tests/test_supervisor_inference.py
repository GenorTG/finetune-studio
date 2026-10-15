"""The supervisor runs the inference worker as a component: restarted EMPTY after a crash, visible in status."""
from __future__ import annotations

import asyncio
import os
import shutil
import signal
import tempfile
from pathlib import Path

import pytest

from finetune_studio.inference.remote import EngineCrashed, RemoteEngine
from finetune_studio.supervisor import control
from finetune_studio.supervisor.core import Supervisor
from finetune_studio.supervisor.events import EventLog
from finetune_studio.supervisor.inference_component import inference_spec
from finetune_studio.supervisor.paths import inference_socket_path, socket_path


@pytest.fixture
def root(monkeypatch):
    d = tempfile.mkdtemp(prefix="fts")
    monkeypatch.setenv("FTS_ROOT", d)
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_spec_runs_the_worker_module_with_the_socket():
    spec = inference_spec(Path("/tmp/x.sock"), engine="fake")
    argv = spec.argv()
    assert "finetune_studio.inference.worker" in argv and "/tmp/x.sock" in argv and spec.name == "inference"


def test_supervised_worker_crash_is_restarted_empty_and_reported(root):
    sock = inference_socket_path()
    spec = inference_spec(sock, engine="fake")
    spec.backoff_base = 0.2

    async def main():
        sup = Supervisor([spec], EventLog(), meta={"launcher": "manual"})
        server = await control.serve(sup, socket_path())
        runner = asyncio.create_task(sup.run())
        try:
            for _ in range(600):
                if sup.snapshot()["components"]["inference"]["state"] == "ready":
                    break
                await asyncio.sleep(0.1)
            eng = RemoteEngine(sock)
            await asyncio.to_thread(eng.load, "/models/demo.gguf")
            assert (await asyncio.to_thread(eng.worker_info))["transport"] == "supervised"
            with pytest.raises(EngineCrashed) as exc:
                await asyncio.to_thread(eng.generate, [{"role": "user", "content": "__crash__"}])
            for _ in range(600):
                c = sup.snapshot()["components"]["inference"]
                if c["spawns"] == 2 and c["state"] == "ready":
                    break
                await asyncio.sleep(0.1)
            c = sup.snapshot()["components"]["inference"]
            assert c["last_exit"]["signal"] == "SIGABRT" and c["restarts"] == 1
            assert "SIGABRT" in str(exc.value) or "died" in str(exc.value)
            assert (await asyncio.to_thread(lambda: eng.model)) is None  # new worker, nothing loaded: no automatic reload
            info = await asyncio.to_thread(eng.worker_info)
            assert info["pid"] == c["pid"] and info["last_exit"]["signal"] == "SIGABRT"
            # It is usable again once someone loads a model.
            await asyncio.to_thread(eng.load, "/models/demo.gguf")
            assert eng.model is not None
        finally:
            sup.shutdown()
            await runner
            await server.stop()

    asyncio.run(main())


def test_requested_stop_then_load_asks_the_supervisor_to_start_it(root):
    sock = inference_socket_path()
    spec = inference_spec(sock, engine="fake")

    async def main():
        sup = Supervisor([spec], EventLog(), meta={"launcher": "manual"})
        server = await control.serve(sup, socket_path())
        runner = asyncio.create_task(sup.run())
        try:
            for _ in range(600):
                if sup.snapshot()["components"]["inference"]["state"] == "ready":
                    break
                await asyncio.sleep(0.1)
            sup.stop("inference")
            for _ in range(100):
                if sup.snapshot()["components"]["inference"]["state"] == "stopped":
                    break
                await asyncio.sleep(0.1)
            eng = RemoteEngine(sock)
            assert (await asyncio.to_thread(lambda: eng.model)) is None
            await asyncio.to_thread(eng.load, "/models/demo.gguf")  # ensure_worker -> supervisor start
            assert sup.snapshot()["components"]["inference"]["state"] == "ready" and eng.model is not None
        finally:
            sup.shutdown()
            await runner
            await server.stop()

    asyncio.run(main())


def test_sigkill_of_the_worker_is_a_kill_not_a_requested_stop(root):
    sock = inference_socket_path()
    spec = inference_spec(sock, engine="fake")
    spec.backoff_base = 0.2

    async def main():
        sup = Supervisor([spec], EventLog())
        runner = asyncio.create_task(sup.run())
        try:
            for _ in range(600):
                if sup.snapshot()["components"]["inference"]["state"] == "ready":
                    break
                await asyncio.sleep(0.1)
            pid = sup.snapshot()["components"]["inference"]["pid"]
            os.kill(pid, signal.SIGKILL)
            for _ in range(600):
                c = sup.snapshot()["components"]["inference"]
                if c["spawns"] == 2 and c["state"] == "ready":
                    break
                await asyncio.sleep(0.1)
            assert sup.snapshot()["components"]["inference"]["last_exit"]["signal"] == "SIGKILL"
        finally:
            sup.shutdown()
            await runner

    asyncio.run(main())
