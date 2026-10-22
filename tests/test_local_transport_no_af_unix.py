"""Platforms without AF_UNIX (Windows CPython): local servers listen on a loopback port + token, the web app starts.

On a fresh Windows 11 VM (2026-10-22) the web app died in its lifespan (``socket.AF_UNIX`` missing in the supervisor
client) and the inference worker could not bind its Unix socket. ``FTS_NO_AF_UNIX=1`` runs the same path on Linux.
"""
from __future__ import annotations

import http.client
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from finetune_studio.inference import launch
from finetune_studio.inference.remote import RemoteEngine
from finetune_studio.supervisor import client


@pytest.fixture
def no_af_unix(monkeypatch):
    monkeypatch.setattr(client, "HAS_AF_UNIX", False)
    d = Path(tempfile.mkdtemp(prefix="fts"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_missing_server_is_unavailable_not_a_crash(no_af_unix) -> None:
    from finetune_studio.webui.restart_cause import restart_cause

    with pytest.raises(client.SupervisorUnavailable):
        client.SupervisorClient(no_af_unix / "control.sock", timeout=1.0).status()
    assert restart_cause() == ""
    assert launch.worker_alive(no_af_unix / "inference.sock") is False


def test_worker_serves_on_loopback_with_a_token(no_af_unix) -> None:
    sock = no_af_unix / "run" / "inference.sock"
    proc = subprocess.Popen([sys.executable, "-m", "finetune_studio.inference.worker", "--socket", str(sock), "--engine", "fake"],
                            env={**os.environ, "FTS_NO_AF_UNIX": "1", "FTS_ROOT": str(no_af_unix)},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        end = time.monotonic() + 60
        while not launch.worker_alive(sock):
            assert proc.poll() is None, proc.stdout.read().decode()
            assert time.monotonic() < end, "worker never answered on loopback"
            time.sleep(0.1)
        assert not sock.exists() and client.port_file(sock).exists()
        port, _ = client.read_port_file(sock)
        bare = http.client.HTTPConnection("127.0.0.1", port, timeout=5)   # another local process, no token
        bare.request("GET", "/v1/health")
        assert bare.getresponse().status == 403
        eng = RemoteEngine(sock)
        eng.load("/models/demo.gguf", n_ctx=4096)
        assert eng.model_path == "/models/demo.gguf"
        assert eng.generate([{"role": "user", "content": "hi"}])
        eng.unload()
    finally:
        proc.terminate()
        proc.wait(15)
    assert not client.port_file(sock).exists()   # a clean exit removes the port file


def test_job_liveness_without_proc_uses_the_process_table(monkeypatch, tmp_path) -> None:
    """No /proc (Windows): pid_alive read every running job as gone, so describe() wrote 'crashed' at once."""
    from finetune_studio.jobs import store

    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    job_id = "training-20261023-000000-abc123"
    store.job_dir(job_id).mkdir(parents=True)
    runner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "finetune_studio.jobs.runner", "--id", job_id])
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    class NoProc(type(Path())):
        def read_bytes(self):  # type: ignore[override]
            if str(self).startswith("/proc/"):
                raise FileNotFoundError(str(self))
            return super().read_bytes()

        def exists(self, *a, **k):  # type: ignore[override]
            return False if str(self).startswith("/proc") else super().exists(*a, **k)

    monkeypatch.setattr(store, "Path", NoProc)
    try:
        store.write_pid(job_id, runner.pid)
        assert store.pid_alive(job_id) is True
        store.write_pid(job_id, other.pid)          # a recycled pid running something else
        assert store.pid_alive(job_id) is False
        runner.kill()
        runner.wait(10)
        store.write_pid(job_id, runner.pid)
        assert store.pid_alive(job_id) is False
    finally:
        for p in (runner, other):
            p.kill()
            p.wait(10)


def test_worker_exits_when_its_parent_dies_even_without_reparenting(no_af_unix) -> None:
    """Windows keeps getppid() after the parent dies (and a venv python.exe is a launcher): watch the pid itself."""
    sock = no_af_unix / "run" / "inference.sock"
    code = ("import subprocess, sys; p = subprocess.Popen([sys.executable, '-m', 'finetune_studio.inference.worker', "
            f"'--socket', {str(sock)!r}, '--engine', 'fake']); print(p.pid, flush=True); import time; time.sleep(60)")
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "FTS_NO_AF_UNIX": "1"})
    child = int(parent.stdout.readline())
    end = time.monotonic() + 60
    while not launch.worker_alive(sock):
        assert time.monotonic() < end, "worker never answered"
        time.sleep(0.1)
    parent.kill()
    parent.wait(10)
    import psutil
    end = time.monotonic() + 15
    while psutil.pid_exists(child) and psutil.Process(child).status() != psutil.STATUS_ZOMBIE:
        assert time.monotonic() < end, "the worker outlived its parent"
        time.sleep(0.2)
