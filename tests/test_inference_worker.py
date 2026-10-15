"""The inference worker process and its proxy: same interface as InferenceEngine, honest errors, crash = reported, not reloaded."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from finetune_studio.inference import launch

_real_spawn = launch._spawn_private  # captured at import, before the conftest guard replaces it
from finetune_studio.inference.remote import (
    EngineCrashed,
    RemoteEngine,
    WorkerUnavailable,
)


@pytest.fixture
def root(monkeypatch):
    d = tempfile.mkdtemp(prefix="fts")  # AF_UNIX paths are limited to ~100 chars
    monkeypatch.setenv("FTS_ROOT", d)
    yield Path(d)
    launch.stop_private_worker()
    shutil.rmtree(d, ignore_errors=True)


def start_fake_worker(sock: Path) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "finetune_studio.inference.worker", "--socket", str(sock), "--engine", "fake"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    end = time.monotonic() + 60
    while time.monotonic() < end:
        if launch.worker_alive(sock):
            return proc
        assert proc.poll() is None, proc.stdout.read().decode()
        time.sleep(0.1)
    raise AssertionError("fake worker never answered")


@pytest.fixture
def worker(root):
    sock = root / "run" / "inference.sock"
    proc = start_fake_worker(sock)
    yield sock, proc
    if proc.poll() is None:
        proc.terminate()
        proc.wait(10)


def test_no_worker_means_no_model_and_no_spawn_on_reads(root):
    eng = RemoteEngine(root / "run" / "inference.sock")
    assert eng.model is None and eng.model_path is None and eng.n_ctx is None and eng.idle_seconds == 0
    assert eng._busy == 0 and eng._last_used == 0.0 and eng._loading_path is None
    assert "no inference worker" in eng.last_error
    with pytest.raises(RuntimeError, match="No model loaded"):
        eng.generate([{"role": "user", "content": "hi"}])
    eng.unload()  # nothing running: a no-op, not an error
    assert launch._private is None


def test_load_generate_attributes_chat_completion_unload(worker):
    sock, proc = worker
    eng = RemoteEngine(sock)
    eng.load("/models/demo.gguf", n_ctx=8192)
    assert eng.model is not None and eng.model_path == "/models/demo.gguf" and eng.is_gguf and eng.n_ctx == 8192
    assert eng.offload["total_layers"] == 32 and eng.vision is False
    assert eng.generate([{"role": "user", "content": "hello"}], max_tokens=8) == "echo: hello"
    out = eng.model.create_chat_completion(messages=[{"role": "user", "content": "direct"}], max_tokens=8)
    assert out["choices"][0]["message"]["content"] == "echo: direct"
    assert eng.worker_info()["pid"] == proc.pid and eng.worker_info()["transport"] == "shared"
    eng.unload()
    assert eng.model is None
    with pytest.raises(RuntimeError, match="No model loaded"):
        eng.generate([{"role": "user", "content": "hello"}])


def test_load_failure_is_the_workers_real_error_and_leaves_nothing_loaded(worker):
    sock, _ = worker
    eng = RemoteEngine(sock)
    with pytest.raises(RuntimeError, match="model not found"):
        eng.load("/models/nope.missing")
    assert eng.model is None


def test_crash_mid_generate_raises_engine_crashed_and_the_model_is_gone(worker):
    sock, proc = worker
    eng = RemoteEngine(sock)
    eng.load("/models/demo.gguf")
    with pytest.raises(EngineCrashed, match="died during the generation"):
        eng.generate([{"role": "user", "content": "__crash__"}])
    proc.wait(10)
    assert proc.returncode < 0 and signal.Signals(-proc.returncode) is signal.SIGABRT
    assert eng.model is None  # dead socket: no model, no automatic reload


def test_crash_during_load_is_reported_with_the_model_name(worker):
    sock, proc = worker
    eng = RemoteEngine(sock)
    with pytest.raises(EngineCrashed, match="load of boom.crash"):
        eng.load("/models/boom.crash")
    proc.wait(10)


def test_assigning_model_none_unloads_like_the_old_engine(worker):
    sock, _ = worker
    eng = RemoteEngine(sock)
    eng.load("/models/demo.gguf")
    eng.model = None
    assert eng.model is None


def test_private_worker_is_spawned_once_and_stopped(root, monkeypatch):
    sock = root / "run" / "inference.sock"
    # A private worker runs the REAL engine by default; the conftest guard blocks that, so go through the
    # seam explicitly with the fake engine.
    spawn = launch._spawn_private.__wrapped__ if hasattr(launch._spawn_private, "__wrapped__") else _real_spawn
    monkeypatch.setattr(launch, "_spawn_private", lambda sock, extra_args=(): spawn(sock, ("--engine", "fake")))
    assert launch.ensure_worker(sock) == "private"
    pid = launch._private.pid
    assert launch.ensure_worker(sock) == "shared" and launch._private.pid == pid
    eng = RemoteEngine(sock)
    eng.load("/models/demo.gguf")
    assert eng.worker_info()["transport"] == "private"
    launch.stop_private_worker()
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not launch.worker_alive(sock)


def test_worker_exits_when_its_parent_dies(root):
    sock = root / "run" / "inference.sock"
    code = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-m', 'finetune_studio.inference.worker', '--socket', {str(sock)!r}, '--engine', 'fake'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    wpid = int(parent.stdout.readline())
    end = time.monotonic() + 60
    while time.monotonic() < end and not launch.worker_alive(sock):
        time.sleep(0.1)
    assert launch.worker_alive(sock)
    parent.kill()
    parent.wait()
    end = time.monotonic() + 10
    while time.monotonic() < end:
        try:
            os.kill(wpid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    os.kill(wpid, signal.SIGKILL)
    raise AssertionError("worker survived its parent")


def test_worker_unavailable_type_is_distinct():
    assert issubclass(WorkerUnavailable, RuntimeError) and issubclass(EngineCrashed, RuntimeError)
