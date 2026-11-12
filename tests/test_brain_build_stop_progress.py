"""Smart Brain builds can be stopped and followed (2026-10-28, beta work order B1).

Live before: a Qwen3-Embedding build on the CPU ran for 9+ minutes with the page at "0 of 6 files, 0 s"; `fts jobs
stop` and SIGTERM did nothing (the whole corpus was one `encode` call), only SIGKILL ended it; a full GPU showed the
raw "CUDA out of memory" text. Now: embedding goes slice by slice, each slice publishes progress and checks for a
stop; nothing on disk changes until every vector exists, so a stopped or failed build (or a stopped rebuild from
scratch) leaves the old index whole; the page has a Stop button and real elapsed / passages / time left.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.jobs import launch
from finetune_studio.knowledge import index
from finetune_studio.smart_brain import BrainIndex
from finetune_studio.smart_brain.store import encode_batched
from finetune_studio.webui import brain_build_jobs
from tests.knowledge_helpers import add_source, wait_build

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"


def _encode(texts):
    items = [texts] if isinstance(texts, str) else list(texts)
    out = np.zeros((len(items), 8), dtype=np.float32)
    for i, t in enumerate(items):
        out[i, hash(t) % 8] = 1.0
    return out[0] if isinstance(texts, str) else out


class _Gate:
    """An embedder that blocks after its first slice until released, so a test can stop a build mid-way."""

    def __init__(self) -> None:
        self.calls = 0
        self.first_done = threading.Event()
        self.release = threading.Event()

    def get_embedder(self, name: str = "", device: str = "cpu"):
        from finetune_studio.smart_brain.schema import EmbeddingModelInfo

        def encode(texts):
            self.calls += 1
            if self.calls == 2:
                self.first_done.set()
                assert self.release.wait(20), "test never released the embedder"
            return _encode(texts)
        return encode, EmbeddingModelInfo(name=name or "fake", dim=8, normalize=True, distance="cosine")


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> _Gate:
    g = _Gate()
    monkeypatch.setattr("finetune_studio.smart_brain.store.get_embedder", g.get_embedder)
    monkeypatch.setattr("finetune_studio.smart_brain.store.EMBED_SLICE", 2)
    yield g
    g.release.set()


def test_encode_batched_reports_each_slice_and_can_be_stopped() -> None:
    seen: list[tuple[int, int]] = []
    texts = [f"t{i}" for i in range(7)]
    out = encode_batched(_encode, texts, lambda d, t: seen.append((d, t)), slice_size=3)
    assert out.shape == (7, 8) and np.array_equal(out, _encode(texts))
    assert seen == [(0, 7), (3, 7), (6, 7), (7, 7)]

    def stop_at_3(done: int, total: int) -> None:
        if done >= 3:
            raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        encode_batched(_encode, texts, stop_at_3, slice_size=3)


def _big_project(client, name: str) -> str:
    pid = client.post("/api/projects", json={"name": name}).json()["id"]
    for i in range(3):
        add_source(pid, f"f{i}.txt", "\n\n".join(f"File {i} paragraph {j}: the gate code is {i}{j}{j}." * 3
                                                  for j in range(4)))
    return pid


def test_stop_ends_a_build_mid_embedding_and_keeps_the_old_index(client, gate: _Gate, monkeypatch) -> None:
    pid = _big_project(client, "Stop mid-build")
    gate.release.set()
    first = wait_build(client, pid, client.post(f"/api/projects/{pid}/brain/build", json={}))
    assert first["ok"], first
    corpus = index.brain_index_dir(pid)
    before = {p.name: p.read_bytes() for p in corpus.rglob("*") if p.is_file()}

    gate.calls, gate.release = 0, threading.Event()
    gate.first_done.clear()
    r = client.post(f"/api/projects/{pid}/brain/build", json={"reset": True})
    assert r.status_code == 202, r.text
    assert gate.first_done.wait(20)
    snap = client.get(f"/api/projects/{pid}/brain/build/status").json()
    assert snap["phase"] == "embedding" and snap["passages_total"] > 2 and snap["passages_done"] >= 2, snap
    assert snap["elapsed_s"] >= 0
    stop = client.post(f"/api/projects/{pid}/brain/build/stop")
    assert stop.status_code == 200 and stop.json()["stopping"] is True, stop.text
    gate.release.set()
    done = wait_build(client, pid, r)
    assert done["ok"] is False and done["error"].startswith("stopped on request"), done
    assert launch.wait(brain_build_jobs.job_id(done["build_id"]), timeout=10)["exit"]["status"] == "stopped"
    after = {p.name: p.read_bytes() for p in corpus.rglob("*") if p.is_file()}
    assert after == before                                   # the working index is untouched
    assert not corpus.with_name(corpus.name + ".rebuild").exists()
    assert client.post(f"/api/projects/{pid}/brain/build/stop").status_code == 409
    snap = client.get(f"/api/projects/{pid}/brain/build/status").json()
    assert snap["phase"] == "error" and "stopped on request" in snap["message"]


def test_a_rebuild_from_scratch_swaps_in_the_new_index_only_when_it_finished(client, gate: _Gate) -> None:
    pid = _big_project(client, "Reset swap")
    gate.release.set()
    assert wait_build(client, pid, client.post(f"/api/projects/{pid}/brain/build", json={}))["ok"]
    corpus = index.brain_index_dir(pid)
    add_source(pid, "new.txt", "A new file says the depot opens at 06:00.")
    done = wait_build(client, pid, client.post(f"/api/projects/{pid}/brain/build", json={"reset": True}))
    assert done["ok"] and done["documents"] == 4
    assert BrainIndex(corpus).read_manifest().documents == 4
    assert not corpus.with_name(corpus.name + ".rebuild").exists() and not corpus.with_name(corpus.name + ".old").exists()


def test_errors_in_plain_words() -> None:
    from finetune_studio.jobs.runner import Stopped

    assert index.explain_build_error(Stopped("x")).startswith("stopped on request")

    class OutOfMemoryError(RuntimeError):
        pass
    text = index.explain_build_error(OutOfMemoryError("CUDA out of memory. Tried to allocate 188.00 MiB. GPU 0 ..."))
    assert text.startswith("the GPU ran out of memory while embedding")
    assert "smaller embedder" in text and "Settings → Compute device" in text and "CUDA out of memory" in text
    assert index.explain_build_error(ValueError("bad thing")) == "bad thing"


def test_gpu_memory_report_names_the_card_and_its_holders(monkeypatch) -> None:
    import subprocess

    from finetune_studio.smart_brain import devices

    outs = {"--query-gpu=index,uuid,memory.free,memory.total": "0, GPU-a, 100, 24576\n1, GPU-b, 1000, 8192\n",
            "--query-compute-apps=gpu_uuid,process_name,used_memory":
                "GPU-a, /home/u/Comfy/ComfyUI/.venv/bin/python, 17764\nGPU-b, /x/llama-server, 4380\n"
                "GPU-a, /usr/libexec/gnome-remote-desktop-daemon, 260\n"}
    monkeypatch.setattr(devices.shutil if hasattr(devices, "shutil") else __import__("shutil"), "which",
                        lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **_k: type("R", (), {"stdout": outs[cmd[1]]})())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert devices.gpu_memory_report() == " (GPU 0: 0.1 of 24.0 GB free; held by: ComfyUI python 17.3 GB)"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b")
    assert devices.gpu_memory_report() == " (GPU 1: 1.0 of 8.0 GB free; held by: llama-server 4.3 GB)"


def test_the_page_has_a_stop_button_and_real_progress() -> None:
    html = (TEMPLATES / "brain.html").read_text(encoding="utf-8")
    assert 'id="build-stop-btn"' in html and "'/brain/build/stop'" in html
    assert "function buildProgressText(p)" in html and "passages_done" in html and " left'" in html
    assert "'Stopped: the index was left as it was before this build.'" in html
    assert "elapsed_s: elapsed" not in html
    assert "The embedder runs on the CPU: a large one" in html          # warned before a slow CPU build
    assert "buildRate" in html                                          # time left from the embedding rate only


def test_elapsed_comes_from_the_build_row_not_the_stream(client, gate: _Gate) -> None:
    pid = _big_project(client, "Elapsed")
    r = client.post(f"/api/projects/{pid}/brain/build", json={})
    assert gate.first_done.wait(20)
    time.sleep(1.2)
    assert client.get(f"/api/projects/{pid}/brain/build/status").json()["elapsed_s"] >= 1
    gate.release.set()
    assert wait_build(client, pid, r)["ok"]
