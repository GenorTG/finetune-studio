"""Training and exports (merge, GGUF, abliterated, Heretic) wait for the GPU like builds and picture reading; a Chat
model load is refused at once and names what holds the GPU (2026-11-17).

Before: training and every export answered 409 ``gpu_busy`` while a Smart Brain build or a picture read held the GPU
lease, and the user had to retry by hand. Now they join the lease queue (``JobSpec.queue_gpu``) and start by
themselves; the pages say *waiting* and behind what; a stop while waiting reads as stopped / cancelled. An
interactive model load does not queue (someone waiting to chat must not sit behind a 20-minute build): it answers
409 ``gpu_busy`` with the job, its plain-words label and how long it has been running.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune_studio import db
from finetune_studio.jobs import launch, store
from finetune_studio.jobs.store import JobSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def holder(tmp_path, monkeypatch):
    """A long GPU job in this process holding the lease (a build, as far as the queue is concerned)."""
    monkeypatch.setenv("FTS_JOBS_INPROCESS", "1")
    monkeypatch.setattr(launch, "LOCAL_QUEUE_POLL_S", 0.05)
    spec = JobSpec(id=store.new_id("selftest"), kind="selftest", params={"steps": 400, "delay_s": 0.05}, gpu=True,
                   stop_grace_s=5, title="holder")
    launch.start(spec)
    yield spec
    launch.stop(spec.id)
    launch.wait(spec.id, timeout=30)


def _wait(fn, timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(0.05)
    return False


# ── training ─────────────────────────────────────────────────────────────────

def test_training_queues_behind_a_gpu_job_and_says_so(holder, monkeypatch) -> None:
    from finetune_studio.training import remote
    from finetune_studio.training.engine import TrainingConfig

    monkeypatch.setattr("finetune_studio.smart_brain.model_cache.release_brain_models", lambda why: None)
    pid = db.create_project(name="q", description="")["id"]
    rid = db.create_run(project_id=pid, name="r")["id"]
    eng = remote.RemoteTrainingEngine()
    jid = eng.start(TrainingConfig(), [{"messages": [{"role": "user", "content": "hi"}]}], run_id=rid,
                    output_dir="out", project_id=pid)        # before: JobRejected 409 gpu_busy
    assert store.read_spec(jid).queue_gpu is True
    st = eng.state
    assert st.status == "waiting" and st.message.startswith("Waiting for the GPU: ")
    assert "starts by itself" in st.message and eng.waiting_for()
    with pytest.raises(remote.TrainingBusy):                 # one run at a time: a waiting run counts
        eng.start(TrainingConfig(), [], run_id=rid, output_dir="out", project_id=pid)
    eng.stop()
    assert _wait(lambda: not launch.describe(jid)["running"])
    eng._cache = None
    st = eng.state
    assert st.status == "stopped" and st.message == remote.STOPPED_WAITING
    assert db.get_run(rid)["status"] == "stopped"


def test_the_training_page_shows_waiting() -> None:
    page = (ROOT / "src/finetune_studio/webui/templates/project_training.html").read_text()
    assert "const ACTIVE = { waiting: 1, loading: 1, training: 1, saving: 1 };" in page
    assert 'waiting: "solid-amber"' in page
    js = (ROOT / "src/finetune_studio/webui/static/js/app.js").read_text()
    assert '"Training waits for the GPU: " + (d.waiting_for' in js
    route = (ROOT / "src/finetune_studio/webui/routes/training.py").read_text()
    assert '"waiting_for_gpu": True, "waiting_for": waiting' in route
    from finetune_studio.training.monitor import ACTIVE_STATUSES
    assert "waiting" in ACTIVE_STATUSES


# ── exports (merge / GGUF / abliterated / Heretic: one export job kind) ───────

def test_an_export_waits_for_the_gpu_and_a_cancel_while_waiting_reads_cancelled(holder) -> None:
    from finetune_studio.webui import export_jobs, export_service

    pid = db.create_project(name="e", description="")["id"]
    rid = db.create_run(project_id=pid, name="r")["id"]
    row = export_jobs.start_job(project_id=pid, run_id=rid, fmt="gguf", quant="Q4_K_M", quants=None,
                                work=("single_quant_work", {}))      # before: ExportBusy (409)
    eid = row["id"]
    assert row["status"] == "queued" and row["phase"] == export_jobs.WAITING_PHASE
    assert row["phase_detail"].endswith("it starts by itself when the GPU is free")
    assert store.read_spec(export_jobs.job_id(eid)).queue_gpu is True
    view = export_service.row_view(db.get_export(eid))
    assert view["waiting_for_gpu"] is True and view["phase"] == "waiting_gpu"
    assert export_jobs.cancel_job(eid) is True
    done = db.get_export(eid)
    assert done["status"] == "cancelled" and done["error"] == export_jobs.STOPPED_WAITING
    assert _wait(lambda: not launch.describe(export_jobs.job_id(eid))["running"])


def test_the_export_page_shows_waiting() -> None:
    page = (ROOT / "src/finetune_studio/webui/templates/export_models.html").read_text()
    assert "waiting_gpu: 'Waiting for the GPU'" in page
    assert "const waiting = row.status === 'queued' && row.phase === 'waiting_gpu';" in page


# ── a chat model load is refused at once, naming the holder ────────────────────

def _gpu(monkeypatch, is_gpu: bool) -> None:
    import finetune_studio.accel as accel_pkg
    monkeypatch.setattr(accel_pkg, "get_accelerator", lambda refresh=False: SimpleNamespace(is_gpu=is_gpu))


def test_the_refusal_names_the_job_and_how_long_it_runs(holder, monkeypatch) -> None:
    from finetune_studio.webui.chat_load_guard import gpu_busy_refusal

    _gpu(monkeypatch, True)
    body = gpu_busy_refusal()
    assert body is not None and body["code"] == "gpu_busy" and body["gpu_holder"]["job_id"] == holder.id
    assert body["detail"].startswith("A chat model cannot load: the GPU is busy — a selftest job is using it (for ")
    assert f"job {holder.id}" in body["detail"] and f"fts jobs stop {holder.id}" in body["detail"]
    _gpu(monkeypatch, False)                       # CPU-only: the model loads into RAM, nothing to refuse
    assert gpu_busy_refusal() is None


def test_no_refusal_when_the_gpu_is_free(monkeypatch) -> None:
    from finetune_studio.webui.chat_load_guard import gpu_busy_refusal

    _gpu(monkeypatch, True)
    monkeypatch.setattr(launch, "gpu_holder", lambda exclude="": None)
    assert gpu_busy_refusal() is None


def test_label_and_duration_words() -> None:
    assert launch.holder_label("training job t-1 (training run r)") == "a training run is running"
    assert launch.holder_label("export job e-1 (export gguf)") == "an export is running"
    assert launch.duration_text(42) == "42 s" and launch.duration_text(12 * 60 + 5) == "12 min"
    assert launch.duration_text(3600 + 5 * 60) == "1 h 05 min"


@pytest.mark.parametrize(("url", "key"), [("/api/models/load", "path"), ("/api/chat-v2/load", "model_path")])
def test_chat_load_routes_answer_409_gpu_busy(client, tmp_path, monkeypatch, url, key) -> None:
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}")
    info = {"job_id": "training-x", "kind": "training", "title": "training run r", "label": "a training run is running",
            "running_s": 754}
    monkeypatch.setattr(launch, "gpu_busy_info", lambda: info)
    _gpu(monkeypatch, True)
    loads = []
    from finetune_studio.webui import app as app_mod
    monkeypatch.setattr(app_mod.inference_engine, "load", lambda *a, **k: loads.append(a), raising=False)
    r = client.post(url, json={key: str(model)})
    assert r.status_code == 409 and r.json()["code"] == "gpu_busy"
    assert r.json()["detail"] == ("A chat model cannot load: the GPU is busy — a training run is running (for 12 min, "
                                  "job training-x). Load it when that ends, or stop that job on its page (or "
                                  "`fts jobs stop training-x`).")
    assert loads == []


def test_the_chat_pages_show_the_refusal() -> None:
    inf = (ROOT / "src/finetune_studio/webui/templates/inference.html").read_text()
    assert "if (e && e.code === 'gpu_busy') {" in inf and "note.id = 'load-gpu-busy';" in inf
    chat = (ROOT / "src/finetune_studio/webui/templates/chat_v2.html").read_text()
    assert "e && e.code === 'gpu_busy' ? e.message" in chat
