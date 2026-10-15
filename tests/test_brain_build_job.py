"""Smart Brain index builds run as their own job (2026-10-15): the routes answer 202 + build id, the build row carries
the outcome (the real error when it fails), one build per project at a time, a web restart leaves a running build
alone, and a build job that died without finishing its row is reported as failed. The last test crosses the real
process boundary (the rest of the suite runs jobs in-process, ``jobs.launch``)."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.jobs import launch, store
from finetune_studio.knowledge import index
from finetune_studio.webui import brain_build_jobs
from tests.knowledge_helpers import add_source, wait_build


def _project(client, name: str = "Build job") -> str:
    return client.post("/api/projects", json={"name": name}).json()["id"]


class _GatedIndex:
    """A BrainIndex whose build waits for ``gate`` (or raises ``fail``)."""

    gate = threading.Event()
    fail: str = ""

    def __init__(self, d: Path) -> None:
        self.dir = Path(d)

    def build_from_documents(self, docs, **_kw: object) -> dict:
        assert self.gate.wait(20), "test never released the build"
        if self.fail:
            raise RuntimeError(self.fail)
        return {"documents": len(docs), "chunks": 3, "vector_dim": 8, "skipped": 0}


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch) -> type[_GatedIndex]:
    cls = type("Gated", (_GatedIndex,), {"gate": threading.Event(), "fail": ""})
    monkeypatch.setattr("finetune_studio.smart_brain.BrainIndex", cls)
    yield cls
    cls.gate.set()


def test_build_is_a_job_one_per_project_and_the_progress_follows_its_row(client, gated) -> None:
    pid = _project(client)
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    r = client.post(f"/api/projects/{pid}/brain/build", json={})
    assert r.status_code == 202, r.text
    body = r.json()
    bid = body["build_id"]
    assert body["building"] is True and body["queued_files"] == 1
    assert body["status_url"] == f"/api/projects/{pid}/brain/builds/{bid}"
    assert store.read_spec(brain_build_jobs.job_id(bid)).project_id == pid

    snap = client.get(f"/api/projects/{pid}/brain/build/status").json()
    assert snap["phase"] in ("queued", "embedding") and snap["build_id"] == bid
    busy = client.post(f"/api/projects/{pid}/brain/build", json={})
    assert busy.status_code == 409 and busy.json()["code"] == "build_running", busy.text
    assert busy.json()["build_id"] == bid
    row = client.get(f"/api/projects/{pid}/brain/builds/{bid}").json()
    assert row["finished"] is False and row["running"] is True

    gated.gate.set()
    done = wait_build(client, pid, r)
    assert done["ok"] and done["status"] == "done" and done["documents"] == 1 and done["chunks"] == 3
    assert launch.wait(brain_build_jobs.job_id(bid), timeout=10)["exit"]["status"] == "done"
    assert brain_build_jobs.running_build_ids() == set()
    assert client.get(f"/api/projects/{pid}/brain/builds/nope1234").status_code == 404


def test_a_failed_build_is_a_failed_row_with_the_real_error(client, gated) -> None:
    pid = _project(client)
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    gated.fail = "CUDA out of memory while embedding"
    gated.gate.set()
    r = client.post(f"/api/projects/{pid}/brain/build", json={})
    b = wait_build(client, pid, r)
    assert b["ok"] is False and b["status"] == "error" and "CUDA out of memory" in b["error"], b
    snap = client.get(f"/api/projects/{pid}/brain/build/status").json()
    assert snap["phase"] == "error" and "CUDA out of memory" in snap["message"]


def test_quick_answers_202_with_the_build_and_a_build_that_cannot_start_is_build_failed(client, gated) -> None:
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/brain/quick", json={})
    assert r.status_code == 400 and r.json()["code"] == "build_failed", r.text
    assert "no parsed files" in r.json()["detail"]
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    gated.gate.set()
    r = client.post(f"/api/projects/{pid}/brain/quick", json={})
    assert r.status_code == 202, r.text
    assert r.json()["build"]["build_id"] == r.json()["build_id"]
    assert wait_build(client, pid, r)["ok"]


def test_a_gpu_busy_rejection_fails_the_row_and_answers_409(client, monkeypatch) -> None:
    pid = _project(client)
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")

    def _busy(_spec):
        raise launch.JobRejected(409, "the GPU is busy with training job t-1 (run); wait for it or stop it",
                                 "gpu_busy")

    monkeypatch.setattr(launch, "start", _busy)
    r = client.post(f"/api/projects/{pid}/brain/build", json={})
    assert r.status_code == 409 and r.json()["code"] == "gpu_busy", r.text
    row = db.brain_builds.latest_for_project(pid)
    assert row["status"] == "error" and "GPU is busy" in row["error"]


def test_a_job_that_died_without_finishing_its_row_is_reported_failed(client) -> None:
    pid = _project(client)
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    bid = index.prepare(pid)["build_id"]
    jid = brain_build_jobs.job_id(bid)
    store.write_spec(store.JobSpec(id=jid, kind=brain_build_jobs.KIND, project_id=pid, params={"build_id": bid}))
    store.write_exit(jid, {"status": "crashed", "detail": "the job process is gone without an exit record"})
    b = client.get(f"/api/projects/{pid}/brain/builds/{bid}").json()
    assert b["finished"] and b["ok"] is False and "gone without an exit record" in b["error"], b


def test_startup_reconcile_keeps_builds_whose_job_is_alive(client) -> None:
    pid = _project(client)
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    alive, dead = index.prepare(pid)["build_id"], index.prepare(pid)["build_id"]
    assert db.reconcile_stale_brain_builds("interrupted by service restart", keep={alive}) == 1
    assert db.get_brain_build(alive)["status"] == "queued"
    assert db.get_brain_build(dead)["status"] == "failed"


def test_a_build_runs_in_its_own_process_and_reports_its_real_error(client, tmp_path, monkeypatch) -> None:
    """Real job process, CPU, offline: an embedder that cannot load fails the row with the loader's own error."""
    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)
    monkeypatch.setenv("FTS_DEVICE", "cpu")              # never touch a GPU from a test's job process
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")            # nothing is downloaded
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.chdir(tmp_path)                          # a stray cwd-relative DB would land here
    pid = _project(client, "Build process")
    add_source(pid, "a.txt", "File A says the gate closes at 22:00.")
    r = client.post(f"/api/projects/{pid}/brain/build",
                    json={"embedder": "embedder_local:" + str(tmp_path / "no-such-embedder")})
    assert r.status_code == 202, r.text
    bid = r.json()["build_id"]
    jid = brain_build_jobs.job_id(bid)
    try:
        d = launch.wait(jid, timeout=180)
    finally:
        if launch.describe(jid)["running"]:
            launch.stop(jid)
    assert d["pid"] != os.getpid() and d["exit"]["status"] == "failed", d
    b = client.get(f"/api/projects/{pid}/brain/builds/{bid}").json()
    assert b["finished"] and b["ok"] is False and b["error"], b
    assert "no-such-embedder" in b["error"] or "offline" in b["error"].lower(), b
    assert not (tmp_path / "data").exists(), "the job used a cwd-relative DB instead of the web app's"
