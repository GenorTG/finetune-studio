"""HF Pull must create a tracked download job and refresh the model registry."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.jobs import launch
from finetune_studio.webui import download_jobs
from finetune_studio.webui.app import app
from finetune_studio.webui.routes import hf_models


@pytest.fixture
def client_db(tmp_path, monkeypatch):
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    local = tmp_path / "hf_models"
    local.mkdir()
    monkeypatch.setattr(hf_models, "_LOCAL", local)
    # Keep registry refresh cheap / isolated.
    monkeypatch.setattr(
        settings,
        "model_dirs",
        [str(tmp_path / "models")],
    )
    monkeypatch.setattr(
        settings,
        "model_dirs_extra",
        [str(local)],
    )
    (tmp_path / "models").mkdir()
    # The download is a job (in-process under the test suite): the hub is faked where the job calls it.
    monkeypatch.setattr(download_jobs, "_remote_size", lambda repo_id, filename, revision: 1024)
    client = TestClient(app)
    yield client, local


def _fake_hub(repo_id: str, filename: str, revision: str, dest: str) -> str:
    d = Path(dest)
    (d / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"x" * 1024)
    return str(d)


def _wait(client: TestClient, job_id: str, timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        p = client.get("/api/hf/download/progress", params={"job_id": job_id}).json()
        if p["status"] in download_jobs.TERMINAL_STATUSES:
            return p
        time.sleep(0.05)
    raise AssertionError(f"download {job_id} still running: {p}")


def test_hf_download_returns_job_id_and_appears_in_activity(client_db, monkeypatch):
    client, local = client_db
    monkeypatch.setattr(download_jobs, "_hub_fetch", _fake_hub)
    r = client.post("/api/hf/download", json={"repo_id": "Org/TinyTest"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    assert body.get("job_id"), body

    job_id = body["job_id"]
    final = _wait(client, job_id)
    assert final["status"] == "completed", final
    assert final["bytes_done"] == final["bytes_total"] > 0
    assert final["path"] == str(local / "Org__TinyTest")

    activity = client.get("/api/activity").json()
    downloads = [t for t in activity["tasks"] if t.get("kind") == "download"]
    assert any(t.get("id") == job_id for t in downloads), activity
    match = next(t for t in downloads if t.get("id") == job_id)
    assert match["status"] in ("done", "running", "queued")
    assert "TinyTest" in (match.get("project_name") or "")


def test_hf_download_completion_refreshes_registry(client_db, monkeypatch):
    client, local = client_db
    scanned: list[list[str]] = []

    def _scan(dirs: list) -> list:
        scanned.append(list(dirs))
        return []

    monkeypatch.setattr(
        "finetune_studio.models.registry.scan_models",
        _scan,
    )
    monkeypatch.setattr(download_jobs, "_hub_fetch", _fake_hub)
    r = client.post("/api/hf/download", json={"repo_id": "Org/RefreshMe"})
    assert r.status_code == 200, r.text
    assert _wait(client, r.json()["job_id"])["status"] == "completed"

    assert scanned, "registry refresh should call scan_models"
    # HF Explorer cache dir must be included.
    flat = [str(Path(d)) for batch in scanned for d in batch]
    assert any("hf_models" in d or str(local) in d for d in flat), flat


def test_hf_download_failure_is_a_failed_row_with_the_hub_error(client_db, monkeypatch):
    client, _local = client_db

    def _boom(repo_id: str, filename: str, revision: str, dest: str) -> str:
        raise RuntimeError("404 Client Error: Repository Not Found")

    monkeypatch.setattr(download_jobs, "_hub_fetch", _boom)
    r = client.post("/api/hf/download", json={"repo_id": "Org/Missing", "filename": "x.gguf"})
    final = _wait(client, r.json()["job_id"])
    assert final["status"] == "error" and "Repository Not Found" in final["error"], final


def test_hf_download_cancel_stops_it_and_unknown_ids_are_404(client_db, monkeypatch):
    client, _local = client_db
    release = threading.Event()

    def _slow(repo_id: str, filename: str, revision: str, dest: str) -> str:
        release.wait(20)
        return _fake_hub(repo_id, filename, revision, dest)

    monkeypatch.setattr(download_jobs, "_hub_fetch", _slow)
    job_id = client.post("/api/hf/download", json={"repo_id": "Org/Slow"}).json()["job_id"]
    deadline = time.monotonic() + 10
    while client.get("/api/hf/download/progress", params={"job_id": job_id}).json()["status"] != "downloading":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    r = client.post("/api/hf/download/cancel", params={"job_id": job_id})
    assert r.status_code == 200 and r.json()["status"] == "cancelled", r.text
    launch.wait(download_jobs.job_id(job_id), timeout=10)
    release.set()
    p = client.get("/api/hf/download/progress", params={"job_id": job_id}).json()
    assert p["status"] == "cancelled" and p["running"] is False, p
    assert client.post("/api/hf/download/cancel", params={"job_id": "nope0000"}).status_code == 404
    assert client.get("/api/hf/download/progress", params={"job_id": "nope0000"}).status_code == 404


def test_startup_reconcile_fails_dead_downloads_and_keeps_running_ones(client_db, monkeypatch):
    dead = db.create_hf_download("Org/Dead", "")
    alive = db.create_hf_download("Org/Alive", "")
    monkeypatch.setattr(download_jobs, "running_download_ids", lambda: {alive["id"]})
    assert hf_models.restore_in_progress_downloads() == 1
    assert db.get_hf_download(dead["id"])["status"] == "error"
    assert db.get_hf_download(alive["id"])["status"] == "queued"


def test_app_js_opens_activity_on_job_id() -> None:
    src = (
        Path(__file__).resolve().parents[1]
        / "src/finetune_studio/webui/static/js/app.js"
    ).read_text(encoding="utf-8")
    assert "d.job_id" in src
    assert "ftsActivity.open" in src
    assert "Download queued" in src


def test_config_includes_hf_explorer_cache() -> None:
    extras = list(settings.model_dirs_extra)
    assert any("hf_models" in d for d in extras)
