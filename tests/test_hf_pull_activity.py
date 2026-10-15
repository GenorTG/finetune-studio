"""HF Pull must create a tracked download job and refresh the model registry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.webui.app import app
from finetune_studio.webui.routes import hf_models


@pytest.fixture
def client_db(tmp_path, monkeypatch):
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    db.init_db()
    hf_models._DOWNLOADS.clear()
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
    client = TestClient(app)
    yield client, local
    hf_models._DOWNLOADS.clear()


def test_hf_download_returns_job_id_and_appears_in_activity(client_db):
    client, local = client_db

    def _fake_worker(job_id: str, repo_id: str, filename, revision: str) -> None:
        dest = local / repo_id.replace("/", "__")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
        (dest / "model.safetensors").write_bytes(b"x" * 1024)
        hf_models._DOWNLOADS[job_id].update({
            "status": "downloading",
            "bytes_done": 0,
            "bytes_total": 1024,
        })
        # Simulate completion path without calling huggingface_hub.
        from finetune_studio import db as _db
        _db.mark_hf_download_running(job_id)
        path = str(dest)
        _db.mark_hf_download_done(job_id, path=path, bytes_total=1024, bytes_done=1024)
        hf_models._DOWNLOADS[job_id].update({
            "status": "completed",
            "bytes_done": 1024,
            "bytes_total": 1024,
            "path": path,
        })
        hf_models._refresh_model_registry()

    with patch.object(hf_models, "_download_worker", side_effect=_fake_worker):
        r = client.post("/api/hf/download", json={"repo_id": "Org/TinyTest"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    assert body.get("job_id"), body

    job_id = body["job_id"]
    assert job_id in hf_models._DOWNLOADS

    # Force the background task path if TestClient didn't run it yet.
    if hf_models._DOWNLOADS[job_id].get("status") == "queued":
        _fake_worker(job_id, "Org/TinyTest", None, "main")

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

    def _worker(job_id: str, repo_id: str, filename, revision: str) -> None:
        dest = local / repo_id.replace("/", "__")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "ok.txt").write_text("done", encoding="utf-8")
        from finetune_studio import db as _db
        _db.mark_hf_download_running(job_id)
        _db.mark_hf_download_done(
            job_id, path=str(dest), bytes_total=4, bytes_done=4
        )
        hf_models._DOWNLOADS[job_id].update({
            "status": "completed",
            "path": str(dest),
            "bytes_done": 4,
            "bytes_total": 4,
        })
        hf_models._refresh_model_registry()

    with patch.object(hf_models, "_download_worker", side_effect=_worker):
        r = client.post("/api/hf/download", json={"repo_id": "Org/RefreshMe"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    if hf_models._DOWNLOADS[job_id].get("status") == "queued":
        _worker(job_id, "Org/RefreshMe", None, "main")

    assert scanned, "registry refresh should call scan_models"
    # HF Explorer cache dir must be included.
    flat = [str(Path(d)) for batch in scanned for d in batch]
    assert any("hf_models" in d or str(local) in d for d in flat), flat


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
