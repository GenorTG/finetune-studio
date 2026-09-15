"""Data Prep UI bridge: review pending Q&A → approve → export for Training.

Browser gap: after Agent/prep, Prepped Q&A listed pending rows with no
visible approve/export controls, so Training still said “No datasets yet”.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.webui.app import app

_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "data_prep.html"
)
_TRAINING = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "project_training.html"
)


@pytest.fixture
def client_and_db(tmp_path, monkeypatch):
    db_path = tmp_path / "fts_test.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    root = tmp_path / "fts_root"
    projects = root / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
    monkeypatch.setattr(qa_fs, "project_dir", lambda pid: projects / pid)
    db.init_db()
    return TestClient(app), db_path, projects


def _create_project(client: TestClient) -> str:
    r = client.post(
        "/api/projects",
        json={"name": f"review-ui-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _seed_pending_pair(pid: str, projects: Path, *, qid: str | None = None) -> str:
    qid = qid or uuid.uuid4().hex[:12]
    (projects / pid / "qa" / "pairs").mkdir(parents=True, exist_ok=True)
    qa_fs.write_qa_pair(
        pid,
        {
            "id": qid,
            "source_id": "src-seed",
            "question": "What is the project code?",
            "answer": "OCTOPUS-7741.",
            "status": "pending",
            "created_at": time.time(),
            "created_via": "test",
        },
    )
    return qid


def test_template_exposes_review_approve_export_bridge() -> None:
    src = _TEMPLATE.read_text(encoding="utf-8")
    assert "dp-bulk-approve" in src
    assert "dp-export-approved" in src
    assert "Approve all pending" in src
    assert "Export approved → Training" in src
    assert "dpBulkAction" in src
    assert "dpExportApproved" in src
    assert "dpRefreshExportedDatasets" in src
    assert "/data-prep/qa/bulk" in src
    assert "/data-prep/export" in src
    assert "Review in Data Editor" in src
    assert "Select in Training" in src
    # Primary surface is a table + controls, not a raw JSON dump.
    assert "dp-results-table" in src
    assert "dp-bridge-status" in src
    export_js = src.split("async function dpExportApproved", 1)[1].split(
        "(function dpWireReviewControls", 1
    )[0]
    assert "JSON.stringify(data" not in export_js
    assert "JSON.stringify(r" not in export_js
    assert "dpBridgeStatus(" in export_js


def test_training_empty_state_points_at_approve_export() -> None:
    html = _TRAINING.read_text(encoding="utf-8")
    assert "Export approved → Training" in html
    assert "approve pending" in html.lower()


def test_data_prep_page_renders_bridge_controls(client_and_db) -> None:
    client, _db_path, _projects = client_and_db
    pid = _create_project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, r.text
    body = r.text
    assert "Export approved → Training" in body
    assert "dp-bulk-approve" in body
    assert "dp-bridge-status" in body
    assert "Open Training" in body
    # Must not dump the QA list as a top-level JSON blob in the shell HTML.
    assert '"items":' not in body.split('id="dp-results"', 1)[1][:400]


def test_approve_then_export_registers_training_dataset(client_and_db) -> None:
    client, _db_path, projects = client_and_db
    pid = _create_project(client)
    qid = _seed_pending_pair(pid, projects)

    listed = client.get(f"/api/projects/{pid}/data-prep/qa").json()["items"]
    assert len(listed) == 1
    assert listed[0]["status"] == "pending"
    assert listed[0]["id"] == qid

    bulk = client.post(
        f"/api/projects/{pid}/data-prep/qa/bulk",
        json={"ids": [qid], "action": "approve"},
    )
    assert bulk.status_code == 200, bulk.text
    assert bulk.json()["ok"] is True
    assert bulk.json()["updated"] == 1

    after = client.get(f"/api/projects/{pid}/data-prep/qa").json()["items"]
    assert after[0]["status"] == "approved"

    before_ds = client.get(f"/api/projects/{pid}/datasets").json()["datasets"]
    assert before_ds == []

    exp = client.get(
        f"/api/projects/{pid}/data-prep/export",
        params={"fmt": "sharegpt", "only": "approved"},
    )
    assert exp.status_code == 200, exp.text
    assert "application/x-ndjson" in (exp.headers.get("content-type") or "")
    lines = [ln for ln in exp.text.splitlines() if ln.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert "conversations" in row

    datasets = client.get(f"/api/projects/{pid}/datasets").json()["datasets"]
    assert len(datasets) >= 1
    hit = next(d for d in datasets if d.get("source") == "data-prep-export")
    assert hit["qa_count"] >= 1
    assert Path(hit["data_path"]).is_file()

    train = client.get(f"/projects/{pid}/training")
    assert train.status_code == 200
    assert "No datasets yet" not in train.text
    assert hit["name"] in train.text or hit["id"] in train.text


def test_export_without_approved_still_empty_or_empty_file(client_and_db) -> None:
    """Pending-only projects must not silently populate Training."""
    client, _db_path, projects = client_and_db
    pid = _create_project(client)
    _seed_pending_pair(pid, projects)

    exp = client.get(
        f"/api/projects/{pid}/data-prep/export",
        params={"fmt": "sharegpt", "only": "approved"},
    )
    assert exp.status_code == 200
    # Empty approved set → empty body; registry may still create a 0-pair file.
    body = exp.text.strip()
    assert body == "" or all("conversations" not in ln for ln in body.splitlines())

    datasets = client.get(f"/api/projects/{pid}/datasets").json()["datasets"]
    if datasets:
        assert all(int(d.get("qa_count") or 0) == 0 for d in datasets)
