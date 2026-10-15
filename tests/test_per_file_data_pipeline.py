"""Per-file upload/parse, readiness from the mining inventory, and the per-file UI (mining itself: tests/test_mining.py)."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.webui.app import app


@pytest.fixture
def pipeline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    db_path = tmp_path / "pipeline.db"
    monkeypatch.setattr(settings, "db_path", str(db_path))
    root = tmp_path / "fts"
    projects = root / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
    monkeypatch.setattr(qa_fs, "project_dir", lambda pid: projects / pid)
    db.init_db()
    return TestClient(app), projects


def _project(client: TestClient) -> str:
    response = client.post(
        "/api/projects",
        json={"name": f"per-file-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_bulk_upload_stages_parser_supported_files_for_background_parse(
    pipeline_env: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _projects = pipeline_env
    pid = _project(client)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "finetune_studio.webui.routes.file_library._parse_source_background",
        lambda project_id, source_id: calls.append((project_id, source_id)),
    )

    response = client.post(
        f"/api/projects/{pid}/files/upload",
        files=[
            ("files", ("one.md", b"# One\n\nA grounded sentence for parsing.", "text/markdown")),
            ("files", ("two.txt", b"A second grounded sentence for parsing.", "text/plain")),
        ],
    )
    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert [row["parse_status"] for row in report] == ["queued", "queued"]
    assert len(calls) == 2
    assert {source_id for _, source_id in calls} == {
        row["source_id"] for row in report
    }


def test_upload_streams_large_files_in_bounded_chunks() -> None:
    route = (
        Path(__file__).parents[1]
        / "src/finetune_studio/webui/routes/file_library.py"
    ).read_text(encoding="utf-8")
    assert "_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024" in route
    assert "await upload.read(_UPLOAD_CHUNK_BYTES)" in route
    assert "await up.read()" not in route
    assert "write_staged_upload" in route


def test_data_prep_ui_exposes_per_file_mining_and_assembler() -> None:
    template = (
        Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates" / "data_prep.html"
    ).read_text(encoding="utf-8")
    for needle in ("Mine pairs, file by file", "dp-mine-selected", "dp-fill-selected", "Build dataset from selected",
                   "dpMiningBase() + '/jobs'", "/datasets/subset", "dpEnsureHelperLoaded", "dp-source-check", "dp-file-panel",
                   "'/pairs/'", "Rounds"):
        assert needle in template, needle
    for gone in ("start-bulk", "data-prep/start", "prep-qpc", "dp-fill-unreviewed", "fill-preview", "Approved coverage",
                 "Run a prep job", "dp-generate-selected"):
        assert gone not in template, gone


def test_sources_endpoint_reports_readiness_from_the_inventory(
    pipeline_env: tuple[TestClient, Path],
) -> None:
    client, _projects = pipeline_env
    pid = _project(client)
    from tests.knowledge_helpers import add_source

    source = add_source(pid, "src-one.md", "# One\n\nThis answer is the grounded fact of the file.").as_dict()
    row = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"][0]
    assert row["parse_status"] == "ready" and row["mining"]["status"] == "parsed" and not row["mining"]["training_ready"]

    db.mining.insert_items(pid, source["id"], "job1", [
        {"chunk_idx": 1, "origin": "agent", "kind": "fact", "text": "Grounded fact", "values": ["this answer"]}])
    qa_fs.write_qa_pair(pid, {"id": "qa-ready", "source_id": source["id"], "chunk_idx": 1,
                              "question": "What is grounded?", "answer": "This answer.", "status": "pending"})
    row = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"][0]
    assert row["mining"]["status"] == "review" and row["mining"]["counts"]["pending"] == 1

    qa_fs.update_qa_pair(pid, "qa-ready", status="approved")
    row = client.get(f"/api/projects/{pid}/data-prep/sources").json()["sources"][0]
    assert row["mining"]["status"] == "complete" and row["mining"]["training_ready"] is True
    assert row["mining"]["coverage_pct"] == 100.0 and row["mining"]["pairs"]["approved"] == 1
