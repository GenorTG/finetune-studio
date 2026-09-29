"""Per-file parse, Q&A generation, readiness, and dataset assembly workflow."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db
from finetune_studio.config import settings
from finetune_studio.data.fs import file_library as fl
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.data.prep.source_state import summarize_source
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
    fl._PARSED_CACHE.clear()
    db.init_db()
    return TestClient(app), projects


def _project(client: TestClient) -> str:
    response = client.post(
        "/api/projects",
        json={"name": f"per-file-{uuid.uuid4().hex[:6]}", "base_model": "x/test"},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _source(source_id: str = "src-one", chunks: int = 2) -> dict[str, Any]:
    return {
        "id": source_id,
        "filename": f"{source_id}.md",
        "name": f"{source_id}.md",
        "sha256": source_id * 4,
        "status": "ready",
        "chunk_count": chunks,
        "char_count": 100,
        "uploaded_at": time.time(),
    }


def test_training_ready_is_derived_from_approved_chunk_coverage(
    pipeline_env: tuple[TestClient, Path],
) -> None:
    client, _projects = pipeline_env
    pid = _project(client)
    source = _source(chunks=2)
    qa_fs.write_qa_source(pid, source)

    parsed = summarize_source(pid, source)
    assert parsed["stage"] == "parsed"
    assert parsed["training_ready"] is False

    qa_fs.write_qa_pair(
        pid,
        {
            "id": "qa-one",
            "source_id": source["id"],
            "chunk_idx": 1,
            "question": "Q1?",
            "answer": "A1",
            "status": "approved",
        },
    )
    partial = summarize_source(pid, source)
    assert partial["coverage_pct"] == 50.0
    assert partial["stage"] == "needs_review"

    qa_fs.write_qa_pair(
        pid,
        {
            "id": "qa-two",
            "source_id": source["id"],
            "chunk_idx": 2,
            "question": "Q2?",
            "answer": "A2",
            "status": "approved",
        },
    )
    ready = summarize_source(pid, source)
    assert ready["coverage_pct"] == 100.0
    assert ready["training_ready"] is True
    assert ready["stage"] == "training_ready"


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


def test_bulk_generation_queues_source_scoped_lazy_jobs(
    pipeline_env: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, projects = pipeline_env
    pid = _project(client)
    source_ids: list[str] = []
    for index in range(2):
        source_id = f"src-{index}"
        raw = projects / pid / f"{source_id}.md"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(f"Source {index} grounded content.", encoding="utf-8")
        source = {
            **_source(source_id, chunks=1),
            "data_path": str(raw),
            "path": str(raw),
        }
        qa_fs.write_qa_source(pid, source)
        source_ids.append(source_id)

    monkeypatch.setattr(
        "finetune_studio.data.prep.generator.resolve_generator",
        lambda: object(),
    )
    monkeypatch.setattr(
        "finetune_studio.webui.routes.data_prep._run_prep_background",
        lambda *args: None,
    )
    response = client.post(
        f"/api/projects/{pid}/data-prep/start-bulk",
        json={"source_ids": source_ids, "qa_per_chunk": 2},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["queued"] == 2
    assert len(body["run_ids"]) == 2

    from finetune_studio.webui.routes import data_prep

    for run_id in body["run_ids"]:
        queued = data_prep._RUNS[(pid, run_id)]["runner"]
        assert queued._delegate is None
        assert queued.path.is_file()


def test_resume_stale_data_prep_runs_requeues_instead_of_failing(
    pipeline_env: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart used to mark every in-flight run `failed` outright
    (`db.reconcile_stale_data_prep`), silently dropping the rest of a
    multi-file batch. `resume_stale_data_prep_runs` must instead re-derive
    the source path + settings from the durable `data_prep_runs` row and
    re-enqueue the job — only a run whose source file vanished should still
    fail."""
    client, projects = pipeline_env
    pid = _project(client)

    raw = projects / pid / "resumable.md"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("Resumable source grounded content.", encoding="utf-8")
    source = {
        **_source("src-resumable", chunks=1),
        "data_path": str(raw),
        "path": str(raw),
    }
    qa_fs.write_qa_source(pid, source)

    resumable_run = db.create_data_prep_run(
        project_id=pid, filename="resumable.md", byte_count=raw.stat().st_size,
        source_id=source["id"],
        settings_obj={"source_id": source["id"], "qa_per_chunk": 2,
                      "difficulty": "hard", "style": "direct"},
    )
    db.mark_data_prep_running(resumable_run["id"])

    orphan_run = db.create_data_prep_run(
        project_id=pid, filename="gone.md", byte_count=10,
        source_id="src-does-not-exist",
        settings_obj={"source_id": "src-does-not-exist"},
    )

    calls: list[str] = []
    monkeypatch.setattr(
        "finetune_studio.webui.routes.data_prep._run_prep_background",
        lambda run_id, runner, log: calls.append(run_id),
    )

    from finetune_studio.webui.routes.data_prep import (
        _RUNS,
        resume_stale_data_prep_runs,
    )

    import asyncio

    async def _drive() -> dict[str, int]:
        result = await resume_stale_data_prep_runs()
        # Let the resumed background task (scheduled via asyncio.create_task)
        # actually run before the loop closes.
        await asyncio.sleep(0.05)
        return result

    outcome = asyncio.run(_drive())
    assert outcome == {"resumed": 1, "failed": 1}

    resumed_row = db.get_data_prep_run(resumable_run["id"])
    assert resumed_row["status"] == "queued"
    queued = _RUNS[(pid, resumable_run["id"])]["runner"]
    assert queued.path == raw
    assert queued.qa_per_chunk == 2
    assert queued.difficulty == "hard"
    assert queued.style == "direct"

    failed_row = db.get_data_prep_run(orphan_run["id"])
    assert failed_row["status"] == "error"
    assert "no longer available" in (failed_row["error"] or "")

    assert resumable_run["id"] in calls


def test_data_prep_ui_exposes_per_file_queue_and_assembler() -> None:
    template = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "finetune_studio"
        / "webui"
        / "templates"
        / "data_prep.html"
    ).read_text(encoding="utf-8")
    assert "Generate pairs for selected" in template
    assert "Build dataset from selected" in template
    assert "/data-prep/start-bulk" in template
    assert "/datasets/subset" in template
    assert "dpEnsureHelperLoaded" in template
    assert "full GPU offload" in template
    assert "dp-source-check" in template
    assert "Approved coverage" in template


def test_sources_endpoint_exposes_per_file_readiness(
    pipeline_env: tuple[TestClient, Path],
) -> None:
    client, _projects = pipeline_env
    pid = _project(client)
    source = _source(chunks=1)
    qa_fs.write_qa_source(pid, source)
    qa_fs.write_qa_pair(
        pid,
        {
            "id": "qa-ready",
            "source_id": source["id"],
            "chunk_idx": 1,
            "question": "What is grounded?",
            "answer": "This answer.",
            "status": "approved",
        },
    )

    response = client.get(f"/api/projects/{pid}/data-prep/sources")
    assert response.status_code == 200, response.text
    row = response.json()["sources"][0]
    assert row["stage"] == "training_ready"
    assert row["training_ready"] is True
    assert row["coverage_pct"] == 100.0
    assert row["pairs_approved"] == 1
