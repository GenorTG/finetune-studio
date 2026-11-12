"""CLI access to the same Mining jobs and Q&A review records used by the WebUI."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from finetune_studio import db
from finetune_studio.cli import _registry
from finetune_studio.data.fs import qa as qa_fs


def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _project() -> dict:
    return db.create_project("CLI review project")


def _pair(pid: str, qa_id: str = "pair-1") -> dict:
    pair = {"id": qa_id, "source_id": "source-1", "chunk_idx": 1, "question": "Where is the pump?",
            "answer": "In the north room.", "status": "pending", "created_at": 1.0}
    qa_fs.write_qa_pair(pid, pair)
    return pair


def test_mining_start_resolves_source_name_and_queues_persistent_job(monkeypatch, capsys):
    from finetune_studio import data
    from finetune_studio.cli.commands import mining as mining_cli

    project = _project()
    monkeypatch.setattr("finetune_studio.knowledge.sources.list_sources",
                        lambda _pid: [SimpleNamespace(id="source-1", filename="maintenance.txt")])
    captured: dict[str, object] = {}

    class Scheduler:
        def submit(self, pid, source_ids, kind):
            captured.update(project_id=pid, source_ids=source_ids, kind=kind)
            return [{"id": "job-1", "kind": kind, "filename": "maintenance.txt", "status": "queued"}]

    monkeypatch.setattr(data.mining, "get_scheduler", lambda: Scheduler())
    code, out, err = run_cli(monkeypatch, capsys, "mining", "start", "--project", project["id"],
                             "maintenance.txt", "--kind", "fill", "--json")

    assert code == 0 and not err
    assert captured == {"project_id": project["id"], "source_ids": ["source-1"], "kind": "fill"}
    assert json.loads(out)["jobs"][0]["id"] == "job-1"
    assert mining_cli._resolve_sources(project["id"], ["source-1"]) == ["source-1"]


def test_mining_start_rejects_ambiguous_file_name_before_queueing(monkeypatch, capsys):
    from finetune_studio import data

    project = _project()
    monkeypatch.setattr("finetune_studio.knowledge.sources.list_sources", lambda _pid: [
        SimpleNamespace(id="source-a", filename="manual.txt"),
        SimpleNamespace(id="source-b", filename="Manual.txt"),
    ])
    calls: list[str] = []
    monkeypatch.setattr(data.mining, "get_scheduler", lambda: calls.append("submit"))

    code, out, err = run_cli(monkeypatch, capsys, "mining", "start", "--project", project["id"], "MANUAL.TXT")

    assert code == 1 and not out and "ambiguous" in err
    assert "source-a" in err and "source-b" in err and calls == []


def test_mining_list_and_status_are_project_scoped(monkeypatch, capsys, client):
    capsys.readouterr()  # discard the app lifespan's model-scan status before checking CLI JSON.
    project = _project()
    other = _project()
    job = db.mining.create_job(project["id"], "source-1", "maintenance.txt", "mine")
    monkeypatch.setattr("finetune_studio.data.mining.scheduler.helper_info", lambda: {"provider_id": "api-helper"})

    code, out, err = run_cli(monkeypatch, capsys, "mining", "list", "--project", project["id"], "--json")
    assert code == 0 and not err
    assert json.loads(out)["jobs"][0]["id"] == job["id"]
    api_jobs = client.get(f"/api/projects/{project['id']}/mining/jobs").json()["jobs"]
    assert api_jobs[0]["id"] == job["id"]

    code, out, err = run_cli(monkeypatch, capsys, "mining", "status", job["id"], "--project", other["id"])
    assert code == 1 and not out and "not found in this project" in err


def test_mining_cancel_uses_shared_scheduler(monkeypatch, capsys):
    from finetune_studio import data

    project = _project()
    job = db.mining.create_job(project["id"], "source-1", "maintenance.txt", "mine")
    cancelled: list[str] = []

    class Scheduler:
        def cancel(self, pid, job_id):
            assert pid == project["id"]
            cancelled.append(job_id)
            return {**job, "status": "running", "cancel_requested_at": 2.0, "message": "Stop requested."}

    monkeypatch.setattr(data.mining, "get_scheduler", lambda: Scheduler())
    code, out, err = run_cli(monkeypatch, capsys, "mining", "cancel", job["id"], "--project", project["id"], "--json")

    assert code == 0 and not err
    assert cancelled == [job["id"]]
    assert json.loads(out)["job"]["cancel_requested_at"] == 2.0


def test_qa_list_uses_slim_review_queue_and_resolves_source_name(monkeypatch, capsys):
    from finetune_studio.data.fs import qa_review

    project = _project()
    pair = _pair(project["id"])
    monkeypatch.setattr(qa_review.sources, "names", lambda _pid: {"source-1": "maintenance.txt"})
    monkeypatch.setattr("finetune_studio.knowledge.sources.list_sources",
                        lambda _pid: [SimpleNamespace(id="source-1", filename="maintenance.txt")])

    code, out, err = run_cli(monkeypatch, capsys, "qa", "list", "--project", project["id"],
                             "--source", "maintenance.txt", "--status", "pending", "--json")

    assert code == 0 and not err
    result = json.loads(out)
    assert result["items"][0]["id"] == pair["id"]
    assert result["items"][0]["source_filename"] == "maintenance.txt"
    assert result["counts"] == {"pending": 1, "approved": 0, "rejected": 0}


def test_qa_approve_updates_shared_pair_status_and_timestamp(monkeypatch, capsys, client):
    capsys.readouterr()  # discard the app lifespan's model-scan status before checking CLI JSON.
    project = _project()
    pair = _pair(project["id"])

    code, out, err = run_cli(monkeypatch, capsys, "qa", "approve", "--project", project["id"], pair["id"], "--json")

    assert code == 0 and not err
    result = json.loads(out)
    assert result["updated"] == 1 and result["counts"]["approved"] == 1
    assert result["updated_ids"] == [pair["id"]]
    saved = qa_fs.list_qa_pairs(project["id"])[0]
    assert saved["status"] == "approved" and saved["reviewed_at"] > 0
    response = client.get(f"/api/projects/{project['id']}/data-prep/qa", params={"status": "approved"})
    assert response.status_code == 200 and response.json()["items"][0]["id"] == pair["id"]


def test_qa_reject_requires_confirmation_if_it_opens_last_covered_fact(monkeypatch, capsys):
    from finetune_studio.data.mining import coverage

    project = _project()
    pair = _pair(project["id"])

    def states(_items, pairs):
        active = [p for p in pairs if p["status"] in ("approved", "pending")]
        return [coverage.ItemState("fact-1", "pending" if active else "open", [p["id"] for p in active])]

    monkeypatch.setattr(coverage, "item_states", states)
    code, out, err = run_cli(monkeypatch, capsys, "qa", "reject", "--project", project["id"], pair["id"], "--json")

    assert code == 1 and not out
    assert "fact-1" in err and "--yes" in err
    assert qa_fs.list_qa_pairs(project["id"])[0]["status"] == "pending"


def test_qa_reject_yes_reports_coverage_warning_and_rejects(monkeypatch, capsys):
    from finetune_studio.data.mining import coverage

    project = _project()
    pair = _pair(project["id"])

    def states(_items, pairs):
        active = [p for p in pairs if p["status"] in ("approved", "pending")]
        return [coverage.ItemState("fact-1", "pending" if active else "open", [p["id"] for p in active])]

    monkeypatch.setattr(coverage, "item_states", states)
    code, out, err = run_cli(monkeypatch, capsys, "qa", "reject", "--project", project["id"], pair["id"],
                             "--yes", "--json")

    assert code == 0 and not err
    result = json.loads(out)
    assert result["updated"] == 1 and result["status"] == "rejected"
    assert result["coverage_warnings"] == [{"source_id": "source-1", "item_id": "fact-1"}]


def test_mining_and_qa_help_show_safe_controls(monkeypatch, capsys):
    code, out, err = run_cli(monkeypatch, capsys, "mining", "--help")
    assert code == 0 and not err and "start" in out and "cancel" in out
    code, out, err = run_cli(monkeypatch, capsys, "qa", "reject", "--help")
    assert code == 0 and not err and "--yes" in out and "--note" in out
