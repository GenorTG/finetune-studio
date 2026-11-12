"""CLI project-version snapshots share the WebUI's manifest and lineage records."""
from __future__ import annotations

import json
import sys

import pytest

from finetune_studio import db
from finetune_studio.cli import _registry


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


@pytest.fixture()
def project(temp_db: None) -> dict:
    return db.create_project("CLI version project")


@pytest.fixture()
def client(temp_db: None):
    from fastapi.testclient import TestClient

    from finetune_studio.webui.app import app

    return TestClient(app)


def test_pin_matches_wizard_manifest_and_defaults_to_latest_parent(
    project: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], client,
) -> None:
    pid = str(project["id"])
    root = db.create_version(pid, label="root")
    monkeypatch.setattr(db, "list_datasets", lambda _pid: [
        {"id": "dataset-1", "name": "reviewed", "data_path": "/data/reviewed.jsonl", "qa_count": 12},
    ])
    monkeypatch.setattr(db.brain_builds, "latest_for_project", lambda _pid: {
        "id": "build-1", "index_id": "index-1", "doc_count": 2, "chunk_count": 9,
    })
    monkeypatch.setattr("finetune_studio.db.runs.list_runs", lambda _pid: [
        {"id": "run-1", "status": "done", "final_loss": 0.25, "output_path": "output/run-1"},
        {"id": "run-live", "status": "running", "final_loss": None, "output_path": ""},
    ])
    capsys.readouterr()  # discard model-scan output emitted by the TestClient lifespan.

    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "pin", "--project", pid,
                                   "--label", "reviewed", "--notes", "before next train", "--json")

    assert code == 0 and not stderr
    pinned = json.loads(stdout)["version"]
    assert pinned["parent_version_id"] == root["id"]
    assert pinned["label"] == "reviewed" and pinned["notes"] == "before next train"
    assert pinned["manifest"] == {
        "datasets": [{"id": "dataset-1", "name": "reviewed", "data_path": "/data/reviewed.jsonl", "qa_count": 12}],
        "brain_builds": [{"build_id": "build-1", "index_id": "index-1", "doc_count": 2, "chunk_count": 9}],
        "training_runs": [{"run_id": "run-1", "status": "done", "final_loss": 0.25,
                            "output_path": "output/run-1"}],
    }
    response = client.get(f"/api/projects/{pid}/versions/{pinned['id']}")
    assert response.status_code == 200
    assert response.json()["manifest"] == pinned["manifest"]


def test_pin_can_start_root_or_branch_from_an_owned_version(
    project: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    pid = str(project["id"])
    parent = db.create_version(pid, label="parent")

    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "pin", "--project", pid,
                                   "--parent", parent["id"], "--json")
    assert code == 0 and not stderr
    assert json.loads(stdout)["version"]["parent_version_id"] == parent["id"]

    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "pin", "--project", pid,
                                   "--root", "--json")
    assert code == 0 and not stderr
    assert json.loads(stdout)["version"]["parent_version_id"] is None

    other = db.create_project("Different version project")
    foreign = db.create_version(str(other["id"]), label="foreign")
    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "pin", "--project", pid,
                                   "--parent", foreign["id"])
    assert code == 1 and not stdout and "not found in this project" in stderr


def test_list_show_and_lineage_are_project_scoped(
    project: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    pid = str(project["id"])
    root = db.create_version(pid, label="root", manifest={"datasets": [{"id": "d1"}]})
    child = db.create_version(pid, label="child", parent_version_id=root["id"])
    other = db.create_project("Another project")
    foreign = db.create_version(str(other["id"]), label="foreign")

    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "list", "--project", pid, "--json")
    assert code == 0 and not stderr
    listing = json.loads(stdout)["versions"]
    assert [row["id"] for row in listing] == [child["id"], root["id"]]
    assert listing[1]["manifest"]["datasets"] == [{"id": "d1"}]

    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "show", child["id"],
                                   "--project", pid, "--json")
    assert code == 0 and not stderr and json.loads(stdout)["version"]["label"] == "child"

    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "lineage", child["id"], "--project", pid,
                                   "--json")
    assert code == 0 and not stderr
    assert [row["id"] for row in json.loads(stdout)["lineage"]] == [root["id"], child["id"]]

    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "lineage", foreign["id"], "--project", pid)
    assert code == 1 and not stdout and "not found in this project" in stderr


def test_delete_requires_yes_noninteractive_and_removes_only_record(
    project: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path,
    client,
) -> None:
    pid = str(project["id"])
    artifact = tmp_path / "retained-dataset.jsonl"
    artifact.write_text('{"messages": []}\n', encoding="utf-8")
    version = db.create_version(pid, label="retained", manifest={
        "datasets": [{"id": "d1", "name": "artifact", "data_path": str(artifact)}],
    })
    capsys.readouterr()  # discard TestClient startup output before asserting JSON/stdout.

    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "delete", version["id"], "--project", pid)
    assert code == 1 and not stdout and "requires --yes" in stderr
    assert db.get_version(version["id"]) is not None and artifact.exists()

    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "delete", version["id"],
                                   "--project", pid, "--yes", "--json")
    assert code == 0 and not stderr and json.loads(stdout)["ok"] is True
    assert db.get_version(version["id"]) is None and artifact.exists()
    assert client.get(f"/api/projects/{pid}/versions/{version['id']}").status_code == 404


def test_version_commands_expose_expected_help(monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "--help")
    assert code == 0 and not stderr
    assert all(command in stdout for command in ("list", "pin", "show", "lineage", "delete"))
    code, stdout, stderr = run_cli(monkeypatch, capsys, "versions", "delete", "--help")
    assert code == 0 and not stderr and "--yes" in stdout
