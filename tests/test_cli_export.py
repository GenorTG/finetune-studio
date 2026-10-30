"""CLI export commands share the WebUI's validation and durable export records."""
from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from finetune_studio import db
from finetune_studio.cli import _registry
from finetune_studio.webui import export_service


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
def project(temp_db: None) -> dict[str, Any]:
    return db.create_project("CLI export project")


def create_run(project_id: str, name: str = "finished") -> dict[str, Any]:
    return db.create_run(project_id=project_id, name=name, base_model="Qwen/Qwen3-0.6B", data_path="/training.jsonl")


def test_start_uses_shared_export_validator_and_supervised_queue(
    project: dict[str, Any], tmp_path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "run"
    run = db.create_run(project_id=str(project["id"]), name="finished", base_model="Qwen/Qwen3-0.6B",
                        data_path="/training.jsonl")
    db.update_run(run["id"], status="done", output_path=str(output))
    validation_calls: list[tuple[dict, dict]] = []
    queued: list[dict[str, Any]] = []

    def validate(row: dict, **kwargs: Any) -> None:
        validation_calls.append((row, kwargs))

    def start_job(**kwargs: Any) -> dict[str, Any]:
        queued.append(kwargs)
        return {"id": "export-1"}

    monkeypatch.setattr(export_service, "validate_export_request", validate)
    monkeypatch.setattr(export_service.export_jobs, "start_job", start_job)
    code, stdout, stderr = run_cli(
        monkeypatch, capsys, "export", "start", run["id"], "--project", "CLI export project",
        "--quants", "q4_k_m", "q5_k_m", "--imatrix", "--base-model", "/models/base", "--force", "--json",
    )

    assert code == 0 and not stderr
    payload = json.loads(stdout)
    assert payload["status"] == "queued" and payload["export_id"] == "export-1"
    assert payload["project"] == {"id": project["id"], "name": project["name"]}
    assert validation_calls[0][1] == {"fmt": "gguf", "quants": ["q4_k_m", "q5_k_m"],
                                     "force": True, "imatrix": True}
    assert queued[0]["project_id"] == project["id"]
    assert queued[0]["run_id"] == run["id"]
    assert queued[0]["work"] == ("multi_export_work", {
        "run": db.get_run(run["id"]), "fmt": "gguf", "quants": ["q4_k_m", "q5_k_m"],
        "force": True, "base_model": "/models/base", "imatrix": True,
    })


def test_single_quant_defaults_and_unmerged_refusal_are_preserved(
    project: dict[str, Any], tmp_path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = db.create_run(project_id=str(project["id"]), name="adapter", base_model="Qwen/Qwen3-0.6B",
                        data_path="/training.jsonl")
    db.update_run(run["id"], status="done", output_path=str(tmp_path / "run"))
    monkeypatch.setattr(export_service, "validate_export_request", lambda *_a, **_kw: None)
    queued: list[dict[str, Any]] = []
    monkeypatch.setattr(export_service.export_jobs, "start_job", lambda **kwargs: queued.append(kwargs) or {"id": "single"})

    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "start", run["id"], "--project", project["id"],
                                   "--quant", "q8_0", "--json")
    assert code == 0 and not stderr
    payload = json.loads(stdout)
    assert payload["quant"] == "Q8_0"
    assert payload["output_path"].endswith("Qwen3-0.6B-Q8_0.gguf")
    assert queued[0]["work"][0] == "single_quant_work"

    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "start", run["id"], "--project", project["id"],
                                   "--quant", "q8_0", "--no-auto-merge")
    assert code == 1 and not stdout and "run has not been merged" in stderr


def test_list_status_cancel_are_project_scoped_and_use_shared_rows(
    project: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    run = create_run(str(project["id"]))
    row = db.create_export(str(project["id"]), run["id"], format="gguf", quant="Q4_K_M", quants=["Q4_K_M"])
    db.update_export(row["id"], status="running", phase="converting", phase_detail="writing GGUF")
    monkeypatch.setattr(export_service.export_jobs, "is_active", lambda _eid: True)
    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "list", "--project", project["id"], "--json")
    assert code == 0 and not stderr and json.loads(stdout)["exports"][0]["id"] == row["id"]

    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "status", row["id"], "--project", project["id"], "--json")
    assert code == 0 and not stderr
    status = json.loads(stdout)
    assert status["phase"] == "converting" and status["phase_detail"] == "writing GGUF" and status["active"]

    monkeypatch.setattr(export_service.export_jobs, "cancel_job", lambda _eid: True)
    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "cancel", row["id"], "--project", project["id"], "--json")
    assert code == 0 and not stderr
    assert json.loads(stdout) == {"ok": True, "status": "cancelling", "export_id": row["id"]}

    other = db.create_project("Other export project")
    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "status", row["id"], "--project", other["id"])
    assert code == 1 and not stdout and "not found in this project" in stderr


def test_terminal_export_cannot_be_cancelled(project: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    run = create_run(str(project["id"]))
    row = db.create_export(str(project["id"]), run["id"])
    db.update_export(row["id"], status="done")
    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "cancel", row["id"], "--project", project["id"])
    assert code == 1 and not stdout and "export already done" in stderr


def test_export_help_lists_supported_controls(monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "start", "--help")
    assert code == 0 and not stderr
    assert all(option in stdout for option in ("--format", "--quant", "--quants", "--imatrix", "--force"))


def test_quant_options_are_unambiguous_and_list_limit_matches_api(
    project: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "list", "--project", project["id"], "--limit", "0")
    assert code == 1 and not stdout and "limit must be between 1 and 1000" in stderr

    code, stdout, stderr = run_cli(monkeypatch, capsys, "export", "start", "run-1", "--project", project["id"],
                                   "--quant", "Q4_K_M", "--quants", "Q5_K_M")
    assert code == 2 and not stdout and "not allowed with argument" in stderr
