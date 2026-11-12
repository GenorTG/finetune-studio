"""Use Cases CLI shares the Training page's validation, records, and evaluation jobs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from finetune_studio import db
from finetune_studio.cli import _registry
from finetune_studio.usecases import evaluate, store, teacher


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
def project() -> dict[str, Any]:
    return db.create_project("CLI use-case project", base_model="/models/base")


def _classification_rows(count: int = 90) -> list[dict[str, str]]:
    labels = ("billing", "shipping", "account")
    return [{"input": f"ticket {i} about {labels[i % len(labels)]}", "label": labels[i % len(labels)]}
            for i in range(count)]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_templates_lists_all_shapes_and_data_requirements(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "templates", "--json")
    assert code == 0 and not stderr
    templates = json.loads(stdout)["templates"]
    assert {row["id"] for row in templates} == {"extraction", "classification", "style", "distillation"}
    classification = next(row for row in templates if row["id"] == "classification")
    assert classification["row_shape"] == {"input": "the text to classify", "label": "the correct label"}
    assert classification["minimum_rows"] == 70


def test_build_uses_shared_rows_and_store_and_is_visible_to_training_ui(
    project: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], client,
) -> None:
    source = tmp_path / "examples.jsonl"
    _write_rows(source, _classification_rows())
    capsys.readouterr()  # TestClient lifespan may print model-discovery diagnostics.

    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "build", "--project", project["id"],
                                   "--template", "classification", "--file", str(source), "--name", "Ticket routing",
                                   "--labels", "billing,shipping,account", "--json")
    assert code == 0 and not stderr
    payload = json.loads(stdout)
    manifest = payload["usecase"]
    assert payload["project"]["id"] == project["id"]
    assert manifest["template"] == "classification" and manifest["train_rows"] == 70
    assert manifest["heldout"] == 20 and manifest["labels"] == ["billing", "shipping", "account"]
    assert Path(manifest["dataset"]["path"]).is_file() and Path(manifest["suite"]["path"]).is_file()
    assert payload["train_url"] == f"/projects/{project['id']}/training?usecase={manifest['id']}"

    listed = client.get(f"/api/projects/{project['id']}/usecases")
    assert listed.status_code == 200 and [row["id"] for row in listed.json()] == [manifest["id"]]
    training = client.get(payload["train_url"])
    assert training.status_code == 200 and 'id="uc-card"' in training.text
    assert client.get(f"/api/projects/{project['id']}/datasets/{manifest['dataset']['id']}").status_code == 200


def test_build_reports_parse_validation_and_small_dataset_errors(
    project: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"input": "oops"\n', encoding="utf-8")
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "build", "--project", project["id"],
                                   "--template", "style", "--file", str(broken))
    assert code == 1 and not stdout and "line 1 is not valid JSON" in stderr

    small = tmp_path / "small.json"
    small.write_text(json.dumps([{"prompt": f"q{i}", "answer": f"a{i}"} for i in range(10)]), encoding="utf-8")
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "build", "--project", project["id"],
                                   "--template", "style", "--file", str(small))
    assert code == 1 and not stdout and "needs at least" in stderr
    assert not store.list_all(project["id"])


def test_distillation_teacher_uses_api_provider_and_records_answer_report(
    project: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "prompts.jsonl"
    _write_rows(source, [{"prompt": f"summarize {i}"} for i in range(130)])

    def fake_chat(_messages: list[dict[str, str]]) -> str:
        return "A concise summary."

    monkeypatch.setattr(teacher, "open_teacher", lambda _provider: (fake_chat, "api-teacher"))
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "build", "--project", project["id"],
                                   "--template", "distillation", "--file", str(source),
                                   "--teacher-provider", "api-1", "--json")
    assert code == 0 and not stderr
    manifest = json.loads(stdout)["usecase"]
    assert manifest["teacher"] == "api-teacher"
    assert manifest["teacher_report"] == {"asked": 130, "answered": 130, "failed": 0,
                                          "errors": [], "model": "api-teacher"}


def test_list_show_and_delete_are_project_scoped_and_confirmed(
    project: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], client,
) -> None:
    source = tmp_path / "examples.json"
    _write_rows(source, _classification_rows())
    capsys.readouterr()  # discard TestClient startup model scan.
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "build", "--project", project["id"],
                                   "--template", "classification", "--file", str(source), "--json")
    assert code == 0 and not stderr
    manifest = json.loads(stdout)["usecase"]
    uid = manifest["id"]

    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "list", "--project", project["id"], "--json")
    assert code == 0 and not stderr and json.loads(stdout)["usecases"][0]["id"] == uid
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "show", uid, "--project", project["id"], "--json")
    assert code == 0 and not stderr and json.loads(stdout)["id"] == uid

    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "delete", uid, "--project", project["id"])
    assert code == 1 and not stdout and "requires --yes" in stderr
    assert client.get(f"/api/projects/{project['id']}/usecases/{uid}").status_code == 200

    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "delete", uid, "--project", project["id"],
                                   "--yes", "--json")
    result = json.loads(stdout)
    assert code == 0 and not stderr and result["kept_dataset"] == manifest["dataset"]["id"]
    assert Path(manifest["dataset"]["path"]).is_file()
    assert not Path(manifest["suite"]["path"]).exists()
    assert client.get(f"/api/projects/{project['id']}/usecases/{uid}").status_code == 404


def test_evaluate_report_and_style_ab_reuse_the_shared_job_lifecycle(
    project: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "style.jsonl"
    _write_rows(source, [{"prompt": f"question {i}", "answer": f"answer {i}"} for i in range(80)])
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "build", "--project", project["id"],
                                   "--template", "style", "--file", str(source), "--json")
    assert code == 0 and not stderr
    uid = json.loads(stdout)["usecase"]["id"]
    record = {"group_id": "1234abcd", "run_id": "run-1", "base_bid": "base-1", "tuned_bid": "tuned-1"}
    started: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(evaluate, "start", lambda pid, usecase, run, *, judge_provider_id="":
                        started.append((pid, usecase, run, judge_provider_id)) or record)
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "evaluate", uid, "run-1", "--project",
                                   project["id"], "--judge-provider", "judge-1", "--json")
    assert code == 0 and not stderr and json.loads(stdout)["group_id"] == record["group_id"]
    assert started == [(project["id"], uid, "run-1", "judge-1")]

    report = {"usecase": {"name": "Style", "template": "style"}, "status": {"run": "done", "judge": ""},
              "base": {"status": "done"}, "tuned": {"status": "done"}, "paired": None,
              "ab": None, "notes": []}
    monkeypatch.setattr(evaluate, "report", lambda *_args: report)
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "report", uid, record["group_id"],
                                   "--project", project["id"], "--json")
    assert code == 0 and not stderr and json.loads(stdout) == report

    ab_calls: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(evaluate, "start_ab", lambda pid, usecase, group, provider="":
                        ab_calls.append((pid, usecase, group, provider)) or {"status": "running", "total": 20})
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "ab", uid, record["group_id"], "--project",
                                   project["id"], "--judge-provider", "judge-2", "--json")
    assert code == 0 and not stderr and json.loads(stdout)["status"] == "running"
    assert ab_calls == [(project["id"], uid, record["group_id"], "judge-2")]


def test_help_documents_controls_and_project_ownership(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "--help")
    assert code == 0 and not stderr
    assert all(action in stdout for action in ("templates", "build", "list", "show", "delete", "evaluate", "report", "ab"))
    code, stdout, stderr = run_cli(monkeypatch, capsys, "usecases", "build", "--help")
    assert code == 0 and not stderr
    assert all(option in stdout for option in ("--project", "--template", "--file", "--schema", "--teacher-provider"))
