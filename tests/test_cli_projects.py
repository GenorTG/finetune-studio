"""Project lifecycle commands for the local `fts` CLI."""
from __future__ import annotations

import io
import json
import sys

import pytest

from finetune_studio import db
from finetune_studio.cli import _registry
from finetune_studio.cli.commands import projects as project_commands


def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
            *argv: str) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_projects_create_list_show_and_webui_share_project_records(monkeypatch, capsys, client):
    capsys.readouterr()  # discard the app lifespan's model-scan status before checking CLI JSON.
    code, out, err = run_cli(monkeypatch, capsys, "projects", "create", "--name", " CLI demo ",
                             "--description", "A test project", "--base-model", "models/base",
                             "--system-prompt", "Be clear.", "--json")
    assert code == 0, err
    project = json.loads(out)
    assert project["name"] == "CLI demo"
    assert project["description"] == "A test project"
    assert project["base_model"] == "models/base"
    assert project["system_prompt"] == "Be clear."
    assert project["id"]

    code, out, err = run_cli(monkeypatch, capsys, "projects", "show", project["id"], "--json")
    assert code == 0, err
    assert json.loads(out) == project

    code, out, err = run_cli(monkeypatch, capsys, "projects", "show", "cli DEMO")
    assert code == 0, err
    assert project["id"] in out and "CLI demo" in out

    code, out, err = run_cli(monkeypatch, capsys, "projects", "list", "--json")
    assert code == 0, err
    listed = json.loads(out)
    assert [row["id"] for row in listed] == [project["id"]]
    response = client.get("/api/projects")
    assert response.status_code == 200
    assert response.json()[0]["id"] == project["id"]


def test_projects_create_trims_name_and_rejects_case_insensitive_duplicate(monkeypatch, capsys):
    assert run_cli(monkeypatch, capsys, "projects", "create", "--name", "Notes")[0] == 0
    code, out, err = run_cli(monkeypatch, capsys, "projects", "create", "--name", " notes ")
    assert code == 1
    assert not out and "already exists" in err and "Traceback" not in err


def test_projects_create_can_explicitly_allow_duplicate_names(monkeypatch, capsys):
    assert run_cli(monkeypatch, capsys, "projects", "create", "--name", "Notes")[0] == 0
    code, out, err = run_cli(monkeypatch, capsys, "projects", "create", "--name", "notes",
                             "--allow-duplicate", "--json")
    assert code == 0, err
    assert json.loads(out)["name"] == "notes"
    assert len(db.list_projects()) == 2


def test_projects_list_empty_state_and_text_output(monkeypatch, capsys):
    code, out, err = run_cli(monkeypatch, capsys, "projects", "list")
    assert code == 0 and not err and "No projects found." in out

    db.create_project("Readable project", base_model="models/base")
    code, out, err = run_cli(monkeypatch, capsys, "projects", "list")
    assert code == 0 and not err
    assert "Readable project" in out and "models/base" in out


def test_projects_show_reports_missing_and_ambiguous_names(monkeypatch, capsys):
    code, out, err = run_cli(monkeypatch, capsys, "projects", "show", "missing")
    assert code == 1 and not out and "project not found" in err

    first = db.create_project("Notes")
    second = db.create_project("notes")
    code, out, err = run_cli(monkeypatch, capsys, "projects", "show", "NOTES")
    assert code == 1 and not out and "ambiguous" in err
    assert first["id"] in err and second["id"] in err


def test_projects_create_rejects_blank_name(monkeypatch, capsys):
    code, out, err = run_cli(monkeypatch, capsys, "projects", "create", "--name", "   ")
    assert code == 1 and not out and "--name must not be empty" in err


def test_projects_help_describes_supported_commands(monkeypatch, capsys):
    code, out, err = run_cli(monkeypatch, capsys, "projects", "--help")
    assert code == 0 and not err
    assert "list" in out and "show" in out and "create" in out
    code, create_help, err = run_cli(monkeypatch, capsys, "projects", "create", "--help")
    assert code == 0 and not err
    assert "--name" in create_help and "--allow-duplicate" in create_help


def test_projects_delete_requires_explicit_confirmation_without_tty(monkeypatch, capsys):
    project = db.create_project("Keep me")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    code, out, err = run_cli(monkeypatch, capsys, "projects", "delete", project["id"])

    assert code == 1 and not out
    assert "requires --yes" in err
    assert db.get_project(project["id"]) is not None


def test_projects_delete_prompt_can_cancel_without_mutation(monkeypatch, capsys):
    project = db.create_project("Keep me")

    class TerminalInput(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", TerminalInput("no\n"))
    code, out, err = run_cli(monkeypatch, capsys, "projects", "delete", project["name"])

    assert code == 0 and not err
    assert "Cancelled." in out
    assert db.get_project(project["id"]) is not None


def test_projects_delete_json_cancel_is_machine_readable(monkeypatch, capsys):
    project = db.create_project("Keep me")

    class TerminalInput(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", TerminalInput("n\n"))
    code, out, err = run_cli(monkeypatch, capsys, "projects", "delete", project["id"], "--json")

    assert code == 0 and "Delete project" in err
    assert json.loads(out) == {"ok": False, "cancelled": True}
    assert db.get_project(project["id"]) is not None


def test_projects_delete_yes_removes_all_project_data_and_refreshes_registry(
    monkeypatch, capsys, mock_settings, tmp_path,
):
    from pathlib import Path

    from finetune_studio.data.fs import paths as fts_paths

    monkeypatch.chdir(tmp_path)
    fts_root = tmp_path / "fts-home"
    monkeypatch.setattr(fts_paths, "_ROOT", fts_root)
    monkeypatch.setattr(fts_paths, "_PROJECTS", fts_root / "projects")
    project = db.create_project("Disposable project")
    project_dirs = (
        fts_root / "projects" / project["id"],
        Path(mock_settings.db_path).parent / "projects" / project["id"],
        tmp_path / "output" / "projects" / project["id"],
        fts_paths.brain_index_dir(project["id"]),
    )
    for directory in project_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "owned.txt").write_text("disposable")
    invalidations: list[str] = []
    monkeypatch.setattr(project_commands, "_invalidate_model_registry", lambda: invalidations.append("invalidate"))

    code, out, err = run_cli(monkeypatch, capsys, "projects", "delete", project["id"], "--yes", "--json")

    assert code == 0 and not err
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["project"]["id"] == project["id"]
    assert db.get_project(project["id"]) is None
    assert all(not directory.exists() for directory in project_dirs)
    assert invalidations == ["invalidate"]


def test_projects_delete_help_documents_confirmation(monkeypatch, capsys):
    code, out, err = run_cli(monkeypatch, capsys, "projects", "delete", "--help")
    assert code == 0 and not err
    assert "--yes" in out and "--json" in out
