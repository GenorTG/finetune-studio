"""Project file-library commands for the local fts CLI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.cli import _registry
from finetune_studio.cli.commands import files as files_command
from finetune_studio.data.fs import file_library as fl


@pytest.fixture
def fts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fts"
    root.mkdir()
    projects = root / "projects"
    projects.mkdir()
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", projects)
    return root


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


def _project_and_file(client, name: str = "CLI Files Test") -> tuple[str, str]:
    project_response = client.post("/api/projects", json={"name": name})
    assert project_response.status_code in (200, 201), project_response.text
    pid = project_response.json()["id"]
    upload_response = client.post(
        f"/api/projects/{pid}/files/upload",
        files=[("files", ("notes.txt", b"Useful CLI parser content, unique to this test.", "text/plain"))],
    )
    assert upload_response.status_code == 200, upload_response.text
    return pid, upload_response.json()["report"][0]["file_id"]


def test_files_list_and_show_share_file_records_with_webui(monkeypatch, capsys, client, fts_root):
    pid, fid = _project_and_file(client)
    capsys.readouterr()

    code, out, err = run_cli(monkeypatch, capsys, "files", "list", "--project", "cli files test", "--json")
    assert code == 0, err
    data = json.loads(out)
    assert data["project"]["id"] == pid
    assert len(data["files"]) == 1
    assert data["files"][0]["id"] == fid
    assert data["files"][0]["original_name"] == "notes.txt"
    assert data["files"][0]["parse"]["status"] == "ready"

    api_file = client.get(f"/api/projects/{pid}/files/{fid}").json()
    assert data["files"][0]["size_bytes"] == api_file["size_bytes"]

    code, out, err = run_cli(monkeypatch, capsys, "files", "show", "--project", pid, "notes.txt", "--json")
    assert code == 0, err
    shown = json.loads(out)
    assert shown["id"] == fid
    assert shown["parse"]["status"] == "ready"


def test_files_parse_and_explicit_reparse_match_webui_preview(monkeypatch, capsys, client, fts_root):
    pid, fid = _project_and_file(client, "CLI Parse Test")
    capsys.readouterr()

    manual = client.put(f"/api/projects/{pid}/files/{fid}/parsed", json={"text": "Temporary hand-edited parsed content."})
    assert manual.status_code == 200, manual.text

    code, out, err = run_cli(monkeypatch, capsys, "files", "parse", "--project", pid, fid, "--reparse", "--json")
    assert code == 0, err
    parsed = json.loads(out)
    assert parsed["file_id"] == fid
    assert parsed["source"]["status"] == "ready"
    assert parsed["source"]["chunk_count"] > 0

    preview = client.get(f"/api/projects/{pid}/files/{fid}/parsed")
    assert preview.status_code == 200, preview.text
    assert "Useful CLI parser content" in preview.json()["parsed_md"]
    assert "Temporary hand-edited" not in preview.json()["parsed_md"]


def test_files_upload_uses_shared_library_with_dedup_versioning_and_parsing(
    monkeypatch, capsys, client, fts_root, tmp_path
):
    project_response = client.post("/api/projects", json={"name": "CLI Upload Test"})
    pid = project_response.json()["id"]
    capsys.readouterr()
    first = tmp_path / "notes.txt"
    binary = tmp_path / "image.bin"
    first.write_text("Useful upload content that the shared parser can read.")
    binary.write_bytes(b"non-parser binary data")

    code, out, err = run_cli(monkeypatch, capsys, "files", "upload", "--project", pid,
                             str(first), str(binary), "--json")
    assert code == 0 and not err and out.strip(), (code, repr(out), err)
    assert out.lstrip().startswith("{"), repr(out)
    result = json.loads(out)
    assert result["counts"] == {"uploaded": 2, "duplicates_skipped": 0, "errors": 0}
    assert result["report"][0]["source"]["status"] == "ready"
    assert result["report"][1]["parse_status"] == "not staged"
    file_id = result["report"][0]["file_id"]
    assert client.get(f"/api/projects/{pid}/files/{file_id}/parsed").json()["parsed_md"].startswith("Useful upload")
    listed = client.get(f"/api/projects/{pid}/files").json()["files"]
    assert {item["original_name"] for item in listed} == {"notes.txt", "image.bin"}

    alias = tmp_path / "same-content.md"
    alias.write_bytes(first.read_bytes())
    code, out, err = run_cli(monkeypatch, capsys, "files", "upload", "--project", pid, str(alias), "--json")
    assert code == 0 and not err
    duplicate = json.loads(out)
    assert duplicate["counts"]["duplicates_skipped"] == 1
    assert duplicate["report"][0]["status"] == "duplicate"
    assert len(client.get(f"/api/projects/{pid}/files").json()["files"]) == 2

    first.write_text("A changed upload creates a new version of this named file.")
    code, out, err = run_cli(monkeypatch, capsys, "files", "upload", "--project", pid, str(first), "--json")
    assert code == 0 and not err
    revised = json.loads(out)
    assert revised["report"][0]["file_id"] == file_id
    assert client.get(f"/api/projects/{pid}/files/{file_id}").json()["current_version"] == 2


def test_files_upload_reports_per_file_failures_and_continues(monkeypatch, capsys, client, fts_root, tmp_path):
    project_response = client.post("/api/projects", json={"name": "CLI Partial Upload Test"})
    pid = project_response.json()["id"]
    capsys.readouterr()
    valid = tmp_path / "valid.txt"
    empty = tmp_path / "empty.txt"
    valid.write_text("This valid file should still upload when other paths fail.")
    empty.touch()
    missing = tmp_path / "missing.txt"

    code, out, err = run_cli(monkeypatch, capsys, "files", "upload", "--project", pid,
                             str(missing), str(empty), str(valid), "--json")
    assert code == 1 and not err
    result = json.loads(out)
    assert result["counts"] == {"uploaded": 1, "duplicates_skipped": 0, "errors": 2}
    assert [item["status"] for item in result["report"]] == ["error", "error", "uploaded"]
    assert len(client.get(f"/api/projects/{pid}/files").json()["files"]) == 1


def test_files_commands_report_missing_items_and_resolve_versioned_name(monkeypatch, capsys, client, fts_root):
    pid, fid = _project_and_file(client, "CLI Versioned Name Test")
    replacement = client.post(
        f"/api/projects/{pid}/files/upload",
        files=[("files", ("notes.txt", b"A replacement version of the same named file.", "text/plain"))],
    )
    assert replacement.status_code == 200, replacement.text
    assert replacement.json()["counts"]["uploaded"] == 1, replacement.json()
    assert replacement.json()["report"][0]["file_id"] == fid
    api_file = client.get(f"/api/projects/{pid}/files/{fid}").json()
    assert api_file["current_version"] == 2
    capsys.readouterr()

    code, out, err = run_cli(monkeypatch, capsys, "files", "show", "--project", pid, "notes.txt", "--json")
    assert code == 0, err
    assert json.loads(out)["id"] == fid
    assert json.loads(out)["current_version"] == 2

    db.create_project("CLI Versioned Name Test")
    code, out, err = run_cli(monkeypatch, capsys, "files", "list", "--project", "CLI Versioned Name Test")
    assert code == 1 and not out and "ambiguous" in err and pid in err

    code, out, err = run_cli(monkeypatch, capsys, "files", "parse", "--project", pid, "missing.txt")
    assert code == 1 and not out and "not found" in err

    code, out, err = run_cli(monkeypatch, capsys, "files", "list", "--project", "missing-project")
    assert code == 1 and not out and "project not found" in err


def test_bare_files_command_prints_usage_without_running_trash(monkeypatch, capsys):
    def should_not_run(_args):
        raise AssertionError("bare files command must not purge trash")

    monkeypatch.setattr(files_command, "_cmd_files_trash", should_not_run)
    code, out, err = run_cli(monkeypatch, capsys, "files")
    assert code == 0 and not err
    assert "Usage: fts files" in out


def test_files_parse_reports_unreadable_file_without_traceback(monkeypatch, capsys, client, fts_root):
    project_response = client.post("/api/projects", json={"name": "CLI Unreadable Test"})
    pid = project_response.json()["id"]
    upload = client.post(
        f"/api/projects/{pid}/files/upload",
        files=[("files", ("scan.pdf", b"%PDF-1.4 no readable text", "application/pdf"))],
    )
    assert upload.status_code == 200, upload.text
    fid = upload.json()["report"][0]["file_id"]
    capsys.readouterr()

    code, out, err = run_cli(monkeypatch, capsys, "files", "parse", "--project", pid, fid, "--json")
    assert code == 1 and not err
    result = json.loads(out)
    assert result["source"]["status"] == "error"
    assert "no text" in result["source"]["error"]


def test_files_trash_list_and_dry_run_never_purge(monkeypatch, capsys):
    project = db.create_project("CLI Trash Test")
    trashed = [{"original_name": "old.txt", "mime_type": "text/plain", "age_days": 8, "deleted_at": 1.0}]
    monkeypatch.setattr(fl, "list_trash", lambda _pid: trashed)
    purged = []
    monkeypatch.setattr(fl, "purge_trash", lambda pid, older_than_days: purged.append((pid, older_than_days)) or {"count": 1})

    code, out, err = run_cli(monkeypatch, capsys, "files", "trash", "--project-id", project["id"], "--list")
    assert code == 0 and not err and "old.txt" in out
    assert not purged

    code, out, err = run_cli(monkeypatch, capsys, "files", "trash", "--project-id", project["id"], "--dry-run")
    assert code == 0 and not err and "Would purge 1 files" in out
    assert not purged


def test_files_help_documents_safe_reparse_flag(monkeypatch, capsys):
    code, out, err = run_cli(monkeypatch, capsys, "files", "--help")
    assert code == 0 and not err
    assert "upload" in out and "list" in out and "show" in out and "parse" in out and "trash" in out

    code, out, err = run_cli(monkeypatch, capsys, "files", "parse", "--help")
    assert code == 0 and not err
    assert "--reparse" in out and "hand-edited" in out
