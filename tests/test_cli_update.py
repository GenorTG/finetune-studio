"""CLI parity for Settings → Updates, using the update worker's safe test mode."""
from __future__ import annotations

import json
import sys

import pytest

from finetune_studio import db
from finetune_studio.cli import _registry


def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
            *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    try:
        _registry.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize(("action", "mode", "confirmation"), [
    ("check", "check", []),
    ("apply", "update", ["--yes"]),
    ("repair", "repair", ["--yes"]),
])
def test_update_operations_use_durable_shared_worker(
    monkeypatch, capsys, mock_settings, action: str, mode: str, confirmation: list[str],
) -> None:
    monkeypatch.setenv("FTS_SKIP_UPDATE", "1")

    code, out, err = run_cli(monkeypatch, capsys, "update", action, "--no-llama", *confirmation, "--json")

    assert code == 0 and not err
    result = json.loads(out)
    assert result["mode"] == mode and result["status"] == "done"
    assert result["triggered_by"] == "cli"
    assert result["options"] == {"no_pull": False, "no_llama": True, "no_restart": False}
    assert "FTS_SKIP_UPDATE=1" in result["log_tail"]
    assert result["log_length"] > 0
    stored = db.get_update(result["id"])
    assert stored["status"] == "done" and "step 5" in stored["log_text"]


def test_update_streams_progress_to_terminal(monkeypatch, capsys, mock_settings) -> None:
    monkeypatch.setenv("FTS_SKIP_UPDATE", "1")

    code, out, err = run_cli(monkeypatch, capsys, "update", "check")

    assert code == 0 and not err
    assert "FTS_SKIP_UPDATE=1" in out and "step 5: restart (skipped)" in out
    assert "Update " in out and "· done" in out


def test_apply_requires_confirmation_in_noninteractive_use(monkeypatch, capsys, mock_settings) -> None:
    monkeypatch.setenv("FTS_SKIP_UPDATE", "1")
    before = db.list_updates_recent()

    code, out, err = run_cli(monkeypatch, capsys, "update", "apply", "--json")

    assert code == 1 and not out
    assert "requires --yes" in err and "restart" in err
    assert db.list_updates_recent() == before


def test_history_and_status_read_the_same_update_records(monkeypatch, capsys, mock_settings) -> None:
    monkeypatch.setenv("FTS_SKIP_UPDATE", "1")
    code, out, err = run_cli(monkeypatch, capsys, "update", "check", "--json")
    assert code == 0 and not err
    created = json.loads(out)

    code, out, err = run_cli(monkeypatch, capsys, "update", "history", "--json")
    assert code == 0 and not err
    history = json.loads(out)["updates"]
    assert history[0]["id"] == created["id"] and history[0]["mode"] == "check"

    code, out, err = run_cli(monkeypatch, capsys, "update", "status", created["id"], "--full", "--json")
    assert code == 0 and not err
    status = json.loads(out)
    assert status["status"] == "done" and status["log_length"] == len(status["log_tail"])
    assert "step 1: git pull (skipped)" in status["log_tail"]


def test_unknown_update_status_is_plain_error(monkeypatch, capsys, mock_settings) -> None:
    code, out, err = run_cli(monkeypatch, capsys, "update", "status", "missing-id")

    assert code == 1 and not out
    assert "unknown update id" in err and "Traceback" not in err
