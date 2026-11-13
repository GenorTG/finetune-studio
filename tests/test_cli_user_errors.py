"""CLI user errors are one line, never a traceback; --out NAME.tar.gz writes that file (walkthrough 2026-11-13 m7,
m8, m31)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from finetune_studio.cli._registry import main
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)


def _run(monkeypatch, capsys, *argv: str) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["fts", *argv])
    with pytest.raises(SystemExit) as stop:
        main()
    out = capsys.readouterr()
    return int(stop.value.code or 0), out.out, out.err


@pytest.mark.parametrize("cmd", ["readiness", "lint"])
def test_unknown_project_is_one_line(client, monkeypatch, capsys, cmd: str) -> None:
    code, _, err = _run(monkeypatch, capsys, "brain", cmd, "--project", "no-such-project-zz")
    assert code == 2 and "project not found: 'no-such-project-zz'" in err and "fts projects list" in err
    assert "Traceback" not in err


def test_help_names_the_fts_command_and_the_package_first(monkeypatch, capsys) -> None:
    code, out, _ = _run(monkeypatch, capsys, "--help")
    assert code == 0 and out.startswith("usage: fts ") and "package knowledge" in out


def test_out_with_an_archive_name_writes_that_file(monkeypatch, capsys, tmp_path: Path, corpus_dir: Path) -> None:  # noqa: F811
    target = tmp_path / "public-bot.tar.gz"
    monkeypatch.setattr(sys, "argv", ["fts", "brain", "export", "--corpus", str(corpus_dir), "--out", str(target),
                                      "--no-semantic", "--no-encrypt"])
    main()
    assert target.is_file() and not target.is_dir()
    assert '"path": "' + str(target) + '"' in capsys.readouterr().out
    code, _, err = _run(monkeypatch, capsys, "brain", "export", "--corpus", str(corpus_dir), "--out",
                        str(tmp_path / "x.zip"), "--format", "tar.gz", "--no-encrypt")
    assert code == 1 and "is a .zip file but --format is tar.gz" in err
