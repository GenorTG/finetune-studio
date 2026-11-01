"""The Heretic export path: batch size, KL budget, leftovers, upstream check, guards.

Heretic itself never runs here: the two subprocess passes and the Optuna study
helper are faked, so these pin what *this app* does with their results.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from finetune_studio import db
from finetune_studio.training import heretic_tool, run_export

USABLE = heretic_tool.HereticStatus(installed=True, version="2.0.0.dev0", ref=heretic_tool.PINNED_REF)


def _write_model(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(b"\x00" * 16)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")


def _merged(tmp_path: Path) -> Path:
    m = tmp_path / "run" / "merged"
    _write_model(m)
    return m


def _fake_heretic(monkeypatch: pytest.MonkeyPatch, front: list[dict[str, Any]]) -> list[list[str]]:
    """Fake both Heretic passes and the study helper; returns every argv run."""
    calls: list[list[str]] = []

    def run_group(cmd: list[str], **_kw: Any) -> None:
        calls.append(cmd)
        save = Path(cmd[cmd.index("--save-directory") + 1])
        study = Path(cmd[cmd.index("--study-checkpoint-dir") + 1])
        study.mkdir(parents=True, exist_ok=True)
        (study / "study.jsonl").write_bytes(b"x" * 1000)
        if cmd[cmd.index("--export-strategy") + 1] == "merge":
            _write_model(save)
        else:
            save.mkdir(parents=True, exist_ok=True)

    def study(study_dir: str, action: str = "read", pos: int = 0) -> dict[str, Any]:
        return {"front": front} if action == "read" else {"ok": True, "pos": pos}

    monkeypatch.setattr(heretic_tool, "probe", lambda: USABLE)
    monkeypatch.setattr(heretic_tool, "_study", study)
    monkeypatch.setattr("finetune_studio.training.proc_runner.run_group_subprocess", run_group)
    return calls


FRONT_OK = [
    {"pos": 0, "trial": 7, "refusals": 0, "kl": 10.0},
    {"pos": 1, "trial": 3, "refusals": 2, "kl": 0.05},
]
FRONT_OVER = [
    {"pos": 0, "trial": 7, "refusals": 0, "kl": 10.0},
    {"pos": 1, "trial": 3, "refusals": 5, "kl": 0.4},
]


def _export(tmp_path: Path, *, force: bool = False) -> dict[str, Any]:
    return run_export._export_heretic(
        merged_path=str(_merged(tmp_path)), output_path=str(tmp_path / "run"), force=force,
    )


def test_export_runs_heretic_with_the_fixed_batch_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never the auto probe (it picked 128, the crash path)."""
    calls = _fake_heretic(monkeypatch, FRONT_OK)
    assert _export(tmp_path)["ok"] is True
    assert len(calls) == 2
    for cmd in calls:
        assert cmd[cmd.index("--batch-size") + 1] == str(heretic_tool.EXPORT_BATCH_SIZE)
    assert heretic_tool.EXPORT_BATCH_SIZE == 8


def test_second_pass_merges_the_trial_chosen_within_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_heretic(monkeypatch, FRONT_OK)
    got = _export(tmp_path)
    search, save = calls
    assert search[search.index("--export-strategy") + 1] == "adapter"
    assert save[save.index("--checkpoint-action") + 1] == "continue"
    assert save[save.index("--trial-index") + 1] == "1"
    assert got["status"] == "exported"
    assert got["budget_met"] is True
    assert got["chosen_kl"] == 0.05 and got["chosen_refusals"] == 2 and got["chosen_trial"] == 3
    assert got["kl_budget"] == heretic_tool.AbliterationRun("m", "o").kl_budget


def test_no_trial_within_budget_is_a_failure_not_an_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_heretic(monkeypatch, FRONT_OVER)
    got = _export(tmp_path)
    assert got["ok"] is False and got["status"] == "failed"
    assert got["budget_met"] is False and got["chosen_kl"] == 0.4
    assert "KL budget" in got["error"] and "Overwrite existing" in got["error"]
    # Moved aside: the next export starts clean, the files stay inspectable.
    assert not (tmp_path / "run" / "heretic").exists()
    assert (tmp_path / "run" / "heretic.over-budget" / "config.json").is_file()


def test_force_keeps_an_over_budget_result_and_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_heretic(monkeypatch, FRONT_OVER)
    got = _export(tmp_path, force=True)
    assert got["ok"] is True and got["status"] == "exported"
    assert got["budget_met"] is False
    assert "outside the KL budget" in got["message"]


def test_a_failed_run_leftover_does_not_block_the_next_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the study checkpoint of an earlier, failed run: that is not "a model already exists"."""
    leftover = tmp_path / "run" / "heretic" / heretic_tool.STUDY_DIRNAME
    leftover.mkdir(parents=True)
    (leftover / "study.jsonl").write_text("{}", encoding="utf-8")
    calls = _fake_heretic(monkeypatch, FRONT_OK)
    got = _export(tmp_path)
    assert got["status"] == "exported"
    assert len(calls) == 2


def test_a_complete_model_is_still_skipped_without_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_model(tmp_path / "run" / "heretic")
    calls = _fake_heretic(monkeypatch, FRONT_OK)
    got = _export(tmp_path)
    assert got["status"] == "skipped"
    assert calls == []


def test_output_size_excludes_the_study_checkpoint(tmp_path: Path) -> None:
    out = tmp_path / "heretic"
    _write_model(out)
    model_size = heretic_tool.inspect_output(str(out))["size_bytes"]
    study = out / heretic_tool.STUDY_DIRNAME
    study.mkdir()
    (study / "study.jsonl").write_bytes(b"x" * 10_000)
    assert heretic_tool.inspect_output(str(out))["size_bytes"] == model_size


# ── upstream check ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("commits", [
    [{"sha": "abc", "commit": {"message": ""}}],
    [{"sha": "abc", "commit": {}}],
    [],
    OSError("network down"),
])
def test_check_upstream_survives_an_empty_or_failed_commit_call(monkeypatch: pytest.MonkeyPatch, commits: Any) -> None:
    def fake(path: str, timeout: float = 20.0) -> Any:
        if path.startswith("/releases"):
            return {"tag_name": "v1.2.0"}
        if isinstance(commits, Exception):
            raise commits
        return commits

    monkeypatch.setattr(heretic_tool, "_github", fake)
    up = heretic_tool.check_upstream()
    assert up.latest_release == "v1.2.0"
    assert up.head_message == ""


def test_upstream_route_answers_when_the_commit_has_no_message(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heretic_tool, "probe", lambda: USABLE)
    monkeypatch.setattr(heretic_tool, "_github", lambda path, timeout=20.0: (
        {"tag_name": "v1"} if path.startswith("/releases") else [{"sha": "f00", "commit": {"message": ""}}]))
    r = client.get("/api/system/heretic/upstream")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["update_available"] is False
    assert body["upstream_ahead"] is True


def test_pinned_install_is_never_reported_as_outdated() -> None:
    """Install always installs PINNED_REF: comparing with upstream HEAD said "update" forever."""
    assert heretic_tool.update_available(USABLE) is False
    old = heretic_tool.HereticStatus(installed=True, version="2", ref="0" * 40)
    assert heretic_tool.update_available(old) is True
    assert heretic_tool.update_available(heretic_tool.HereticStatus(installed=False)) is True
    ahead = heretic_tool.UpstreamInfo(head_commit="f" * 40)
    assert heretic_tool.upstream_ahead(ahead) is True
    assert heretic_tool.upstream_ahead(heretic_tool.UpstreamInfo(head_commit=heretic_tool.PINNED_REF)) is False
    assert heretic_tool.upstream_ahead(heretic_tool.UpstreamInfo(error="offline")) is False


# ── no install/uninstall under a running export ──────────────────────────────

def _active_heretic_export() -> dict:
    proj = db.create_project(name="agent-test-heretic")
    run = db.create_run(proj["id"], "r1")
    row = db.create_export(project_id=proj["id"], run_id=run["id"], format="heretic", quant="safetensors")
    db.mark_export_running(row["id"])
    return row


def test_install_and_uninstall_refuse_while_a_heretic_export_runs(client, monkeypatch: pytest.MonkeyPatch) -> None:
    touched: list[str] = []
    monkeypatch.setattr(heretic_tool, "install", lambda ref, **kw: touched.append("install"))
    monkeypatch.setattr(heretic_tool, "uninstall", lambda: touched.append("uninstall"))
    row = _active_heretic_export()
    for r in (client.post("/api/system/heretic/install", json={}), client.delete("/api/system/heretic")):
        assert r.status_code == 409, r.text
        assert row["id"] in r.json()["detail"]
    assert touched == []


def test_cli_refuses_to_reinstall_during_a_heretic_export(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from finetune_studio.cli.commands.heretic import cmd_heretic

    monkeypatch.setattr(heretic_tool, "install", lambda ref, **kw: pytest.fail("installed during an export"))
    _active_heretic_export()
    with pytest.raises(SystemExit) as e:
        cmd_heretic(SimpleNamespace(heretic_action="update", ref="", json=False))
    assert e.value.code == 1
    assert "Heretic export" in capsys.readouterr().err


def test_cli_check_reports_the_pinned_build_as_current(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from finetune_studio.cli.commands.heretic import cmd_heretic

    monkeypatch.setattr(heretic_tool, "probe", lambda: USABLE)
    monkeypatch.setattr(heretic_tool, "check_upstream", lambda: heretic_tool.UpstreamInfo(head_commit="f" * 40))
    cmd_heretic(SimpleNamespace(heretic_action="check", json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["update_available"] is False and out["upstream_ahead"] is True
