"""While Heretic searches, the export job names the trial it is on, read from the study journal it writes.

A search can run over an hour; the job showed one unchanged "searching abliteration parameters" line for all of it.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune_studio.training import export_job, heretic_tool, run_export


def _journal(study: Path, states: list[int], *, torn_last_line: bool = False) -> None:
    study.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"op_code": 0, "study_name": "heretic"}, separators=(",", ":"))]
    for i, st in enumerate(states):
        lines.append(json.dumps({"op_code": 4, "trial_id": i}, separators=(",", ":")))
        lines.append(json.dumps({"op_code": 6, "trial_id": i, "state": st}, separators=(",", ":")))
    text = "\n".join(lines) + "\n"
    if torn_last_line:
        text += '{"op_code":6,"trial_id":99,"st'
    (study / "model.jsonl").write_text(text, encoding="utf-8")


def test_counts_finished_trials_and_tolerates_a_half_written_line(tmp_path: Path) -> None:
    assert heretic_tool.finished_trials(str(tmp_path / "none")) == 0
    _journal(tmp_path, [1, 1, 2, 3, 0], torn_last_line=True)   # complete, complete, pruned, failed, running
    assert heretic_tool.finished_trials(str(tmp_path)) == 4


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(tick=lambda: None)


def test_tick_reports_each_new_count_once_then_the_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_export, "HERETIC_PROGRESS_EVERY_S", 0.0)
    shown: list[str] = []
    monkeypatch.setattr(export_job, "report", lambda _phase, detail="": shown.append(detail))
    tick = run_export._heretic_progress(_ctx(), SimpleNamespace(checkpoint_dir=str(tmp_path), n_trials=3))
    tick()                       # no journal yet: the "searching" line stays
    _journal(tmp_path, [1])
    tick()
    tick()                       # unchanged: no repeat
    _journal(tmp_path, [1, 1, 1])
    tick()
    assert shown == ["Heretic: trial 1 of 3 done", "Heretic: all 3 trials done, saving the best one"]


def test_reading_is_throttled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_export, "HERETIC_PROGRESS_EVERY_S", 3600.0)
    reads: list[str] = []
    monkeypatch.setattr(heretic_tool, "finished_trials", lambda d: reads.append(d) or 0)
    tick = run_export._heretic_progress(_ctx(), SimpleNamespace(checkpoint_dir=str(tmp_path), n_trials=3))
    tick()
    tick()
    assert len(reads) == 1


def test_a_cancel_during_the_search_stops_the_tick(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_export, "HERETIC_PROGRESS_EVERY_S", 0.0)

    def cancelled(_phase: str, _detail: str = "") -> None:
        raise export_job.ExportCancelled("export cancelled")

    monkeypatch.setattr(export_job, "report", cancelled)
    _journal(tmp_path, [1])
    tick = run_export._heretic_progress(_ctx(), SimpleNamespace(checkpoint_dir=str(tmp_path), n_trials=3))
    with pytest.raises(export_job.ExportCancelled):
        tick()
