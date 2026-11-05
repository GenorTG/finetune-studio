"""A Heretic export survives the GPU: it refuses to start without room and resumes after losing the card.

Measured 2026-11-05 on an RTX 3090 (RESULTS §37): with less free VRAM than the model needs, Accelerate puts part of the
model on the CPU and the search crawls (a 20 s step passed 10 minutes). When another program takes the VRAM mid-search,
Heretic exits with a CUDA out of memory and all its trials looked lost. Its error is on stdout; the runner's tail
showed a warning from stderr instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from finetune_studio.training import heretic_tool, run_export
from finetune_studio.training.proc_runner import SubprocessCancelled, SubprocessFailed
from tests.test_heretic_export_path import USABLE, _merged, _write_model

OOM = """│   351 │   │   new_bad_output = new_bad_output * scale    │
╰──────────────────────────────────────────────────────────────────────────────╯
OutOfMemoryError: CUDA out of
memory. Tried to allocate 54.00 MiB. GPU 0 has a total capacity of 23.56 GiB.
"""
FRONT = [{"pos": 0, "trial": 3, "refusals": 2, "kl": 0.05}]


def _fake(monkeypatch: pytest.MonkeyPatch, failures: list[str]) -> list[list[str]]:
    """Heretic passes that fail with ``failures`` in turn, then succeed. Returns every argv."""
    calls: list[list[str]] = []

    def run_group(cmd: list[str], **_kw: Any) -> None:
        calls.append(cmd)
        Path(cmd[cmd.index("--study-checkpoint-dir") + 1]).mkdir(parents=True, exist_ok=True)
        if failures:
            raise SubprocessFailed("heretic", 1, "Warning: unauthenticated requests to the HF Hub", failures.pop(0))
        save = Path(cmd[cmd.index("--save-directory") + 1])
        if cmd[cmd.index("--export-strategy") + 1] == "merge":
            _write_model(save)
        else:
            save.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(heretic_tool, "probe", lambda: USABLE)
    monkeypatch.setattr(heretic_tool, "_study", lambda _d, action="read", pos=0: {"front": FRONT} if action == "read"
                        else {"ok": True, "pos": pos})
    monkeypatch.setattr(heretic_tool, "_vram_state", lambda: (20.0, ""))
    monkeypatch.setattr(heretic_tool, "VRAM_POLL_S", 0.0)
    monkeypatch.setattr("finetune_studio.training.proc_runner.run_group_subprocess", run_group)
    return calls


def _export(tmp_path: Path) -> dict[str, Any]:
    return run_export._export_heretic(merged_path=str(_merged(tmp_path)), output_path=str(tmp_path / "run"),
                                      force=False, n_trials=4)


def _actions(calls: list[list[str]]) -> list[str]:
    return [c[c.index("--checkpoint-action") + 1] for c in calls]


def test_needs_the_weights_plus_a_margin(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 1024)
    assert heretic_tool.vram_needed_gib(str(tmp_path)) == pytest.approx(1.0, abs=0.05)
    assert heretic_tool.vram_needed_gib(str(tmp_path / "missing")) == 0.0


def test_too_little_vram_refuses_before_starting_and_names_the_holders(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake(monkeypatch, [])
    monkeypatch.setattr(heretic_tool, "vram_needed_gib", lambda _p: 4.2)
    monkeypatch.setattr(heretic_tool, "_vram_state", lambda: (2.0, "comfyui 18.0 GB"))
    got = _export(tmp_path)
    assert got["ok"] is False and not calls
    assert "needs about 4.2 GB, 2.0 GB is free (comfyui 18.0 GB)" in got["error"]


def test_no_gpu_or_unknown_size_does_not_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heretic_tool, "_vram_state", lambda: (None, ""))
    assert heretic_tool.vram_shortfall(str(tmp_path)) == ""
    monkeypatch.setattr(heretic_tool, "_vram_state", lambda: (0.1, ""))
    assert heretic_tool.vram_shortfall(str(tmp_path / "missing")) == ""


@pytest.mark.parametrize("error", [OOM, "RuntimeError: CUDA error: an illegal memory\naccess was encountered"])
def test_a_gpu_failure_resumes_from_the_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str) -> None:
    calls = _fake(monkeypatch, [error])
    monkeypatch.setattr(heretic_tool, "VRAM_MARGIN_GIB", 0.0)
    got = _export(tmp_path)
    assert got["ok"] is True, got
    assert _actions(calls) == ["restart", "continue", "continue"]   # search, resumed search, merge


def test_a_cuda_fault_resumes_synchronously_and_out_of_memory_does_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    envs: list[str | None] = []
    calls = _fake(monkeypatch, [OOM, "RuntimeError: CUDA error: an illegal memory access was encountered"])
    fake_run = __import__("finetune_studio.training.proc_runner", fromlist=["x"]).run_group_subprocess

    def spy(cmd: list[str], **kw: Any) -> None:
        envs.append(kw["env"].get("CUDA_LAUNCH_BLOCKING"))
        fake_run(cmd, **kw)

    monkeypatch.setattr("finetune_studio.training.proc_runner.run_group_subprocess", spy)
    monkeypatch.setattr(heretic_tool, "VRAM_MARGIN_GIB", 0.0)
    assert _export(tmp_path)["ok"] is True
    assert envs == [None, None, "1", "1"], calls   # search, after the OOM, after the fault, merge


def test_a_normal_run_never_pays_for_synchronous_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    assert "CUDA_LAUNCH_BLOCKING" not in heretic_tool.run_env()
    assert heretic_tool.run_env(synchronous=True)["CUDA_LAUNCH_BLOCKING"] == "1"


def test_it_waits_for_the_vram_and_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake(monkeypatch, [OOM])
    monkeypatch.setattr(heretic_tool, "vram_needed_gib", lambda _p: 4.0)
    free = iter([20.0, 1.0, 1.0, 20.0, 20.0, 20.0])
    monkeypatch.setattr(heretic_tool, "_vram_state", lambda: (next(free), "comfyui 18.0 GB"))
    shown: list[str] = []
    monkeypatch.setattr(run_export.export_job, "report", lambda _p, detail="": shown.append(detail))
    assert _export(tmp_path)["ok"] is True
    assert any("paused, 4.0 GB of free VRAM needed, 1.0 GB free (comfyui 18.0 GB)" in s for s in shown), shown
    assert any("resuming from the finished trials" in s for s in shown)
    assert _actions(calls)[:2] == ["restart", "continue"]


def test_vram_that_never_comes_back_stops_and_keeps_the_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, [OOM])
    monkeypatch.setattr(heretic_tool, "vram_needed_gib", lambda _p: 4.0)
    free = iter([20.0])
    monkeypatch.setattr(heretic_tool, "_vram_state", lambda: (next(free, 1.0), ""))
    monkeypatch.setattr(heretic_tool, "VRAM_WAIT_S", 0.0)
    orig = heretic_tool.run_abliteration
    monkeypatch.setattr(heretic_tool, "run_abliteration", lambda run, **kw: orig(run, vram_wait_s=0.0, vram_poll_s=0.0, **kw))
    got = _export(tmp_path)
    assert got["ok"] is False
    assert "stayed too full" in got["error"] and "export again to continue" in got["error"]


def test_a_cancel_while_waiting_stops_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heretic_tool, "_vram_state", lambda: (1.0, ""))
    monkeypatch.setattr(heretic_tool, "vram_needed_gib", lambda _p: 4.0)
    with pytest.raises(SubprocessCancelled):
        heretic_tool._wait_for_vram("m", cancelled=lambda: True, on_tick=None, on_status=None, wait_s=60.0, poll_s=1.0)


def test_resumes_are_limited_then_it_reports_the_real_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake(monkeypatch, [OOM] * (heretic_tool.MAX_GPU_RESUMES + 1))
    got = _export(tmp_path)
    assert got["ok"] is False and len(calls) == heretic_tool.MAX_GPU_RESUMES + 1
    assert got["error"].startswith("Heretic failed: OutOfMemoryError: CUDA out of memory.")
    assert "unauthenticated" not in got["error"]


def test_a_model_problem_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake(monkeypatch, ["ValueError: this model has no layers"])
    got = _export(tmp_path)
    assert got["ok"] is False and len(calls) == 1
    assert "this model has no layers" in got["error"]


def _journal(study: Path, *, finished: bool) -> None:
    study.mkdir(parents=True)
    rows = [{"op_code": 2, "user_attr": {"finished": False}}, {"op_code": 6, "trial_id": 0, "state": 1},
            {"op_code": 2, "user_attr": {"finished": finished}}]
    (study / "m.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize(("finished", "action"), [(False, "continue"), (True, "restart")])
def test_the_next_export_continues_only_an_unfinished_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             finished: bool, action: str) -> None:
    calls = _fake(monkeypatch, [])
    _journal(tmp_path / "run" / "heretic" / heretic_tool.STUDY_DIRNAME, finished=finished)
    assert _export(tmp_path)["ok"] is True
    assert _actions(calls)[0] == action


def test_consumers_are_only_those_on_the_active_card(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.webui import gpu_probe

    seen: list[list[str]] = []
    monkeypatch.setattr(gpu_probe, "_run", lambda cmd, _t=3.0: seen.append(cmd) or "12, 2048\n13, 100\n")
    assert [t["pid"] for t in gpu_probe._consumers(0)] == [12, 13]
    assert seen[0][1:3] == ["-i", "0"]
    named = [{"pid": 1, "vram_mib": 2048, "name": "x"}, {"pid": 2, "vram_mib": 100, "name": "y"}]
    monkeypatch.setattr(gpu_probe, "name_consumers", lambda top: top)
    assert gpu_probe.describe_consumers(named) == "x 2.0 GB"


def test_tail_is_kept_separately_from_the_full_output() -> None:
    e = SubprocessFailed("x", 1, "warn", "warn\nthe error")
    assert e.tail == "warn" and "the error" in e.output
    assert SimpleNamespace(output=SubprocessFailed("x", 1, "only").output).output == "only"


def test_a_cancelled_row_stops_saying_stopping(client) -> None:
    from finetune_studio import db

    pid = client.post("/api/projects", json={"name": "cancel row"}).json()["id"]
    run = db.create_run(project_id=pid, name="r", base_model="m", data_path="d")
    row = db.create_export(pid, run["id"], format="heretic", quant="safetensors")
    db.update_export(row["id"], phase="cancelling", phase_detail="stopping…")
    assert db.mark_export_cancelled(row["id"])["phase_detail"] == ""
