"""Export job lifecycle on a worker thread, driven through REAL child processes.

``tests/_fake_llama.py`` provides a fake convert script + ``llama-quantize``;
the whole pipeline (run_export -> gguf_convert -> proc_runner) runs unmocked.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.webui import export_jobs
from finetune_studio.webui.export_work import multi_export_work
from tests._fake_llama import install_fake_llama, make_merged_run

TERMINAL = {"done", "failed", "cancelled"}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    install_fake_llama(tmp_path / "llama", monkeypatch)
    pid = db.create_project(name="J", description="")["id"]
    run = make_merged_run(db, pid, tmp_path)
    yield pid, run, tmp_path
    deadline = time.monotonic() + 20
    while export_jobs.active_ids() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not export_jobs.active_ids(), "a job outlived its test"


def _start(pid: str, run: dict, quants: list[str]) -> dict:
    return export_jobs.start_job(
        project_id=pid, run_id=run["id"], fmt="gguf", quant=quants[0], quants=quants,
        make_work=lambda eid: multi_export_work(
            eid, run=run, fmt="gguf", quants=quants, force=True, base_model=None,
        ),
    )


def _wait(eid: str, *, until: Callable[[dict], bool] | None = None, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    row: dict = {}
    while time.monotonic() < deadline:
        row = db.get_export(eid) or {}
        if until(row) if until else row.get("status") in TERMINAL:
            return row
        time.sleep(0.03)
    raise AssertionError(f"timed out waiting on export row: {row}")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (ProcessLookupError, OSError):
        return False


def test_phases_progress_queued_to_done_with_one_row_per_quant(env, monkeypatch) -> None:
    pid, run, _ = env
    monkeypatch.setenv("FAKE_CONVERT_SECONDS", "0.4")
    monkeypatch.setenv("FAKE_QUANT_SECONDS", "0.8")
    row = _start(pid, run, ["f16", "q4_k_m"])
    assert row["status"] == "queued" and row["phase"] == "queued"
    assert row["quants_json"] == '["f16", "q4_k_m"]'

    seen: list[tuple[str, str]] = []
    final = _wait(row["id"], until=lambda r: (
        seen.append((r["phase"], r["phase_detail"])) or r["status"] in TERMINAL
    ))
    assert final["status"] == "done", final
    phases = []
    for phase, _detail in seen:
        if not phases or phases[-1] != phase:
            phases.append(phase)
    # The job walks the pipeline in order; it never jumps back.
    order = ["queued", "preparing", "converting", "quantizing", "finalizing", "done"]
    assert phases == [p for p in order if p in phases], phases
    assert {"converting", "quantizing"} <= set(phases), phases
    assert any("Q4_K_M" in d for p, d in seen if p == "quantizing"), seen
    rows = db.list_exports_for_run(run["id"])
    assert sorted(r["quant"] for r in rows) == ["f16", "q4_k_m"]
    assert all(r["status"] == "done" and r["size_bytes"] > 0 for r in rows)
    assert final["finished_at"] and final["duration_ms"] is not None


def test_heartbeat_advances_while_a_long_child_runs(env, monkeypatch) -> None:
    pid, run, _ = env
    monkeypatch.setenv("FAKE_CONVERT_SECONDS", "4.5")
    row = _start(pid, run, ["f16"])
    first = _wait(row["id"], until=lambda r: r["phase"] == "converting")["heartbeat_at"]
    final = _wait(row["id"])
    assert final["status"] == "done"
    # The child ticks the heartbeat every 2 s: the last beat is well after the first.
    assert final["heartbeat_at"] > first + 2.0


def test_failure_surfaces_the_converter_stderr(env, monkeypatch) -> None:
    pid, run, _ = env
    monkeypatch.setenv("FAKE_CONVERT_FAIL", "ValueError: unsupported tensor dtype bf16x")
    final = _wait(_start(pid, run, ["f16"])["id"])
    assert final["status"] == "failed" and final["phase"] == "failed"
    assert "unsupported tensor dtype bf16x" in final["error"]
    assert "rc=3" in final["error"]
    assert final["finished_at"]


def test_quantize_failure_surfaces_its_stderr_and_cleans_partial_output(env, monkeypatch) -> None:
    pid, run, tmp = env
    monkeypatch.setenv("FAKE_QUANT_FAIL", "llama_model_quantize: invalid ftype")
    final = _wait(_start(pid, run, ["q4_k_m"])["id"])
    assert final["status"] == "failed"
    assert "invalid ftype" in final["error"]
    assert not (tmp / "run-out" / "gguf" / "model-q4_k_m.gguf").exists()


def test_timeout_fails_the_job_and_kills_the_child_group(env, monkeypatch, tmp_path) -> None:
    pid, run, _ = env
    pidfile = tmp_path / "gc.pid"
    monkeypatch.setenv("FTS_EXPORT_CMD_TIMEOUT", "2")
    monkeypatch.setenv("FAKE_QUANT_GRANDCHILD_PIDFILE", str(pidfile))
    t0 = time.monotonic()
    final = _wait(_start(pid, run, ["q4_k_m"])["id"], timeout=40)
    assert final["status"] == "failed"
    assert "timed out" in final["error"]
    assert time.monotonic() - t0 < 30
    grandchild = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild), "timed-out export left llama-quantize's child alive"


def test_cancel_stops_a_running_quantize_and_leaves_no_orphans(env, monkeypatch, tmp_path) -> None:
    pid, run, _ = env
    pidfile = tmp_path / "gc.pid"
    monkeypatch.setenv("FAKE_QUANT_GRANDCHILD_PIDFILE", str(pidfile))
    row = _start(pid, run, ["q4_k_m"])
    _wait(row["id"], until=lambda r: r["phase"] == "quantizing" and pidfile.exists())
    assert export_jobs.cancel_job(row["id"]) is True
    final = _wait(row["id"], timeout=20)
    assert final["status"] == "cancelled" and final["phase"] == "cancelled"
    assert not (tmp_path / "run-out" / "gguf" / "model-q4_k_m.gguf").exists()
    grandchild = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild)
    assert export_jobs.cancel_job(row["id"]) is False  # finished: nothing to cancel


def test_second_export_while_one_runs_is_refused_naming_the_active_one(env, monkeypatch) -> None:
    pid, run, _ = env
    monkeypatch.setenv("FAKE_CONVERT_SECONDS", "1.5")
    first = _start(pid, run, ["f16"])
    with pytest.raises(export_jobs.ExportBusy) as busy:
        _start(pid, run, ["q8_0"])
    assert busy.value.active["id"] == first["id"]
    assert len(db.list_exports_for_run(run["id"])) == 1, "a refused request must not leave a row"
    _wait(first["id"])
    # single-flight frees up once the first job is terminal
    second = _start(pid, run, ["q8_0"])
    assert _wait(second["id"])["status"] == "done"


def test_worker_crash_is_recorded_not_left_running(env, monkeypatch) -> None:
    pid, run, _ = env

    def boom(*_a, **_k):
        raise MemoryError("CUDA out of memory while merging")

    monkeypatch.setattr("finetune_studio.training.run_export.export_trained_run", boom)
    final = _wait(_start(pid, run, ["f16"])["id"])
    assert final["status"] == "failed"
    assert "CUDA out of memory while merging" in final["error"]


def test_next_format_starts_while_the_finished_job_is_still_in_its_epilogue(env, monkeypatch) -> None:
    """The UI posts the next selected format the moment it sees 'done'.

    The worker marks the row done, then still refreshes the model registry
    before releasing its slot; that gap must not turn into a 409.
    """
    pid, run, _ = env
    monkeypatch.setattr("finetune_studio.webui.export_work.refresh_registry_quietly",
                        lambda: time.sleep(1.5))
    first = _start(pid, run, ["f16"])
    _wait(first["id"])
    assert export_jobs.active_ids() == [first["id"]], "epilogue should still be running"
    second = _start(pid, run, ["q8_0"])           # must not raise ExportBusy
    assert _wait(second["id"])["status"] == "done"
