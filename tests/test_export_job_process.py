"""Exports run as their own job process (supervisor P3 for exports, 2026-10-15): the export runs in another pid,
writes the web app's DB (the spec carries its absolute path, whatever the job's cwd), and a cancel from the web
side reaches it as SIGTERM. Real processes here (the rest of the suite runs jobs in-process, ``jobs.launch``)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.jobs import launch, store
from finetune_studio.webui import export_jobs
from tests._fake_llama import install_fake_llama, make_merged_run


@pytest.fixture
def real_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)
    root = install_fake_llama(tmp_path / "llama", monkeypatch)
    monkeypatch.setenv("LLAMA_CPP_DIR", str(root))             # the job process cannot see a monkeypatch
    monkeypatch.chdir(tmp_path)                                   # a stray cwd-relative DB would land here
    pid = db.create_project(name="P", description="")["id"]
    run = make_merged_run(db, pid, tmp_path)
    yield pid, run
    for j in launch.list_jobs(export_jobs.KIND, running_only=True):
        launch.stop(j["id"])


def _start(pid: str, run: dict, quants: list[str]) -> dict:
    return export_jobs.start_job(project_id=pid, run_id=run["id"], fmt="gguf", quant=quants[0], quants=quants,
                                 work=("multi_export_work", {"run": run, "fmt": "gguf", "quants": quants,
                                                             "force": True, "base_model": None}))


def _wait_row(eid: str, until, timeout: float = 60) -> dict:
    deadline = time.monotonic() + timeout
    row: dict = {}
    while time.monotonic() < deadline:
        row = db.get_export(eid) or {}
        if until(row):
            return row
        time.sleep(0.1)
    raise AssertionError(f"timed out: {row}; job: {launch.describe(export_jobs.job_id(eid))}")


def test_an_export_runs_in_its_own_process_against_the_web_apps_db(real_jobs, tmp_path) -> None:
    pid, run = real_jobs
    row = _start(pid, run, ["q8_0"])
    jid = export_jobs.job_id(row["id"])
    assert store.read_spec(jid).db_path == os.path.abspath(db.connection.settings.db_path)
    final = _wait_row(row["id"], lambda r: r.get("status") in export_jobs.TERMINAL_STATUSES)
    assert final["status"] == "done", final
    d = launch.wait(jid, timeout=30)
    assert d["exit"]["status"] == "done" and d["pid"] != os.getpid()
    assert store.gpu_released(jid)
    assert not (tmp_path / "data").exists(), "the job used a cwd-relative DB instead of the web app's"


def test_cancel_from_the_web_side_stops_the_job_process(real_jobs, monkeypatch) -> None:
    pid, run = real_jobs
    monkeypatch.setenv("FAKE_CONVERT_SECONDS", "20")
    row = _start(pid, run, ["f16"])
    jid = export_jobs.job_id(row["id"])
    _wait_row(row["id"], lambda r: r.get("status") == "running" and r.get("phase") not in (None, "", "queued"))
    assert export_jobs.is_active(row["id"])
    with pytest.raises(export_jobs.ExportBusy):
        _start(pid, run, ["q8_0"])
    assert export_jobs.cancel_job(row["id"]) is True
    final = _wait_row(row["id"], lambda r: r.get("status") in export_jobs.TERMINAL_STATUSES, timeout=40)
    assert final["status"] == "cancelled", final
    assert launch.wait(jid, timeout=30)["exit"]["status"] == "stopped"
    assert export_jobs.cancel_job(row["id"]) is False
    assert export_jobs.running_export_ids() == set()
