"""E2E-29: reconcile_stale_runs marks orphaned in-flight runs as failed."""
from __future__ import annotations

import time

from finetune_studio import db
from finetune_studio.db.runs import reconcile_stale_runs


def test_reconcile_stale_runs_marks_in_flight_failed_leaves_done(mock_settings) -> None:
    proj = db.create_project(name="stale-runs-probe")
    pid = proj["id"]
    loading = db.create_run(pid, name="loading-run", base_model="m")
    training = db.create_run(pid, name="training-run", base_model="m")
    done = db.create_run(pid, name="done-run", base_model="m")
    db.update_run(loading["id"], status="loading", started_at=time.time())
    db.update_run(training["id"], status="training", started_at=time.time())
    db.update_run(done["id"], status="done", output_path="/tmp/x", finished_at=time.time())

    n = reconcile_stale_runs()
    assert n >= 2

    loading2 = db.get_run(loading["id"])
    training2 = db.get_run(training["id"])
    done2 = db.get_run(done["id"])
    assert loading2 is not None and training2 is not None and done2 is not None
    assert loading2["status"] == "failed"
    assert "Interrupted: server restarted" in (loading2.get("error") or "")
    assert training2["status"] == "failed"
    assert "Interrupted: server restarted" in (training2.get("error") or "")
    assert done2["status"] == "done"
    assert (done2.get("error") or "") == ""


def test_reconcile_stale_runs_exported_from_db_package() -> None:
    assert callable(db.reconcile_stale_runs)
