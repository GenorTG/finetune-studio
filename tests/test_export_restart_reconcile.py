"""Export rows left non-terminal by a dead server are failed at startup, never 'running' forever."""
from __future__ import annotations

from finetune_studio import db


def _row(pid: str, rid: str, status: str, phase: str = "") -> str:
    row = db.create_export(project_id=pid, run_id=rid, quant="Q4_K_M")
    db.update_export(row["id"], status=status, phase=phase or status)
    return row["id"]


def _project_and_run() -> tuple[str, str]:
    pid = db.create_project(name="Stale", description="")["id"]
    rid = db.create_run(project_id=pid, name="r", base_model="m", settings_obj={})["id"]
    return pid, rid


def test_reconcile_fails_queued_and_running_rows_only() -> None:
    pid, rid = _project_and_run()
    queued = _row(pid, rid, "queued")
    running = _row(pid, rid, "running", "quantizing")
    done = _row(pid, rid, "done")
    cancelled = _row(pid, rid, "cancelled")

    assert db.reconcile_stale_exports() == 2

    for eid in (queued, running):
        row = db.get_export(eid)
        assert row["status"] == "failed" and row["phase"] == "failed"
        assert "interrupted by service restart" in row["error"]
        assert row["finished_at"]
    assert db.get_export(done)["status"] == "done"
    assert db.get_export(cancelled)["status"] == "cancelled"
    assert db.list_active_exports() == []


def test_app_startup_runs_the_reconcile(client) -> None:
    """A real lifespan start (what a service restart does) fails the orphaned rows."""
    from fastapi.testclient import TestClient

    import finetune_studio.webui.app as app_module

    pid, rid = _project_and_run()
    orphan = _row(pid, rid, "running", "converting")
    with TestClient(app_module.app):
        pass
    row = db.get_export(orphan)
    assert row["status"] == "failed"
    assert "interrupted" in row["error"]
    got = client.get(f"/api/projects/{pid}/exports/{orphan}").json()
    assert got["status"] == "failed" and got["active"] is False
