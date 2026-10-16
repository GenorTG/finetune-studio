"""Training status is scoped to a project (2026-10-16).

The engine is process-wide and keeps the LAST job's final state readable, so the Training page of project B showed
project A's finished run (DONE badge, full bar, A's step log), the dashboard card said "Done · 54 steps" for days
and the activity feed listed yesterday's run as started now. Now: ``/api/training/status?project_id=`` (and the
SSE) answer ``idle`` for another project's run, the dashboard card says Idle once nothing runs (the last run is
named in the delta line), and the live activity row exists only while a job runs.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

from finetune_studio.training.monitor import training_snapshot


@dataclass
class _State:
    status: str = "idle"
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    final_loss: float | None = None
    learning_rate: float = 0.0
    epoch: float = 0.0
    elapsed: float = 0.0
    eta: float = 0.0
    message: str = ""
    error: str = ""
    log_lines: list = field(default_factory=list)


class _Engine:
    def __init__(self, status: str, project: str | None, step: int = 54) -> None:
        self.state = _State(status=status, current_step=step if status != "idle" else 0, total_steps=step,
                            loss=0.5551, log_lines=["Step 50/54 | loss=0.5551"], message="Training complete!")
        self.current_project_id = project
        self.current_run_id = f"{project}-run1" if project else ""


def test_snapshot_for_another_project_is_idle_and_carries_no_log() -> None:
    eng = _Engine("done", "proj-a")
    own = training_snapshot(eng, "proj-a")
    assert own["status"] == "done" and own["log_lines"] and own["project_id"] == "proj-a" and own["running"] is False
    other = training_snapshot(eng, "proj-b")
    assert other["status"] == "idle" and other["log_lines"] == [] and other["step"] == 0
    assert other["project_id"] == "proj-b" and other["other_project_running"] is False
    unscoped = training_snapshot(eng)
    assert unscoped["status"] == "done" and unscoped["project_id"] == "proj-a"


def test_snapshot_says_when_another_project_holds_the_gpu() -> None:
    eng = _Engine("training", "proj-a", step=10)
    other = training_snapshot(eng, "proj-b")
    assert other["status"] == "idle" and other["other_project_running"] is True
    assert training_snapshot(eng, "proj-a")["running"] is True


def test_status_route_scopes_by_project_and_dashboard_card_is_idle_after_a_run(client, monkeypatch) -> None:
    from finetune_studio import db
    from finetune_studio.webui import app as app_mod

    pid = client.post("/api/projects", json={"name": "owner"}).json()["id"]
    monkeypatch.setattr(app_mod, "training_engine", _Engine("done", pid))
    # The routes module imported the name at import time: patch where it is read.
    from finetune_studio.webui.routes import training as troutes
    monkeypatch.setattr(troutes, "training_engine", app_mod.training_engine)

    assert client.get(f"/api/training/status?project_id={pid}").json()["status"] == "done"
    other = client.get("/api/training/status?project_id=someone-else").json()
    assert other["status"] == "idle" and other["log_lines"] == []
    assert client.get("/api/training/status").json()["status"] == "done"
    assert client.get("/api/training/status-text").text == "Idle"
    delta = client.get("/api/training/progress-text").text
    assert delta.startswith("last run: done · 54 steps") and db.get_project(pid)["name"] in delta

    monkeypatch.setattr(troutes, "training_engine", _Engine("training", pid, step=10))
    assert client.get("/api/training/status-text").text == "Training (10/10)"
    assert client.get("/api/training/progress-text").text.startswith("step 10/10")


def test_activity_live_row_only_while_a_job_runs(monkeypatch) -> None:
    from finetune_studio.webui.routes import activity

    app_mod = types.ModuleType("finetune_studio.webui.app")
    app_mod.training_engine = _Engine("done", "")
    app_mod.inference_engine = types.SimpleNamespace(model=None, model_path="")
    monkeypatch.setitem(sys.modules, "finetune_studio.webui.app", app_mod)
    live = [t for t in activity.collect_activity()["tasks"] if t["kind"] == "training" and t.get("progress") is not None
            and "step" in t.get("message", "")]
    assert live == [], "a finished run is history (its row), not a live task started 'now'"
    app_mod.training_engine = _Engine("training", "", step=10)
    live = [t for t in activity.collect_activity()["tasks"] if t["kind"] == "training" and "step 10/10" in t.get("message", "")]
    assert len(live) == 1
