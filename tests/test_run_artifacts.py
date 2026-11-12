"""A run's actions (Set production, Open in inference, Export) need a model in its output folder, not just a status:
a run stopped in its first seconds saved nothing, and offering them was a dead end."""
from __future__ import annotations

from pathlib import Path

from finetune_studio.training import run_artifacts as ra


def _adapter(d: Path) -> None:
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"x")


def test_a_stopped_run_without_a_checkpoint_has_nothing_to_use(tmp_path: Path) -> None:
    out = tmp_path / "run"
    (out / "adapter").mkdir(parents=True)          # the folder exists, empty
    ok, why = ra.usable({"status": "stopped", "output_path": str(out)})
    assert not ok and "Stopped before a checkpoint" in why
    ok, why = ra.usable({"status": "failed", "output_path": str(out)})
    assert not ok and "Failed before a checkpoint" in why
    ok, why = ra.usable({"status": "done", "output_path": str(tmp_path / "missing")})
    assert not ok and "no model files" in why


def test_finished_and_stopped_runs_with_a_model_are_usable(tmp_path: Path) -> None:
    _adapter(tmp_path / "a" / "adapter")
    assert ra.usable({"status": "done", "output_path": str(tmp_path / "a")}) == (True, "")
    _adapter(tmp_path / "s" / "checkpoint-20")
    assert ra.usable({"status": "stopped", "output_path": str(tmp_path / "s")}) == (True, "")
    (tmp_path / "m" / "merged").mkdir(parents=True)
    (tmp_path / "m" / "merged" / "config.json").write_text("{}")
    (tmp_path / "m" / "merged" / "model.safetensors").write_bytes(b"x")
    assert ra.usable({"status": "done", "output_path": str(tmp_path / "m")})[0]
    ok, why = ra.usable({"status": "running", "output_path": str(tmp_path / "a")})
    assert not ok and "Still running" in why


def test_the_runs_api_and_page_carry_it(client, tmp_path: Path) -> None:
    from finetune_studio import db

    pid = client.post("/api/projects", json={"name": "runs-usable"}).json()["id"]
    empty = tmp_path / "empty"
    (empty / "adapter").mkdir(parents=True)
    run = db.create_run(project_id=pid, name="quick stop")
    db.update_run(run["id"], status="stopped", output_path=str(empty))
    rows = client.get(f"/api/projects/{pid}/runs").json()
    assert rows[0]["usable"] is False and "Stopped before a checkpoint" in rows[0]["usable_reason"]
    page = client.get(f"/projects/{pid}/training").text
    assert "Stopped before a checkpoint was saved" in page
