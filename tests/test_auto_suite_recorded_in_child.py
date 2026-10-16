"""The post-training quiz must be recorded under the run even though the training child never learns the run id.

Found in the live walkthrough: the suite file was written but ``INSERT INTO auto_suites`` failed on
``run_id NOT NULL`` (the child engine's ``current_run_id`` is None), so the Testing page offered no quiz.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from finetune_studio import db
from finetune_studio.training import engine as eng_mod


def test_suite_row_uses_the_run_directory_name_when_the_engine_has_no_run_id(temp_db, tmp_path) -> None:
    proj = db.create_project(name="p", base_model="x/y")
    run = db.create_run(proj["id"], "r1", base_model="x/y")
    out = tmp_path / "output" / "projects" / proj["id"] / "runs" / run["id"]
    out.mkdir(parents=True)
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"conversations": [{"from": "human", "value": "Which vault?"},
                                                  {"from": "gpt", "value": "Vault 7."}]}) + "\n", encoding="utf-8")

    e = object.__new__(eng_mod.TrainingEngine)
    e.current_run_id = None                      # what the child process has
    e.config = SimpleNamespace(data_path=str(data), output_dir=str(out), project_id="")
    e.state = SimpleNamespace(message="", error="")
    e._notify = lambda: None
    result = e._auto_generate_suite()

    assert result["case_count"] == 1
    with db.cursor() as c:
        row = c.execute("SELECT run_id, project_id, case_count FROM auto_suites").fetchone()
    assert row is not None, "the quiz was not recorded"
    assert (row[0], row[1], row[2]) == (run["id"], proj["id"], 1)


def test_job_run_records_the_suite_and_errors_under_the_plain_run_id(temp_db, tmp_path) -> None:
    # Live Track A (DPO job): current_run_id is "<project>-<run>"; the auto-suite INSERT hit the foreign key and
    # update_run(<composite>) matched no row, so post-train errors never reached the run.
    proj = db.create_project(name="p", base_model="x/y")
    run = db.create_run(proj["id"], "r1", base_model="x/y")
    out = tmp_path / "output" / "projects" / proj["id"] / "runs" / run["id"]
    out.mkdir(parents=True)
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"conversations": [{"from": "human", "value": "Which vault?"},
                                                  {"from": "gpt", "value": "Vault 7."}]}) + "\n", encoding="utf-8")

    e = object.__new__(eng_mod.TrainingEngine)
    e.current_run_id = f"{proj['id']}-{run['id']}"   # what the training job kind sets
    e.current_db_run_id = run["id"]
    e.config = SimpleNamespace(data_path=str(data), output_dir=str(out), project_id=proj["id"])
    e.state = SimpleNamespace(message="", error="")
    e._notify = lambda: None
    assert e._auto_generate_suite()["case_count"] == 1
    with db.cursor() as c:
        assert c.execute("SELECT run_id FROM auto_suites").fetchone()[0] == run["id"]

    e._sync_run_error("GGUF export failed: converter missing")
    assert db.get_run(run["id"])["error"] == "GGUF export failed: converter missing"

    del e.current_db_run_id                           # an engine without the plain id still resolves it
    assert e._run_db_id() == run["id"]
