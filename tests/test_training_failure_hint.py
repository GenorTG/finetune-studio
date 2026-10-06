"""Failed training runs: plain-language diagnosis, the /failure endpoint, and the page wiring."""
from __future__ import annotations

from pathlib import Path

from finetune_studio import db
from finetune_studio.training.failure_hint import diagnose_failure

ROOT = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"

OOM = (
    "OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of "
    "23.56 GiB of which 64.94 MiB is free. Process 1762 has 260.00 MiB memory in use. Process 679073 has "
    "6.50 GiB memory in use. Including non-PyTorch memory, this process has 22.86 GiB memory in use."
)


def _knobs(hint) -> dict[str, object]:
    return {k.field: k for k in hint.knobs}


def test_non_oom_errors_get_no_hint() -> None:
    assert diagnose_failure("ValueError: dataset has no rows", {"batch_size": 2}) is None
    assert diagnose_failure("", {}) is None
    assert diagnose_failure("BloomForCausalLM failed to build the tokenizer", {}) is None


def test_oom_knobs_are_computed_from_the_runs_own_settings() -> None:
    hint = diagnose_failure(OOM, {"batch_size": 8, "gradient_accumulation_steps": 1,
                                  "max_seq_length": 4096, "lora_rank": 64, "unsloth": False})
    assert hint is not None and hint.kind == "oom"
    k = _knobs(hint)
    assert (k["batch_size"].current, k["batch_size"].suggested) == (8, 4)
    assert "1 → 2" in k["batch_size"].note          # effective batch preserved
    assert (k["max_seq_length"].current, k["max_seq_length"].suggested) == (4096, 2048)
    assert k["unsloth"].suggested == "on" and k["lora_rank"].suggested == 16
    assert "23.6 GiB" in hint.summary and "22.9 GiB" in hint.summary


def test_oom_does_not_offer_knobs_that_cannot_help() -> None:
    hint = diagnose_failure(OOM, {"batch_size": 1, "gradient_accumulation_steps": 4,
                                  "max_seq_length": 256, "lora_rank": 16, "unsloth": True})
    k = _knobs(hint)
    assert k["batch_size"].suggested == 1 and "cannot go lower" in k["batch_size"].note
    assert "max_seq_length" not in k and "unsloth" not in k and "lora_rank" not in k


def test_oom_names_foreign_gpu_users_only_when_they_hold_real_memory() -> None:
    tips = " ".join(diagnose_failure(OOM, {}).tips)
    assert "PID 679073" in tips and "PID 1762" not in tips
    quiet = OOM.replace("6.50 GiB", "300.00 MiB")
    assert "PID" not in " ".join(diagnose_failure(quiet, {}).tips)


def test_failure_endpoint_reports_only_the_newest_run_when_failed(client) -> None:
    pid = client.post("/api/projects", json={"name": "Fail", "system_prompt": "", "allow_duplicate": True}).json()["id"]
    assert client.get("/api/training/failure", params={"project_id": "nope"}).status_code == 404
    assert client.get("/api/training/failure", params={"project_id": pid}).json() == {"failure": None}
    run = db.create_run(project_id=pid, name="r1", base_model="m", data_path="d.jsonl", rag_ids=[],
                        settings_obj={"batch_size": 4, "gradient_accumulation_steps": 2, "max_seq_length": 2048},
                        system_prompt="", system_prompt_mode="bake")
    db.update_run(run["id"], status="failed", error=OOM)
    f = client.get("/api/training/failure", params={"project_id": pid}).json()["failure"]
    assert f["run_id"] == run["id"] and f["error"] == OOM
    assert f["hint"]["kind"] == "oom"
    assert {k["field"] for k in f["hint"]["knobs"]} >= {"batch_size", "max_seq_length"}
    # A later run supersedes the failure: Live status must not show a stale error.
    later = db.create_run(project_id=pid, name="r2", base_model="m", data_path="d.jsonl", rag_ids=[],
                          settings_obj={}, system_prompt="", system_prompt_mode="bake")
    db.update_run(later["id"], status="done")
    assert client.get("/api/training/failure", params={"project_id": pid}).json() == {"failure": None}


def test_non_oom_failure_returns_error_without_hint(client) -> None:
    pid = client.post("/api/projects", json={"name": "Fail2", "system_prompt": "", "allow_duplicate": True}).json()["id"]
    run = db.create_run(project_id=pid, name="r", base_model="m", data_path="d", rag_ids=[],
                        settings_obj={}, system_prompt="", system_prompt_mode="bake")
    db.update_run(run["id"], status="failed", error="KeyError: 'messages'")
    f = client.get("/api/training/failure", params={"project_id": pid}).json()["failure"]
    assert f["hint"] is None and f["error"] == "KeyError: 'messages'"


def test_status_payloads_carry_the_owning_project_id() -> None:
    from finetune_studio.training.monitor import training_snapshot

    class _Eng:
        current_project_id = "p1"

        class state:  # noqa: N801 — attribute bag standing in for TrainingState
            status, current_step, total_steps, loss, final_loss = "error", 0, 0, 0.0, None
            learning_rate, epoch, elapsed, eta, message, error, log_lines = 0, 0, 0, 0, "m", "e", []

    assert training_snapshot(_Eng())["project_id"] == "p1"


def test_training_page_shows_failed_runs_and_expandable_errors() -> None:
    html = (ROOT / "project_training.html").read_text(encoding="utf-8")
    live = html.split("Live status")[1].split("Past runs")[0]
    assert 'id="train-failure"' in live and 'id="train-failure-full"' in live and "/api/training/failure" in html
    assert "<details" in html.split("Past runs")[1] and 'class="run-err run-err-cell' in html
    assert "run-err-cell" in html.split("function renderRunsTable")[1]   # client-side rows too
    assert "s.project_id !== PID" in html
