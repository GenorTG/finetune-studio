"""KTO route, unpaired-data converter, preference metrics and SFT-merged start points."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest_plugins = ("tests.test_training_start_guard",)

from tests.test_training_start_guard import _project, _start


def _pairs(n: int = 4) -> list[dict]:
    return [
        {
            "prompt": [{"role": "user", "content": f"Question {i}?"}],
            "chosen": [{"role": "assistant", "content": f"Right {i}."}],
            "rejected": [{"role": "assistant", "content": f"Wrong {i}."}],
        }
        for i in range(n)
    ]


def _write(tmp_path: Path, rows: list[dict], name: str = "rows.jsonl") -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _model(tmp_path: Path) -> str:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    return str(model)


# ── data ─────────────────────────────────────────────────────────────────────

def test_preference_rows_split_into_one_good_and_one_bad_example() -> None:
    from finetune_studio.training.data import preference_to_unpaired

    rows = preference_to_unpaired(_pairs(1))
    assert [r["label"] for r in rows] == [True, False]
    assert rows[0]["completion"][0]["content"] == "Right 0."
    assert rows[1]["completion"][0]["content"] == "Wrong 0."
    assert rows[0]["prompt"] == rows[1]["prompt"]


def test_format_for_unpaired_normalizes_strings_and_bakes_system_prompt() -> None:
    from finetune_studio.training.data import format_for_unpaired

    out = format_for_unpaired([
        {"prompt": "Capital?", "completion": "Vael Harbor.", "label": True},
        {"prompt": "Capital?", "completion": "Paris.", "label": False},
    ], "Be precise.")
    assert out[0]["prompt"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Capital?"},
    ]
    assert out[0]["completion"] == [{"role": "assistant", "content": "Vael Harbor."}]
    assert [r["label"] for r in out] == [True, False]


@pytest.mark.parametrize(("rows", "message"), [
    ([{"prompt": "q", "completion": "a", "label": "yes"}], "label must be true"),
    ([{"prompt": "q", "completion": "a", "label": True}], "both good"),
    ([{"prompt": "q", "completion": "", "label": True}, {"prompt": "q", "completion": "b", "label": False}],
     "completion message 1 needs"),
    ([{"prompt": [{"role": "assistant", "content": "x"}], "completion": "a", "label": True},
      {"prompt": "q", "completion": "b", "label": False}], "prompt must end with a user"),
])
def test_format_for_unpaired_rejects_bad_rows(rows: list[dict], message: str) -> None:
    from finetune_studio.training.data import format_for_unpaired

    with pytest.raises((TypeError, ValueError), match=message):
        format_for_unpaired(rows)


def test_kto_dataset_health_counts_labels_and_flags_a_single_label(tmp_path: Path) -> None:
    from finetune_studio.data.dataset_health import check_dataset

    ok = check_dataset(_write(tmp_path, _pairs(12)), training_mode="kto")
    assert ok["trainable"] == 24
    assert ok["stats"] == {"good_examples": 12, "bad_examples": 12}
    assert ok["verdict"] == "ok"

    only_good = [{"prompt": "q", "completion": f"a{i}", "label": True} for i in range(25)]
    bad = check_dataset(_write(tmp_path, only_good, "good.jsonl"), training_mode="kto")
    assert bad["verdict"] == "errors"
    assert any(i["code"] == "one_label" for i in bad["issues"])


# ── metrics ──────────────────────────────────────────────────────────────────

def test_preference_metrics_reach_state_and_step_log() -> None:
    from finetune_studio.training.engine import TrainingState, apply_trainer_log

    state = TrainingState()
    apply_trainer_log(
        state,
        {"loss": 0.61, "learning_rate": 1e-5, "rewards/accuracies": 0.75, "rewards/margins": 0.1234,
         "rewards/chosen": 0.2, "rewards/rejected": 0.08},
        global_step=5, epoch=0.5, total_steps=20, elapsed=10,
    )
    assert state.pref_metrics["rewards/accuracies"] == 0.75
    assert state.pref_metrics["rewards/margins"] == 0.1234
    assert "acc=0.75" in state.log_lines[-1] and "margin=0.1234" in state.log_lines[-1]

    apply_trainer_log(
        state, {"eval_loss": 0.6, "eval_rewards/accuracies": 1.0, "eval_rewards/margins": 0.3},
        global_step=6, epoch=0.6, total_steps=20, elapsed=11,
    )
    assert state.pref_metrics["eval_rewards/accuracies"] == 1.0
    assert "eval_loss=0.6" in state.log_lines[-1] and "acc=1.0" in state.log_lines[-1]


def test_sft_logs_leave_preference_metrics_empty() -> None:
    from finetune_studio.training.engine import TrainingState, apply_trainer_log

    state = TrainingState()
    apply_trainer_log(state, {"loss": 1.2, "learning_rate": 2e-4}, global_step=1, epoch=0.1,
                      total_steps=10, elapsed=1)
    assert state.pref_metrics == {}
    assert state.log_lines[-1] == "Step 1/10 | loss=1.2 | lr=0.0002"


def test_metrics_cross_the_worker_queue_and_land_in_the_run_record() -> None:
    from finetune_studio.training.engine import TrainingEngine, TrainingState
    from finetune_studio.training.monitor import training_snapshot
    from finetune_studio.training.run_persistence import make_run_state_persister
    from finetune_studio.training.worker import _state_payload

    child = TrainingState(status="training", pref_metrics={"rewards/accuracies": 0.9})
    engine = TrainingEngine()
    engine._apply_state_dict(_state_payload(child))
    assert engine.state.pref_metrics == {"rewards/accuracies": 0.9}
    assert training_snapshot(engine)["pref_metrics"] == {"rewards/accuracies": 0.9}

    written: list[dict] = []
    persist = make_run_state_persister("r1", "out", update_run=lambda rid, **f: written.append(f))
    engine.state.status = "done"
    persist(engine.state)
    assert written[-1]["metrics"]["rewards/accuracies"] == 0.9


# ── routes ───────────────────────────────────────────────────────────────────

def test_kto_route_accepts_preference_rows_and_sets_lora_defaults(
    client, fake_engine, fake_home, tmp_path,
) -> None:
    pid = _project(client)
    response = _start(client, pid, _write(tmp_path, _pairs()), _model(tmp_path), training_mode="kto")

    assert response.status_code == 200, response.text
    config = fake_engine.started["config"]
    assert config.training_mode == "kto"
    assert config.learning_rate == 5e-6 and config.warmup_steps == 0
    assert config.unsloth is False


def test_kto_route_rejects_batch_size_one_and_single_label(
    client, fake_engine, fake_home, tmp_path,
) -> None:
    pid = _project(client)
    one = _start(client, pid, _write(tmp_path, _pairs()), _model(tmp_path),
                 training_mode="kto", batch_size=1)
    assert one.status_code == 400 and "batch_size >= 2" in one.json()["error"]

    only_good = [{"prompt": "q", "completion": f"a{i}", "label": True} for i in range(6)]
    single = _start(client, pid, _write(tmp_path, only_good, "good.jsonl"), _model(tmp_path),
                    training_mode="kto")
    assert single.status_code == 400 and "both good" in single.json()["error"]
    assert not fake_engine.started


def test_explicit_learning_rate_and_beta_win_over_preference_defaults(
    client, fake_engine, fake_home, tmp_path,
) -> None:
    pid = _project(client)
    response = _start(client, pid, _write(tmp_path, _pairs()), _model(tmp_path),
                      training_mode="dpo", learning_rate="2e-5", num_epochs=2, preference_beta="0.3")
    assert response.status_code == 200, response.text
    config = fake_engine.started["config"]
    assert (config.learning_rate, config.num_epochs, config.preference_beta) == (2e-5, 2, 0.3)

    bad = _start(client, pid, _write(tmp_path, _pairs()), _model(tmp_path),
                 training_mode="dpo", preference_beta="7")
    assert bad.status_code == 400 and "preference_beta" in bad.json()["error"]


def _finished_run(pid: str, tmp_path: Path, *, merged: bool, status: str = "done") -> str:
    from finetune_studio import db

    run = db.create_run(project_id=pid, name="SFT run", base_model="m", settings_obj={"training_mode": "sft"})
    out = tmp_path / f"out-{run['id']}"
    if merged:
        (out / "merged").mkdir(parents=True)
        (out / "merged" / "model.safetensors").write_bytes(b"x")
    else:
        out.mkdir()
    db.update_run(run["id"], status=status, output_path=str(out))
    return run["id"]


def test_start_points_list_only_finished_merged_runs(client, fake_home, tmp_path) -> None:
    pid = _project(client)
    good = _finished_run(pid, tmp_path, merged=True)
    _finished_run(pid, tmp_path, merged=False)
    _finished_run(pid, tmp_path, merged=True, status="failed")

    listing = client.get(f"/api/training/start-points/{pid}").json()["start_points"]
    assert [p["run_id"] for p in listing] == [good]
    assert listing[0]["merged_path"].endswith("merged")
    assert client.get("/api/training/start-points/nope").status_code == 404


def test_start_run_replaces_the_base_model_with_its_merged_dir(
    client, fake_engine, fake_home, tmp_path,
) -> None:
    pid = _project(client)
    run_id = _finished_run(pid, tmp_path, merged=True)
    response = _start(client, pid, _write(tmp_path, _pairs()), _model(tmp_path),
                      training_mode="dpo", start_run_id=run_id)
    assert response.status_code == 200, response.text
    assert fake_engine.started["config"].model_path.endswith("/merged")


@pytest.mark.parametrize(("merged", "status", "needle"), [
    (False, "done", "no merged model"),
    (True, "failed", "only finished runs"),
])
def test_unusable_start_run_is_an_honest_400(
    client, fake_engine, fake_home, tmp_path, merged: bool, status: str, needle: str,
) -> None:
    pid = _project(client)
    run_id = _finished_run(pid, tmp_path, merged=merged, status=status)
    response = _start(client, pid, _write(tmp_path, _pairs()), _model(tmp_path),
                      training_mode="dpo", start_run_id=run_id)
    assert response.status_code == 400 and needle in response.json()["error"]
    missing = _start(client, pid, _write(tmp_path, _pairs()), _model(tmp_path),
                     training_mode="dpo", start_run_id="deadbeef")
    assert missing.status_code == 400 and "not found" in missing.json()["error"]
    assert not fake_engine.started


# ── quiz ─────────────────────────────────────────────────────────────────────

def test_kto_quiz_asks_only_the_good_answers(tmp_path: Path) -> None:
    from finetune_studio.testing.generate_suite import generate_suite_from_training_data

    rows = [
        {"prompt": "What is safe?", "completion": "Check the lockout.", "label": True},
        {"prompt": "What is safe?", "completion": "Ignore the lockout.", "label": False},
    ]
    result = generate_suite_from_training_data(_write(tmp_path, rows), str(tmp_path / "suite"))
    assert result["case_count"] == 1
    suite = json.loads(next((tmp_path / "suite").glob("suite_*.json")).read_text())
    assert suite["cases"][0]["correct_answer"] == "Check the lockout."


def test_engine_builds_balanced_kto_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from finetune_studio.training.engine import TrainingConfig, TrainingEngine

    captured: dict = {}

    class _Cfg:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    import trl

    monkeypatch.setattr(trl, "KTOConfig", _Cfg)
    monkeypatch.setattr("finetune_studio.accel.pin_trainer_args", lambda a: a)
    engine = TrainingEngine()
    engine.config = TrainingConfig(output_dir=str(tmp_path), training_mode="kto", batch_size=4,
                                   gradient_accumulation_steps=1, eval_steps=0)
    rows = [{"label": True}] * 3 + [{"label": False}] * 9
    plan = SimpleNamespace(bf16=True, fp16=False, use_cpu=False, optim="adamw_torch")
    engine._preference_args("kto", plan, rows, False)
    assert captured["desirable_weight"] == 3.3  # 9/3 * 1.1: inside TRL's [3.0, 3.99] window
    assert "undesirable_weight" not in captured
    assert captured["beta"] == 0.1 and captured["logging_steps"] >= 1
