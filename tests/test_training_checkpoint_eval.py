"""Training form checkpoint/eval/early-stopping/Unsloth fields reach the trainer.

Regression (UI coverage audit 2026-10-03): the project training form sent
``save_checkpoints``, ``early_stopping``, ``save_limit`` and ``eval_steps`` but
``/api/training/start`` never read them, the 10% held-out split was never
passed to the trainer, and the live form had no Unsloth control at all.
"""

from __future__ import annotations

from pathlib import Path

from finetune_studio.training.engine import TrainingConfig
from finetune_studio.training.sft_args import (
    build_sft_args_from_config,
    checkpoint_eval_kwargs,
)
from tests import test_training_start_guard as _guard
from tests.test_training_start_guard import _jsonl, _project, _start

# Re-exported pytest fixtures (module attributes, so pytest collects them here).
fake_engine = _guard.fake_engine
fake_home = _guard.fake_home

# ── pure mapping: TrainingConfig -> SFTConfig kwargs ──────────────────────────


def test_defaults_keep_periodic_checkpoints_and_no_eval() -> None:
    kw = checkpoint_eval_kwargs(TrainingConfig(), has_eval=True)
    assert kw["save_strategy"] == "steps"
    assert kw["save_total_limit"] == 3
    assert kw["eval_strategy"] == "no"
    assert "load_best_model_at_end" not in kw


def test_checkpoints_off_disables_saving() -> None:
    kw = checkpoint_eval_kwargs(TrainingConfig(save_checkpoints=False), has_eval=True)
    assert kw["save_strategy"] == "no"
    assert "save_total_limit" not in kw


def test_eval_steps_enable_step_eval_only_with_eval_data() -> None:
    cfg = TrainingConfig(eval_steps=25)
    on = checkpoint_eval_kwargs(cfg, has_eval=True)
    assert on["eval_strategy"] == "steps" and on["eval_steps"] == 25
    off = checkpoint_eval_kwargs(cfg, has_eval=False)
    assert off["eval_strategy"] == "no"


def test_early_stopping_aligns_saves_to_evals_and_keeps_best() -> None:
    cfg = TrainingConfig(eval_steps=20, early_stopping=True, save_checkpoints=False,
                         save_steps=100, save_total_limit=2)
    kw = checkpoint_eval_kwargs(cfg, has_eval=True)
    # HF requires save_steps to be a multiple of eval_steps for load_best_model_at_end;
    # early stopping must save even when periodic checkpoints were unticked.
    assert kw["save_strategy"] == "steps" and kw["save_steps"] == 20
    assert kw["load_best_model_at_end"] is True
    assert kw["metric_for_best_model"] == "eval_loss"
    assert kw["greater_is_better"] is False
    assert kw["save_total_limit"] == 2


def test_early_stopping_without_eval_data_degrades_to_plain_run() -> None:
    cfg = TrainingConfig(eval_steps=20, early_stopping=True)
    kw = checkpoint_eval_kwargs(cfg, has_eval=False)
    assert kw["eval_strategy"] == "no"
    assert "load_best_model_at_end" not in kw


def test_sft_config_accepts_early_stopping_combo(tmp_path: Path) -> None:
    cfg = TrainingConfig(output_dir=str(tmp_path), bf16=False, eval_steps=10,
                         early_stopping=True, save_total_limit=2)
    args = build_sft_args_from_config(cfg, has_eval=True)
    assert args.load_best_model_at_end is True
    assert args.save_steps == 10 and args.eval_steps == 10
    assert args.save_total_limit == 2


# ── route: form fields reach TrainingConfig ──────────────────────────────────


def _model(tmp_path: Path) -> str:
    m = tmp_path / "m"
    m.mkdir()
    return str(m)


def test_form_fields_reach_config(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    # Exactly what Object.fromEntries(new FormData(form)) sends: strings, hidden "0"
    # overwritten by the checked box's "1".
    r = _start(client, pid, _jsonl(tmp_path), _model(tmp_path),
               save_checkpoints="1", early_stopping="1", save_limit="2",
               eval_steps="40", unsloth="1")
    assert r.json().get("status") == "started", r.text
    cfg = fake_engine.started["config"]
    assert cfg.save_checkpoints is True
    assert cfg.early_stopping is True
    assert cfg.save_total_limit == 2
    assert cfg.eval_steps == 40
    assert cfg.unsloth is True


def test_unticked_boxes_are_honoured(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), _model(tmp_path),
               save_checkpoints="0", early_stopping="0", unsloth="0", eval_steps="0")
    assert r.json().get("status") == "started", r.text
    cfg = fake_engine.started["config"]
    assert cfg.save_checkpoints is False
    assert cfg.early_stopping is False
    assert cfg.unsloth is False


def test_api_callers_omitting_fields_keep_defaults(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), _model(tmp_path))
    assert r.json().get("status") == "started", r.text
    cfg = fake_engine.started["config"]
    assert cfg.save_checkpoints is True and cfg.early_stopping is False
    assert cfg.eval_steps == 0


def test_early_stopping_without_eval_is_400(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    r = _start(client, pid, _jsonl(tmp_path), _model(tmp_path),
               early_stopping="1", eval_steps="0")
    assert r.status_code == 400, r.text
    assert "Eval every N steps" in r.json()["error"]
    assert not fake_engine.started


def test_bad_save_limit_is_400(client, fake_engine, fake_home, tmp_path):
    pid = _project(client)
    model = _model(tmp_path)
    for bad in ("0", "abc"):
        r = _start(client, pid, _jsonl(tmp_path), model, save_limit=bad)
        assert r.status_code == 400, (bad, r.text)
    assert not fake_engine.started


# ── template: the live form exposes every field the route reads ──────────────


def test_live_training_form_exposes_fields() -> None:
    from finetune_studio import webui

    html = (Path(webui.__file__).parent / "templates" / "project_training.html").read_text()
    for name in ("save_checkpoints", "early_stopping", "save_limit", "eval_steps", "unsloth"):
        assert f'name="{name}"' in html, name
    # Unchecked checkboxes are omitted by FormData; a hidden "0" before each makes
    # "unticked" reach the server instead of falling back to the API default.
    for name in ("save_checkpoints", "unsloth"):
        assert f'type="hidden" name="{name}" value="0"' in html, name


# ── engine: an early stop is named in the final status ───────────────────────


def test_early_stop_step_only_when_stopped_short() -> None:
    from types import SimpleNamespace

    from finetune_studio.training.engine import _early_stop_step

    es = TrainingConfig(early_stopping=True, eval_steps=5)
    assert _early_stop_step(es, SimpleNamespace(global_step=35, max_steps=420)) == 35
    assert _early_stop_step(es, SimpleNamespace(global_step=420, max_steps=420)) == 0
    plain = TrainingConfig()
    assert _early_stop_step(plain, SimpleNamespace(global_step=35, max_steps=420)) == 0


def test_completion_message_names_early_stop() -> None:
    from finetune_studio.training.engine import TrainingEngine

    eng = TrainingEngine()
    eng.state.total_steps = 420
    assert eng._completion_message() == "Training complete!"
    eng._early_stop_step = 35
    msg = eng._completion_message()
    assert "Stopped early at step 35/420" in msg and msg.startswith("Training complete!")
