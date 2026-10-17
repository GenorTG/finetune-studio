"""Short runs keep their learning rate, and the advisor's LoRA alpha reaches the run (walkthrough 2026-10-17)."""
from __future__ import annotations

from pathlib import Path

from finetune_studio.training.engine import TrainingConfig, TrainingEngine

TEMPLATE = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/project_training.html"


def _engine(warmup: int) -> TrainingEngine:
    eng = TrainingEngine.__new__(TrainingEngine)
    eng.config = TrainingConfig(warmup_steps=warmup)
    eng.state = type("S", (), {"log_lines": []})()
    return eng


def test_warmup_is_capped_at_a_tenth_of_a_short_run_and_says_so() -> None:
    eng = _engine(100)
    assert eng._warmup_for(112) == 11
    assert "Warmup 100 -> 11 steps" in eng.state.log_lines[0]


def test_warmup_within_the_cap_is_kept_silently() -> None:
    eng = _engine(30)
    assert eng._warmup_for(4200) == 30
    assert eng.state.log_lines == []


def test_zero_warmup_stays_zero() -> None:
    assert _engine(0)._warmup_for(10) == 0


def test_training_form_has_the_alpha_field_the_advisor_fills() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'name="lora_alpha"' in html
    assert "lora_alpha: d.lora_alpha" in html  # the advisor writes into [name="lora_alpha"]
