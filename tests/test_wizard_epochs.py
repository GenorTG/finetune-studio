"""Quick-start wizard: epochs control and the tiny-run warning."""
from __future__ import annotations

from pathlib import Path

TEMPLATE = (Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
            / "project_wizard.html")


def test_wizard_has_an_epochs_control_mapped_to_num_epochs() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="wiz-epochs"' in html and "startBody.num_epochs = ep" in html
    assert "TINY_STEPS = 100" in html and "optimizer steps" in html
