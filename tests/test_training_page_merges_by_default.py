"""The Training page merges by default, like the Quick-work wizard.

Found in the live walkthrough: with the box unticked a default run ended 'done' with only an adapter, and the Testing
page's 'auto (latest merged)' model then failed with 'no completed training run found ... run training + merge first'.
"""
from __future__ import annotations

import re
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
        / "project_training.html").read_text(encoding="utf-8")


def test_merge_on_save_is_ticked_by_default() -> None:
    tag = re.search(r'<input[^>]*id="merge-on-save-check"[^>]*>', HTML)
    assert tag and " checked" in tag.group(0)


def test_the_label_tells_what_unticking_costs() -> None:
    assert "auto (latest merged)" in HTML and "merge later at export time" in HTML
