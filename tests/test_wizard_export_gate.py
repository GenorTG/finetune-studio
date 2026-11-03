"""Quick-start wizard: the export gate offers its remedies, and the quiz reports both skills."""
from __future__ import annotations

from pathlib import Path

TEMPLATE = (Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
            / "project_wizard.html")


def _html() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def test_blocked_export_offers_force_and_the_pairs_page() -> None:
    html = _html()
    # the 409 body names the files; the wizard turns the two remedies into controls
    assert 'id="wiz-dataset-gate"' in html
    assert "r.status === 409" in html and "d.incomplete_files" in html
    assert '"?force=true"' in html and "window.wizExport(true)" in html
    assert "Export the approved pairs anyway" in html and "Open the pairs page to finish" in html


def test_force_is_never_implicit() -> None:
    """Only an explicit click passes force; the chained Run-all path and the plain button do not."""
    html = _html()
    assert 'onclick="wizExport()"' in html
    assert "force === true" in html


def test_empty_approved_set_disables_dataset_and_training_actions() -> None:
    html = _html()
    assert 'id="wiz-dataset-btn"' in html and 'id="wiz-dataset-btn" onclick="wizExport()" title="Approve at least one Q&A pair first" disabled' in html
    assert 'id="flow-dataset-export-btn" onclick="flowExport()" disabled' in html
    assert 'id="wiz-train-btn"' in html and 'id="wiz-train-btn" onclick="wizTrain()" title="Build a non-empty dataset and choose a base model first" disabled' in html
    assert "wizApprovedPairs = approved;" in html
    assert "button.disabled = busy || !hasApprovedPairs;" in html
    assert "review or add a pair before building a dataset" in html
    assert "Number.isFinite(newestRows) && newestRows > 0" in html
    assert "newest dataset is empty or has no rows" in html
    assert "if (wizDatasetRows < 1)" in html


def test_quiz_result_separates_memory_from_with_context_rows() -> None:
    html = _html()
    assert "x.judge_input && x.judge_input.grounded" in html
    assert "from memory (no context)" in html


def test_run_all_fills_open_points_before_the_export() -> None:
    """Mining left 2 of 17 points open, so Run all stopped at the export gate although one more pass covers them."""
    html = _html()
    run_all = html[html.index("window.wizRunAll"):]
    run_all = run_all[:run_all.index("if (!await wizTrain()) return;")]
    assert run_all.index("await wizGenerate()") < run_all.index("wizOpenPoints()") < run_all.index("await wizExport()")
    assert "pass <= MAX_FILL_PASSES" in run_all
    # a pass that closes nothing ends the loop: no endless helper calls on a point it cannot state
    assert "open.points >= lastOpen) break;" in run_all
    assert "await wizExport(true)" not in run_all  # still never forces the gate


def test_dataset_step_does_not_promise_an_extractive_fill() -> None:
    html = _html()
    assert "extractive pairs" not in html.lower()
    assert "offers to export the" in html and "approved pairs anyway" in html
