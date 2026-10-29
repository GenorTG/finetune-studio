"""Quick-start wizard: epochs control and the tiny-run warning."""
from __future__ import annotations

from pathlib import Path

TEMPLATE = (Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
            / "project_wizard.html")


def test_wizard_has_an_epochs_control_mapped_to_num_epochs() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="wiz-epochs"' in html and "startBody.num_epochs = ep" in html
    assert "TINY_STEPS = 100" in html and "optimizer steps" in html


def test_dataset_refresh_recalculates_the_optimizer_step_hint() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    refresh = html.split('if (!which || which === "dataset") {', 1)[1].split(
        'if (!which || which === "brain") {', 1
    )[0]
    export = html.split("window.wizExport = async function (force) {", 1)[1].split(
        "async function wizExplainFailure()", 1
    )[0]

    assert "wizDatasetRows = newest ? Number(newest.qa_count) || 0 : 0;" in refresh
    assert "wizUpdateStepsHint();" in refresh
    assert 'r.headers.get("X-Rows")' in export
    assert "wizDatasetRows = Number(rows) || 0;" in export


def test_quick_start_keeps_tiny_datasets_as_smokes_and_propagates_test_failure() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    hint = html.split("function wizUpdateStepsHint() {", 1)[1].split(
        "(function wireEpochs()", 1
    )[0]
    run_all = html.split("window.wizRunAll = async function () {", 1)[1].split(
        "/* presets + version form", 1
    )[0]

    assert "smoke test, not a quality benchmark" in hint
    assert "Quick Start keeps the selected preset" in hint
    assert "GOOD_STEPS" not in html
    assert "epochsEl.value = String(recommendedEpochs)" not in run_all
    assert "Quick Start kept the " in run_all
    assert "if (!await wizTest()) return;" in run_all
    assert run_all.index("if (!await wizTest()) return;") < run_all.index("Quick start finished:")


def test_wizard_calls_its_generated_quiz_training_set_recall_not_generalization() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    step_test = html.split('id="wiz-step-test"', 1)[1].split('id="wiz-step-version"', 1)[0]

    assert "training-set recall check" in step_test
    assert "not a held-out generalization score" in step_test
    assert "held-out or reworded quiz" in step_test


def test_wizard_recovers_a_missing_merged_model_through_export_job() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    ensure_merged = html.split("async function wizEnsureMerged(run) {", 1)[1].split(
        "window.wizTest = async function () {", 1
    )[0]
    test = html.split("window.wizTest = async function () {", 1)[1].split(
        "window.wizRunAll = async function () {", 1
    )[0]

    assert '"/runs/" + encodeURIComponent(run.id) + "/export"' in ensure_merged
    assert '{ format: "merged" }' in ensure_merged
    assert '"/exports/" + encodeURIComponent(queued.export_id)' in ensure_merged
    assert 'if (!await hasWeights()) throw new Error("merged export finished without loadable model weights")' in ensure_merged
    assert "var mergedPath = await wizEnsureMerged(run);" in test
    assert "model_path: mergedPath" in test
