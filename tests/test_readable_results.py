"""Readable result surfaces — no raw JSON dumps on main run/result panels."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES = _ROOT / "src" / "finetune_studio" / "webui" / "templates"
_TRAINING = _TEMPLATES / "project_training.html"
_KV = _TEMPLATES / "_kv_grid.html"
_CASE = _TEMPLATES / "_case_results.html"
_EXPORT = _TEMPLATES / "export_models.html"
_TESTING = _TEMPLATES / "project_testing.html"


def test_training_detail_uses_kv_grid_not_json_pre() -> None:
    html = _TRAINING.read_text(encoding="utf-8")
    assert '_kv_grid.html' in html
    assert "detail_run.settings" in html
    # Must not dump settings/metrics as indented JSON <pre>.
    assert "detail_run.settings|tojson(indent=2)" not in html
    assert "detail_run.metrics|tojson(indent=2)" not in html
    assert "run-error" in html
    assert '<pre class="mono text-xs run-settings">' not in html


def test_kv_grid_renders_fields() -> None:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("_kv_grid.html")
    html = tpl.render(
        title="Settings",
        rows={"epochs": 3, "lr": 0.0002, "nested": {"a": 1}},
    )
    assert "Settings" in html
    assert "epochs" in html
    assert "3" in html
    assert "lr" in html
    assert "run-kv-table" in html
    assert "<pre" not in html


def test_case_results_is_table_not_json_dump() -> None:
    html = _CASE.read_text(encoding="utf-8")
    assert "case-results-table" in html
    assert "verdict-badge" in html
    assert "tojson(indent" not in html


def test_testing_results_render_table() -> None:
    html = _TESTING.read_text(encoding="utf-8")
    assert "case-results-table" in html
    assert "renderResults" in html
    assert "JSON.stringify(scores" not in html


def test_export_result_panel_is_field_grid() -> None:
    html = _EXPORT.read_text(encoding="utf-8")
    assert 'id="export-result"' in html
    assert "run-kv-table" in html
    assert "showResultCard" in html
    # Failures must surface actionable text (not a silent 200).
    assert "export failed" in html
    assert "Missing" in html
    assert "non-empty GGUF" in html or "without artifacts" in html
