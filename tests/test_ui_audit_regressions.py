"""Regression pins for issues found during the 2026-10-01 UI audit."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "src/finetune_studio/webui/templates"


def test_project_delete_confirmation_treats_project_name_as_text() -> None:
    source = (TEMPLATES / "projects.html").read_text(encoding="utf-8")
    assert "this.closest('.proj-card').dataset.name" in source
    assert "body.textContent = `Permanently delete ${name" in source
    assert "body.innerHTML = `Permanently delete ${name" not in source


def test_data_prep_inline_handlers_encode_dynamic_strings_as_js_literals() -> None:
    source = (TEMPLATES / "data_prep.html").read_text(encoding="utf-8")
    assert "function jsArg(value)" in source
    assert "escapeHtml(JSON.stringify(String(value)))" in source
    assert "flPromptRenameFolder(' + jsArg(f.id)" in source
    assert "flPreviewFile(' + jsArg(f.id)" in source
    assert "flRestoreFile(' + jsArg(it.id)" in source


def test_wizard_loads_helper_and_only_approves_new_successful_pairs() -> None:
    source = (TEMPLATES / "project_wizard.html").read_text(encoding="utf-8")
    assert "prov.active.id !== helperId" in source
    assert "var existingQaIds = new Set" in source
    assert "succeededSourceIds.add(s.id)" in source
    assert "!existingQaIds.has(i.id) && succeededSourceIds.has(i.source_id)" in source


def test_spa_navigation_emits_lifecycle_events_and_resource_timer_cleans_up() -> None:
    spa = (ROOT / "src/finetune_studio/webui/static/js/spa.js").read_text(encoding="utf-8")
    resources = (TEMPLATES / "_resources.html").read_text(encoding="utf-8")
    assert '"fts:beforeNavigate"' in spa
    assert '"fts:navigated"' in spa
    assert "clearInterval(pollTimer)" in resources


def test_rag_clear_copy_does_not_claim_directory_deletion() -> None:
    source = (TEMPLATES / "rag.html").read_text(encoding="utf-8")
    assert "Clear indexed corpus" in source
    assert "keeps the corpus directory and bundled model files" in source
