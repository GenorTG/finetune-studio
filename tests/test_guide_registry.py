"""Allow-list registry: pages, controls and fields are real, and values are validated."""
from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio.guide import registry
from finetune_studio.guide.settings_help import SETTING_HELP, explain_setting

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"


def _html(page_key: str) -> str:
    return (TEMPLATES / registry.PAGES[page_key].template).read_text(encoding="utf-8") + \
        (TEMPLATES / "base.html").read_text(encoding="utf-8")


def test_every_page_template_exists() -> None:
    for page in registry.PAGES.values():
        assert (TEMPLATES / page.template).is_file(), page.key


def test_page_urls_are_real_pages(client) -> None:
    """Each allow-listed URL answers 200 (project pages: for a real project)."""
    from finetune_studio import db

    project = db.create_project(name="guide registry probe")
    pid = project["id"] if isinstance(project, dict) else project
    for page in registry.PAGES.values():
        url, err = registry.resolve_page_url(page.key, pid)
        assert err is None
        response = client.get(url)
        assert response.status_code == 200, f"{page.key}: GET {url} -> {response.status_code}"


@pytest.mark.parametrize("control", sorted(registry.CONTROLS))
def test_control_selector_exists_in_its_template(control: str) -> None:
    spec = registry.CONTROLS[control]
    assert spec.page in registry.PAGES
    html = _html(spec.page)
    tokens = registry.selector_tokens(spec.selector)
    assert tokens, spec.selector
    for kind, token in tokens:
        assert f'{kind}="{token}"' in html, f"{control}: {kind}={token} missing from {spec.page} template"


@pytest.mark.parametrize(("page", "name"), sorted(registry.FIELDS))
def test_field_selector_and_scope_exist_in_template(page: str, name: str) -> None:
    spec = registry.FIELDS[(page, name)]
    html = _html(page)
    for kind, token in registry.selector_tokens(spec.selector) + registry.selector_tokens(spec.scope):
        assert f'{kind}="{token}"' in html, f"{page}.{name}: {kind}={token} missing"
    if spec.kind == "choice" and spec.name == "training_mode":
        for value in spec.choices:
            assert f'value="{value}"' in html


def test_choice_fields_match_the_template_options() -> None:
    """Choice values the guide may set are options the page really offers."""
    for (page, name), spec in registry.FIELDS.items():
        if spec.kind != "choice" or name in ("training_mode", "embedder"):
            continue
        html = _html(page)
        for value in spec.choices:
            assert f'value="{value}"' in html, f"{page}.{name}: option {value!r} not in template"
    html = _html("rag")
    for value in registry.FIELDS[("rag", "embedder")].choices:
        assert value in html


def test_every_field_has_help_text() -> None:
    names = {name for (_page, name) in registry.FIELDS}
    assert names <= set(SETTING_HELP), sorted(names - set(SETTING_HELP))
    for name in names:
        row = explain_setting(name)
        assert not row.get("error")
        assert row["meaning"] and row["guidance"] and row["pitfall"]


def test_explain_setting_aliases_and_unknown() -> None:
    assert explain_setting("LR")["setting"] == "learning_rate"
    assert explain_setting("lora rank")["setting"] == "lora_rank"
    assert explain_setting("Epochs")["setting"] == "num_epochs"
    unknown = explain_setting("flux capacitor")
    assert "error" in unknown and "learning_rate" in unknown["known"]


def test_resolve_page_url_requires_project_for_project_pages() -> None:
    assert registry.resolve_page_url("training", "abc") == ("/projects/abc/training", None)
    url, err = registry.resolve_page_url("training", None)
    assert url is None and "needs a project" in (err or "")
    assert registry.resolve_page_url("settings", None) == ("/settings", None)
    url, err = registry.resolve_page_url("../etc/passwd", "abc")
    assert url is None and "unknown page" in (err or "")


@pytest.mark.parametrize(
    ("page", "name", "value", "ok"),
    [
        ("training", "num_epochs", 6, True),
        ("training", "num_epochs", "12", True),
        ("training", "num_epochs", 0, False),
        ("training", "num_epochs", 2.5, False),
        ("training", "num_epochs", "many", False),
        ("training", "lora_rank", 100000, False),
        ("training", "learning_rate", "2e-4", True),
        ("training", "learning_rate", "5", False),
        ("training", "learning_rate", "fast", False),
        ("training", "training_mode", "dpo", True),
        ("training", "training_mode", "orpo", False),
        ("training", "preset", "precision", True),
        ("training", "early_stopping", "true", True),
        ("training", "early_stopping", "maybe", False),
        ("pairs", "difficulty", "expert", True),
        ("pairs", "difficulty", "impossible", False),
        ("pairs", "qa_per_chunk", 11, False),
        ("rag", "chunk_size", 400, True),
        ("rag", "chunk_size", 50, False),
    ],
)
def test_coerce_field_value(page: str, name: str, value: object, ok: bool) -> None:
    coerced, problem = registry.coerce_field_value(registry.FIELDS[(page, name)], value)
    assert (problem is None) is ok, (coerced, problem)
