"""Visual pass 2026-10-11: one sticky header with one path row, compact resources everywhere, foldable intros,
no decorative promises, and background tabs that stop polling."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "src" / "finetune_studio" / "webui"


def _r(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_breadcrumb_is_the_path_row_not_a_second_bar():
    base = _r("templates/base.html")
    bar = base[base.index('<div class="sb-active-bar">'):base.index("</header>")]
    assert 'id="project-breadcrumb"' in bar and "{% block sb_path %}" in bar
    assert base.count('id="project-breadcrumb"') == 1
    css = _r("static/css/app.css")
    assert ".session-bar { position: sticky; top: 0;" in css
    assert ".sb-active-bar .project-breadcrumb {" in css


def test_every_project_page_uses_the_compact_resources_strip():
    for name in ("project_training", "project_testing", "project", "project_data", "brain", "export_models", "project_settings"):
        t = _r(f"templates/{name}.html")
        if '_resources.html' in t:
            assert "{% set res_compact = true %}" in t, name


def test_intros_lead_with_one_sentence_and_fold_the_rest():
    for name in ("project_wizard", "project_testing", "benchmarks"):
        t = _r(f"templates/{name}.html")
        head = t[t.index('<div class="page-head">'):t.index("</details>")]
        lead = re.search(r'<div class="sub">(.*?)</div>', head, re.DOTALL).group(1)
        assert len(re.sub(r"<[^>]+>|\s+", " ", lead)) < 330, (name, len(lead))
        assert '<details class="page-more"><summary>More about this page</summary>' in head
    assert 'id="quiz-vs-benchmark"' not in _r("templates/benchmarks.html")


def test_dashboard_has_no_fake_terminal():
    idx = _r("templates/index.html")
    assert "ready_for_input" not in idx and 'type "help"' not in idx


def test_polling_is_cheap_and_pauses_in_background_tabs():
    app = _r("static/js/app.js")
    conn = app[app.index("function connCheck"):app.index("function connStart")]
    assert "/api/health" in conn and "/api/system/resources" not in conn and "document.hidden" in conn
    res = _r("templates/_resources.html")
    assert "if (!document.hidden) poll()" in res and "setInterval(poll, 3000)" not in res
    assert "visibilitychange" in res


def test_asset_versions_bumped_with_the_change():
    base = _r("templates/base.html")
    assert "app.css?v=139" in base and "app.js?v=29" in base


def test_chat_page_shows_model_status_once():
    chat = _r("templates/chat_v2.html")
    assert "chat-status-pill" not in chat  # the sprite row (sprites.js) is the one status
    assert "chat-model-warn" in chat and "_chatCheckModel" in chat


def test_host_resources_stay_off_the_knowledge_pages_and_below_the_title():
    """The card pushed every project page's title and next action down (a quarter of a phone screen): it is on the
    dashboard, Training, Inference and My models only, after the page title; other pages have the top-bar VRAM pill."""
    for name in ("project", "project_data", "brain", "project_testing", "export_models", "project_settings",
                 "project_models", "data_editor"):
        assert '_resources.html' not in _r(f"templates/{name}.html"), name
    for name in ("index", "project_training", "inference", "models_index"):
        t = _r(f"templates/{name}.html")
        assert '{% include "_resources.html" %}' in t, name
        assert t.index("<h1") < t.index('{% include "_resources.html" %}'), name
    css = _r("static/css/app.css")
    assert ".res-compact #accel-line, .res-compact #sys-resources-updated { display: none; }" in css
