"""In-app (SPA) navigation keeps pages in the same state as a full load (found in the 2026-10-17 browser walkthrough)."""
from __future__ import annotations

from pathlib import Path

W = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"


def _r(rel: str) -> str:
    return (W / rel).read_text(encoding="utf-8")


def test_flow_bar_is_inserted_as_its_own_header_row() -> None:
    # Entering a project from /projects created #workspace-subnav INSIDE the crumb row (.sb-active-bar), where the
    # flex row squeezed the breadcrumb to "dashboard / project-"; the server renders it after that row.
    spa = _r("static/js/spa.js")
    assert 'crumb.closest(".sb-active-bar")' in spa
    assert 'typeof afterId === "function" ? afterId()' in spa
    base = _r("templates/base.html")
    assert base.index('class="sb-active-bar"') < base.index('id="workspace-subnav"')


def test_new_page_starts_at_the_top_of_the_window() -> None:
    # Only #content was scrolled; the document scrolls on the window, so a new page kept the old scrollY.
    assert 'window.scrollTo({ top: 0, behavior: "instant" })' in _r("static/js/spa.js")


def test_training_live_panel_is_torn_down_when_leaving_not_when_arriving() -> None:
    page = _r("templates/project_training.html")
    live = page.split("function arm()", 1)[1].split("</script>", 1)[0]
    assert 'addEventListener("fts:beforeNavigate"' in live
    assert 'addEventListener("fts:navigated"' not in live   # fired after this page's scripts: killed the stream


def test_select_in_training_passes_the_dataset() -> None:
    assert "/training?dataset=' + encodeURIComponent(d.id)" in _r("templates/data_prep.html")


def test_training_dataset_option_states_the_row_count_once() -> None:
    # Exports are named "<project> · sharegpt · 56 rows"; the option appended "· 56 rows" again.
    from jinja2 import Environment

    page = _r("templates/project_training.html")
    line = next(ln for ln in page.splitlines() if "ds.qa_count" in ln and "size_bytes" in ln).strip()
    render = Environment().from_string(line).render
    exported = render(ds={"name": "p · sharegpt · 56 rows", "qa_count": 56, "size_bytes": 16589})
    uploaded = render(ds={"name": "korvane_golden", "qa_count": 2455, "size_bytes": 1024})
    assert exported.count("56 rows") == 1 and "2455 rows" in uploaded


def test_training_template_prompt_matches_short_exact_answers() -> None:
    assert "expert tutor" not in _r("templates/projects.html").split("const _TEMPLATES", 1)[1].split("};", 1)[0]


def test_testing_start_message_is_not_replaced_by_ready() -> None:
    page = _r("templates/project_testing.html")
    fin = page.split("window.runTest = async function()", 1)[1].split("// ── Runs list", 1)[0].split("} finally {", 1)[1]
    assert "const said = status.textContent;" in fin and fin.index("syncRunButton()") < fin.index("status.textContent = said")


def test_saved_comparisons_show_answering_progress() -> None:
    page = _r("templates/project_compare.html")
    groups = page.split("async function loadGroups()", 1)[1].split("catch (e)", 1)[0]
    assert "'answering ' + done + '/' + total" in groups and "'answering ' + (m.progress_done || 0)" in groups


def test_files_content_search_is_named_for_what_it_does() -> None:
    page = _r("templates/project_data.html")
    assert 'id="fb-content-toggle"> search inside files' in page
    assert "'Search inside files and facts… (Enter)'" in page


def test_training_start_does_not_cover_the_live_panel_with_the_activity_drawer() -> None:
    app = _r("static/js/app.js")
    assert 'document.querySelector("[data-live-status]")' in app
    starts = [ln for ln in app.splitlines() if "Training started" in ln]
    assert starts and "window.ftsActivity?.open()" not in "".join(starts)
    assert app.count("surfaceRun();") == 2
    assert "data-live-status" in _r("templates/project_training.html")


def test_review_queue_scrolls_its_own_box_not_the_page() -> None:
    # scrollIntoView also scrolls the window: arriving on Pairs jumped the page down to the queue.
    page = _r("templates/data_prep.html")
    code = "\n".join(line for line in page.splitlines() if not line.strip().startswith("//"))
    assert ".scrollIntoView(" not in code.split("function rqRenderList()", 1)[1].split("function rqRenderDetail()", 1)[0]
    helper = code.split("function rqKeepRowVisible(row)", 1)[1].split("\n}\n", 1)[0]
    assert "box.scrollTop" in helper and "dp-rq-list" in helper
