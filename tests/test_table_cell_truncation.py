"""Table cells that hold long free text must wrap or carry a tooltip — the
visual probe found suite names clipped 250-363px with no way to read them."""
from __future__ import annotations

from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates"
CSS = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/static/css/app.css"


def _t(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def test_suite_name_cells_wrap() -> None:
    assert '<td class="cell-wrap"><span class="mono">\' + esc(g ? g.label || r.suite_name : r.suite_name)' in _t("project_testing.html")
    assert '<td class="mono cell-wrap">{{ b.suite_name' in _t("benchmarks.html")


def test_truncated_architecture_and_path_have_tooltips() -> None:
    assert 'models-col-arch" title="{{ m.architecture }}"' in _t("models_index.html")
    assert 'title="{{ loaded.path }}"' in _t("inference.html")


def test_export_format_column_fits_two_pills() -> None:
    css = CSS.read_text(encoding="utf-8")
    line = next(ln for ln in css.splitlines() if "#trained-exports-table .exports-col-format" in ln)
    assert "width: 9rem" in line
