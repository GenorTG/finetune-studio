"""New-project template keeps the optional training model list relevant."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finetune_studio"
    / "webui"
    / "templates"
    / "projects.html"
)


def test_base_model_options_exclude_project_exports_and_duplicate_paths() -> None:
    body = _TEMPLATE.read_text(encoding="utf-8")

    assert "'/api/models/list?for_training=true'" in body
    assert "models.filter(m => !m.project_id)" in body
    assert "seenPaths.has(pathKey)" in body
