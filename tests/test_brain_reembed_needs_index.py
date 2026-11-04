"""Smart Brain "Re-embed only" stays off until an index exists (the API answers 404 without one).

Found by the B3 control sweep: on a project with no index the button was enabled and every click ended in
``404 POST /brain/rebuild-vectors``.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

from tests.test_project_brain import _project, _write_fake_corpus

_TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates" / "brain.html"


def _button(html: str) -> str:
    m = re.search(r'<button onclick="reembedIndex\(\)" id="reembed-btn"[^>]*>', html, re.DOTALL)
    assert m, "Re-embed only button missing"
    return m.group(0)


def test_no_index_renders_reembed_disabled_with_the_reason(client) -> None:
    pid = _project(client)
    with patch("finetune_studio.webui.routes.brain_documents.list_indexed_docs", return_value=[]):
        html = client.get(f"/projects/{pid}/brain").text
    btn = _button(html)
    assert " disabled" in btn
    assert "No index yet: build it first" in btn
    assert "let hasIndex = false;" in html


def test_existing_index_renders_reembed_enabled(client, tmp_path: Path, monkeypatch) -> None:
    pid = _project(client)
    corpus = tmp_path / "corpora" / pid
    _write_fake_corpus(corpus)
    monkeypatch.setattr("finetune_studio.webui.routes.brain_documents.corpus_dir",
                        lambda p: corpus if p == pid else tmp_path / "corpora" / p)
    html = client.get(f"/projects/{pid}/brain").text
    btn = _button(html)
    assert " disabled" not in btn
    assert "replace only their vectors" in btn
    assert "let hasIndex = true;" in html


def test_busy_toggle_keeps_reembed_off_without_an_index_and_builds_turn_it_on() -> None:
    src = _TEMPLATE.read_text(encoding="utf-8")
    assert "b.disabled = !!on || (id === 'reembed-btn' && !hasIndex);" in src
    # Every path that follows a finished build (Quick index, Build/Rebuild, a build resumed on load) marks the index.
    assert src.count("markIndexed();") == 3
    assert src.count("await followBuild(") == 4   # the fourth is Re-embed itself, which needs the index already
