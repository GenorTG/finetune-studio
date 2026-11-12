"""Smart Brain "Re-embed only" stays off until an index exists (the API answers 404 without one).

Found by the B3 control sweep: on a project with no index the button was enabled and every click ended in
``404 POST /brain/rebuild-vectors``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

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


def test_the_button_follows_the_index_status() -> None:
    src = _TEMPLATE.read_text(encoding="utf-8")
    assert "b.disabled = !!on || (id === 'reembed-btn' && !hasIndex);" in src
    # A build creates the index and Clear the index removes it: every status refresh sets the button (it stayed on
    # after a clear, then every click was a 404).
    assert "setHasIndex(d.exists && d.chunks > 0);" in src
    assert "markIndexed" not in src


def _empty_index(root: Path) -> Path:
    """What Clear the index leaves: the folder and its files, no documents, no chunks."""
    root.mkdir(parents=True)
    pd.DataFrame({"id": [], "text": []}).to_parquet(root / "chunks.parquet", index=False)
    np.save(root / "vectors.npy", np.zeros((0, 8)))
    (root / "vectors.idx.json").write_text("{}", encoding="utf-8")
    (root / "bm25.json").write_text("{}", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "name": "t", "version": "2", "created_at": 1.0, "updated_at": 1.0,
        "embedding_model": {"name": "x", "dim": 8},
        "chunk_settings": {"size": 400, "overlap": 80, "splitter": "word"},
        "rag_settings": {}, "documents": 0, "chunks": 0, "extra": {},
    }), encoding="utf-8")
    return root


def test_api_refuses_an_emptied_index_with_the_reason(client, tmp_path: Path, monkeypatch) -> None:
    pid = _project(client)
    corpus = _empty_index(tmp_path / "c")
    monkeypatch.setattr("finetune_studio.knowledge.index.brain_index_dir", lambda _p: corpus)
    r = client.post(f"/api/projects/{pid}/brain/rebuild-vectors", json={})
    assert r.status_code == 404, r.text
    assert "the index is empty" in r.json()["detail"]


def test_store_refuses_an_emptied_index_before_loading_an_embedder(tmp_path: Path) -> None:
    from finetune_studio.smart_brain import BrainIndex

    store = BrainIndex(_empty_index(tmp_path / "c"))
    assert store.exists()
    with patch("finetune_studio.smart_brain.store.get_embedder") as embedder, pytest.raises(ValueError, match="empty"):
        store.rebuild_vectors()
    embedder.assert_not_called()
