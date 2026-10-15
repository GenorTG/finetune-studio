"""rag_corpora root must follow FTS_ROOT in every reader/writer."""
from __future__ import annotations

import json

import pytest

from finetune_studio.data.fs import paths


@pytest.fixture
def fts_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_ROOT", tmp_path)
    monkeypatch.setattr(paths, "_PROJECTS", tmp_path / "projects")
    return tmp_path


def test_helper_honors_root(fts_root):
    assert paths.brain_root() == fts_root / "brain"
    assert paths.brain_index_dir("p1") == fts_root / "brain" / "p1"
    assert not (fts_root / "brain").exists()


def test_helper_rejects_traversal(fts_root):
    with pytest.raises(ValueError):
        paths.brain_index_dir("../x")


def test_brain_route_corpus_dir(fts_root):
    from finetune_studio.webui.routes import brain
    assert brain._corpus_dir("p1") == fts_root / "brain" / "p1"


def test_brain_documents_corpus_dir(fts_root):
    from finetune_studio.webui.routes import brain_documents
    assert brain_documents.corpus_dir("p1") == fts_root / "brain" / "p1"


def test_brain_suite_default_corpus_path(fts_root):
    from finetune_studio.testing.brain_suite import default_corpus_path
    assert default_corpus_path("p1") == fts_root / "brain" / "p1"


def test_index_state_reads_fts_root(fts_root):
    from finetune_studio.knowledge import index
    d = fts_root / "brain" / "p1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"extra": {"built_from": {"abcdef123456": "h"}}}))
    assert index.built_from("p1") == {"abcdef123456": "h"}
