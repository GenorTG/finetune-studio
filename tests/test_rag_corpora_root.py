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
    assert paths.rag_corpora_root() == fts_root / "rag_corpora"
    assert paths.rag_corpus_dir("p1") == fts_root / "rag_corpora" / "p1"
    assert not (fts_root / "rag_corpora").exists()


def test_helper_rejects_traversal(fts_root):
    with pytest.raises(ValueError):
        paths.rag_corpus_dir("../x")


def test_rag_route_corpus_dir(fts_root):
    from finetune_studio.webui.routes import rag
    assert rag._corpus_dir("p1") == fts_root / "rag_corpora" / "p1"


def test_project_rag_corpus_dir(fts_root):
    from finetune_studio.webui.routes import project_rag
    assert project_rag.corpus_dir("p1") == fts_root / "rag_corpora" / "p1"


def test_rag_suite_default_corpus_path(fts_root):
    from finetune_studio.testing.rag_suite import default_corpus_path
    assert default_corpus_path("p1") == fts_root / "rag_corpora" / "p1"


def test_parsed_edit_corpus_sha12s_reads_fts_root(fts_root):
    from finetune_studio.data.parsed_edit import corpus_sha12s
    d = fts_root / "rag_corpora" / "p1"
    d.mkdir(parents=True)
    sha = "abcdef123456"
    (d / "manifest.json").write_text(json.dumps(
        {"extra": {"documents_meta": [{"source": f"/x/files/{sha}/parsed.txt"}]}}))
    assert sha in corpus_sha12s("p1")
