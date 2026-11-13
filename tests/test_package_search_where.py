"""A package says how it searches where it runs (walkthrough 2026-11-13 M10, m3).

A default package (no Include models) shipped the vectors without an embedder: at the user's site it ran keyword only,
while the studio's self-test, run with the studio's embedder, reported "exercised fusion, keyword, semantic search".
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)
from tests.test_package_scope import PASS, export_env  # noqa: F401  (fixture)


def test_runtime_search_names_the_mode() -> None:
    cfg = BrainExportConfig()
    assert cfg.runtime_search(bundled_embedder=False).startswith("word search only, until the host sets an embeddings")
    assert cfg.runtime_search(bundled_embedder=True) == "meaning + word search (the embedder is inside)"
    api = BrainExportConfig(embed_base_url="http://127.0.0.1:11434/v1/", embed_model="nomic-embed-text")
    assert api.runtime_search(bundled_embedder=False) == \
        "meaning + word search (embeddings from http://127.0.0.1:11434/v1)"
    assert api.runtime_config()["embed_base_url"] == "http://127.0.0.1:11434/v1"
    assert "embed_base_url" not in cfg.runtime_config()
    assert BrainExportConfig(semantic_search=False).runtime_search(bundled_embedder=False).startswith("word search")


def test_package_config_and_readme_are_honest(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=BrainExportConfig(kdf_log_n=10), passphrase=PASS)
    assert built.runtime_search.startswith("word search only")
    with tarfile.open(built.path) as tf:
        cfg = json.loads(tf.extractfile(next(n for n in tf.getnames() if n.endswith("/rag.config.json"))).read())
        readme = tf.extractfile(next(n for n in tf.getnames() if n.endswith("/README.md"))).read().decode()
    assert cfg["reranker"] is False                                  # no reranker inside: it said true (m3)
    assert "The original files are not in this package." in readme
    assert "the original documents are inside" not in readme


def test_route_and_form_say_where_it_runs(client, export_env) -> None:  # noqa: F811
    pid = export_env
    r = client.post(f"/api/projects/{pid}/brain/standalone", json={"passphrase": PASS, "semantic_search": False})
    assert r.status_code == 200 and r.json()["runtime_search"] == "word search only (meaning search is off)"
    page = client.get(f"/projects/{pid}/export").text
    for marker in ('id="mcp-search-where"', 'id="mcp-embed-url"', "embed_base_url: g('mcp-embed-url')",
                   "Where it runs: word search only", "Search where it runs: "):
        assert marker in page, marker


def test_a_word_search_package_is_not_told_it_needs_an_embedder(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    from finetune_studio.smart_brain.server_package import verify_package

    fake = tmp_path / "embedder"
    fake.mkdir()
    built = build_package(corpus_dir, tmp_path / "w.tar.gz", config=BrainExportConfig(kdf_log_n=10, semantic_search=False),
                          passphrase=PASS)
    out = verify_package(built.path, PASS, studio_models={"embedder": fake})
    assert out["ok"] is True, out
    assert "runtime_search" not in out and "where it runs" not in out["detail"]
