"""A package's bundled reranker defaults to on with a GPU and off on a CPU-only host.

Measured 2026-11-16/17 (tests/corpus/tarrowfield/RESULTS.md): bge-reranker-v2-m3 scores ~1 passage per second on a
6-core CPU at 512 tokens; scoring the index's 50 candidates made a CPU-only search take 57-62 s against 0.13 s without
it. ``--reranker`` / ``RAG_RERANKER=on`` still turns it on, and then it scores only ``CPU_RERANK_TOP_N`` candidates.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from finetune_studio.smart_brain import standalone_server as srv
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import _layers_section, _readme, build_package
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)


def _root(corpus_dir: Path, tmp: Path) -> Path:  # noqa: F811
    out = build_package(corpus_dir, tmp / "p.tar.gz", name="T", encrypt=False, config=BrainExportConfig()).path
    with tarfile.open(out, "r:gz") as tf:
        tf.extractall(tmp / "x", filter="data")
    (root,) = [p for p in (tmp / "x").iterdir() if p.is_dir()]
    man = root / "corpus" / "manifest.json"
    data = json.loads(man.read_text())
    data.setdefault("rag_settings", {}).update(rerank_enabled=True, rerank_top_n=50)   # as if a reranker shipped
    man.write_text(json.dumps(data))
    return root


def _corpus(root: Path, monkeypatch: pytest.MonkeyPatch, device: str, flags: dict | None = None,
            env: dict | None = None, file_cfg: dict | None = None) -> srv.Corpus:
    monkeypatch.setattr(srv, "_resolve_device", lambda requested: device)
    cfg = srv.resolve_config({"embed_base_url": "http://embed.invalid/v1", **(flags or {})}, env or {},
                             file_cfg or {})[0]
    return srv.Corpus(root / "corpus", cfg)


def test_cpu_host_turns_the_reranker_off_by_default(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    c = _corpus(_root(corpus_dir, tmp_path), monkeypatch, "cpu")
    assert c.rerank_enabled is False and c.local_reranker() is None
    assert c.rerank_reason == "off: running on the CPU; pass --reranker (or RAG_RERANKER=on) to turn it on"
    assert c.layers()["reranker"] == "off (running on the CPU; pass --reranker (or RAG_RERANKER=on) to turn it on)"
    assert "rerank" not in c.info()["mode"]


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "xpu", "mps"])
def test_gpu_host_keeps_the_reranker_on_by_default(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                                   monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    c = _corpus(_root(corpus_dir, tmp_path), monkeypatch, device)
    assert c.rerank_enabled is True and c.rerank_reason == f"on: running on {device}"
    assert c.rerank_pool_size() == 50


@pytest.mark.parametrize(("device", "flags", "env", "file_cfg", "expected"), [
    ("cpu", {"reranker": True}, {}, {}, True),                     # --reranker
    ("cpu", {}, {"RAG_RERANKER": "on"}, {}, True),
    ("cpu", {}, {}, {"reranker": True}, True),                     # rag.config.json edited to true
    ("cuda", {"reranker": False}, {}, {}, False),                   # --no-reranker
    ("cuda", {}, {"RAG_RERANKER": "off"}, {}, False),
    ("cuda", {}, {}, {"reranker": "auto"}, True),                   # what a new export writes
    ("cpu", {}, {}, {"reranker": "auto"}, False),
    ("cpu", {"reranker": "auto"}, {"RAG_RERANKER": "on"}, {}, False),   # --reranker-auto beats the env
])
def test_explicit_settings_override_the_device_default(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                                       monkeypatch: pytest.MonkeyPatch, device: str, flags: dict,
                                                       env: dict, file_cfg: dict, expected: bool) -> None:
    c = _corpus(_root(corpus_dir, tmp_path), monkeypatch, device, flags, env, file_cfg)
    assert c.rerank_enabled is expected


def test_cli_flags_parse_to_the_tri_state() -> None:
    p = srv.build_parser()
    assert p.parse_args([]).reranker is None
    assert p.parse_args(["--reranker"]).reranker is True and p.parse_args(["--rerank"]).reranker is True
    assert p.parse_args(["--no-reranker"]).reranker is False and p.parse_args(["--no-rerank"]).reranker is False
    assert p.parse_args(["--reranker-auto"]).reranker == "auto"
    assert srv.DEFAULT_CONFIG["reranker"] == "auto"
    with pytest.raises(srv.ConfigError):
        srv.resolve_config({}, {"RAG_RERANKER": "maybe"}, {})


def test_explicit_reranker_on_cpu_scores_only_a_few_candidates(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    c = _corpus(_root(corpus_dir, tmp_path), monkeypatch, "cpu", {"reranker": True})
    seen: list[int] = []

    class Reranker:
        def predict(self, pairs):
            seen.append(len(pairs))
            return [float(len(t)) for _q, t in pairs]

    c._reranker, c._rerank_device = Reranker(), "cpu"
    c.max_per_document = 0
    c.chunks = c.chunks * 10                    # enough candidates to see the cap
    c.chunk_ids = [f"{x}-{i}" for i, x in enumerate(c.chunk_ids * 10)]
    c.idx_map = {cid: i for i, cid in enumerate(c.chunk_ids)}
    c.keyword, c.semantic = True, False
    monkeypatch.setattr(c, "bm25_scores", lambda q: __import__("numpy").ones(len(c.chunks)))
    hits = c.search("anything", 5)
    assert seen == [srv.CPU_RERANK_TOP_N] and srv.CPU_RERANK_TOP_N < 50
    assert all("ce_score" in h for h in hits)
    assert c.layers()["reranker"] == f"on (bundled cross-encoder, cpu, best {srv.CPU_RERANK_TOP_N} candidates)"


def test_export_ships_auto_and_says_so(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    cfg = BrainExportConfig()
    readme = _readme(title="T", slug="t", embed_model="e", encrypted=False, models=True, cfg=cfg,
                     bundled_embedder=True, bundled_reranker=True)
    assert "| reranker | `--reranker` / `--no-reranker` | `RAG_RERANKER` | `auto (on with a GPU, off on the CPU)` |" \
        in readme
    assert "On a CPU-only host it\nis off by default" in readme
    assert "off on a CPU-only host unless `--reranker`" in _layers_section(cfg, bundled_embedder=True,
                                                                             bundled_reranker=True)


def test_self_test_forces_a_bundled_reranker_on() -> None:
    import inspect

    from finetune_studio.smart_brain import server_package
    src = inspect.getsource(server_package.verify_package)
    assert 'env["RAG_RERANKER"] = "on"' in src
