"""The standalone Smart Brain server answers each HTTP request on its own thread: its bundled models must never run
(or load) on two threads at once.

Found by the Quillmoor benchmark (2026-10-29): 4 clients searching an exported package with a bundled embedder and
reranker crashed its server after 165 searches with "CUDA error: an illegal memory access was encountered".
"""
from __future__ import annotations

import sys
import tarfile
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.smart_brain import standalone_server as srv
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)


class _Overlap:
    """Counts how many threads are inside a model call at the same time."""

    def __init__(self) -> None:
        self.inside = 0
        self.max_inside = 0
        self.lock = threading.Lock()

    def enter(self) -> None:
        with self.lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        time.sleep(0.01)          # a GPU call takes a while: long enough for others to arrive
        with self.lock:
            self.inside -= 1


def _corpus(corpus_dir: Path, tmp: Path) -> srv.Corpus:  # noqa: F811
    out = build_package(corpus_dir, tmp / "p.tar.gz", name="T", encrypt=False, config=BrainExportConfig()).path
    with tarfile.open(out, "r:gz") as tf:
        tf.extractall(tmp / "x", filter="data")
    (root,) = [p for p in (tmp / "x").iterdir() if p.is_dir()]
    cfg = srv.resolve_config({"embed_base_url": "http://embed.invalid/v1"}, {}, {})[0]
    return srv.Corpus(root / "corpus", cfg)


def test_concurrent_searches_never_run_a_model_twice_at_once(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    c = _corpus(corpus_dir, tmp_path)
    ov = _Overlap()

    class Embedder:
        def encode(self, texts, normalize_embeddings=True):
            ov.enter()
            return np.array([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32)

    class Reranker:
        def predict(self, pairs):
            ov.enter()
            return [float(len(t)) for _q, t in pairs]

    c._local_embedder, c._reranker, c.rerank_enabled = Embedder(), Reranker(), True
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda i: c.search(f"salt oath {i}", 3), range(40)))
    assert all(results)
    assert ov.max_inside == 1


def test_concurrent_first_requests_load_a_bundled_model_once(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    c = _corpus(corpus_dir, tmp_path)
    loads: list[str] = []

    class SlowST:
        def __init__(self, path: str, device: str = "cpu") -> None:
            time.sleep(0.05)
            loads.append(path)

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=SlowST, CrossEncoder=SlowST))
    monkeypatch.setattr(c, "_looks_like_model", lambda d: True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        models = list(ex.map(lambda _i: c.local_embedder(), range(6)))
    assert len(loads) == 1
    assert all(m is models[0] for m in models)
