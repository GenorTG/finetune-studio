"""The reranker a new Smart Brain index gets depends on the machine.

Quillmoor S, same hybrid candidates, one client on an RTX 3090 (tests/corpus/quillmoor/RESULTS.md §5):
bge-reranker-v2-m3 found the answer in the top 5 for 89.2 % of questions vs 78.6 % for MiniLM (reverse look-ups
40/40 vs 24/40); fp16 kept the same 89.2 % at 190 ms p50 instead of 731 ms in fp32. On a CPU it would read 50
passages with ~2 GB of weights per search, so a CPU-only machine keeps MiniLM.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from finetune_studio.smart_brain import rerankers
from finetune_studio.smart_brain.constants import CPU_RERANKER, DEFAULT_RERANKER


@pytest.mark.parametrize("device, expected", [
    ("cuda", DEFAULT_RERANKER), ("cuda:0", DEFAULT_RERANKER), ("xpu", DEFAULT_RERANKER), ("cpu", CPU_RERANKER),
])
def test_default_reranker_follows_the_device(device: str, expected: str) -> None:
    with patch("finetune_studio.smart_brain.devices.resolve_device", return_value=device):
        assert rerankers.default_reranker() == expected
    assert DEFAULT_RERANKER == "BAAI/bge-reranker-v2-m3"
    assert CPU_RERANKER == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_half_precision_only_on_cuda() -> None:
    import torch

    assert rerankers.half_precision_kwargs("cuda:0") == {"model_kwargs": {"torch_dtype": torch.float16}}
    assert rerankers.half_precision_kwargs("cpu") == {}
    assert rerankers.half_precision_kwargs("mps") == {}


def test_get_reranker_loads_fp16_on_a_gpu() -> None:
    import torch

    with patch("sentence_transformers.CrossEncoder") as ce, \
         patch("finetune_studio.smart_brain.devices.resolve_device", return_value="cuda:0"), \
         patch("finetune_studio.smart_brain.shared_refs.resolve_model_ref", side_effect=lambda n, _k: n):
        rerankers.get_reranker("BAAI/bge-reranker-v2-m3")
    assert ce.call_args.kwargs["model_kwargs"] == {"torch_dtype": torch.float16}
    assert ce.call_args.kwargs["max_length"] == 512


def test_package_server_loads_its_reranker_fp16_on_a_gpu() -> None:
    import inspect

    from finetune_studio.smart_brain import standalone_server as srv

    text = inspect.getsource(srv.Corpus._load_reranker)   # server.py ships alone: it cannot import rerankers.py
    assert '"max_length": 512' in text and "torch.float16" in text and 'startswith("cuda")' in text


def test_a_new_index_gets_the_device_default(tmp_path) -> None:
    import numpy as np

    from finetune_studio.smart_brain import BrainIndex
    from finetune_studio.smart_brain.schema import EmbeddingModelInfo

    def fake_embedder(name: str = "fake", device: str = "cpu"):
        def encode(texts):
            n = 1 if isinstance(texts, str) else len(texts)
            v = np.full((n, 4), 0.5, dtype=np.float32)
            return v[0] if isinstance(texts, str) else v
        return encode, EmbeddingModelInfo(name=name, dim=4, normalize=True, distance="cosine", cached_at="x")

    src = tmp_path / "docs"
    src.mkdir()
    (src / "a.txt").write_text("The pump PMP-4471-B is serviced every 6,500 operating hours.")
    for device, expected in (("cpu", CPU_RERANKER), ("cuda", DEFAULT_RERANKER)):
        corpus = tmp_path / device
        with patch("finetune_studio.smart_brain.store.get_embedder", side_effect=fake_embedder), \
             patch("finetune_studio.smart_brain.store.register_shared_models"), \
             patch("finetune_studio.smart_brain.devices.resolve_device", return_value=device):
            BrainIndex(corpus).build_from_directory(src, name="t", embedder="org/e", chunk_size=80, overlap=10,
                                                    extensions=[".txt"])
        assert BrainIndex(corpus).read_manifest().rag_settings.reranker == expected


def test_settings_form_shows_the_default_before_any_index(client) -> None:
    pid = client.post("/api/projects", json={"name": "reranker default"}).json()["id"]
    with patch("finetune_studio.smart_brain.devices.resolve_device", return_value="cpu"):
        r = client.get(f"/api/projects/{pid}/brain/settings")
    assert r.status_code == 200
    assert r.json() == {"exists": False, "default_reranker": CPU_RERANKER}


def test_brain_page_offers_both_and_applies_the_default() -> None:
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/brain.html").read_text()
    select = html.split('id="s-reranker"', 1)[1].split("</select>", 1)[0]
    assert 'value="BAAI/bge-reranker-v2-m3" selected' in select and "ms-marco-MiniLM-L-6-v2" in select
    assert "if (!s.exists && s.default_reranker) s.reranker = s.default_reranker;" in html
