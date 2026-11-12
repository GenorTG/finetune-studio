"""Opening a package made with models gives the studio index usable model refs (embedder AND reranker)."""
from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio.smart_brain import package_import


@pytest.fixture
def store(tmp_path: Path, monkeypatch):
    from finetune_studio.data import shared_models as sm

    present = {("embedder", "intfloat__multilingual-e5-large@b97e"), ("reranker", "BAAI__bge-reranker-v2-m3@d9e3")}

    def resolve(short_id: str, kind: str) -> Path:
        if (kind, short_id) not in present:
            raise FileNotFoundError(short_id)
        d = tmp_path / kind / short_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(sm, "resolve", resolve)


def test_bundled_embedder_and_reranker_get_shared_refs(store) -> None:
    manifest = {"embedding_model": {"name": "intfloat__multilingual-e5-large@b97e"},
                "rag_settings": {"embedder": "intfloat__multilingual-e5-large@b97e",
                                 "reranker": "BAAI__bge-reranker-v2-m3@d9e3"}}
    package_import._portable_embedder(manifest)
    assert manifest["embedding_model"]["name"] == "shared:embedder:intfloat__multilingual-e5-large@b97e"
    assert manifest["rag_settings"]["embedder"] == "shared:embedder:intfloat__multilingual-e5-large@b97e"
    assert manifest["rag_settings"]["reranker"] == "shared:reranker:BAAI__bge-reranker-v2-m3@d9e3"


def test_unknown_or_hub_refs_stay_as_they_are(store) -> None:
    manifest = {"rag_settings": {"embedder": "shared:embedder:x", "reranker": "BAAI/bge-reranker-v2-m3"}}
    package_import._portable_embedder(manifest)
    assert manifest["rag_settings"] == {"embedder": "shared:embedder:x", "reranker": "BAAI/bge-reranker-v2-m3"}
