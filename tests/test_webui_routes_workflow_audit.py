"""Regression tests for discrepancies fixed during the B2 webui-routes-workflow audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.smart_brain.schema import EmbeddingModelInfo
from finetune_studio.smart_brain.store import BrainIndex


def _fake_get_embedder(name: str = "fake-deterministic", device: str = "cpu"):
    dim = 16

    def encode(texts: list[str] | str) -> np.ndarray:
        items = [texts] if isinstance(texts, str) else list(texts)
        out = np.zeros((len(items), dim), dtype=np.float32)
        for i, text in enumerate(items):
            out[i, hash((name, text)) % dim] = 1.0
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out[0] if isinstance(texts, str) else out

    info = EmbeddingModelInfo(
        name=name or "fake-deterministic", dim=dim, normalize=True,
        distance="cosine", cached_at="1970-01-01T00:00:00Z",
    )
    return encode, info


@pytest.fixture()
def patched_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "finetune_studio.smart_brain.store.get_embedder",
        _fake_get_embedder,
    )


def _sha_dir(root: Path, sha: str, original: str, body: str) -> Path:
    d = root / sha
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(
        json.dumps({"original_filename": original}), encoding="utf-8",
    )
    (d / "parsed.txt").write_text(body, encoding="utf-8")
    return d


def test_rebuild_vectors_applies_pending_embedder_from_settings_patch(
    tmp_path: Path, patched_embedder: None, client,
) -> None:
    """A re-embed must not silently drop an embedder change made via POST /brain/settings when the rebuild call
    omits the embedder.

    Before the fix: rebuild_vectors(embedder=None) fell back to the manifest's *old* embedding_model.name
    (store.py), so a settings-patched embedder was silently discarded on the very call the settings endpoint's
    own docstring says is required to apply it. The default now lives in ``BrainIndex.rebuild_vectors`` itself
    (``rag_settings.embedder`` first), so the CLI gets it too; the route runs it as a build job (2026-10-16).
    """
    from finetune_studio.data.fs.paths import brain_index_dir
    from tests.knowledge_helpers import wait_build

    pid = client.post("/api/projects", json={"name": "rebuild-embedder-test"}).json()["id"]
    src = tmp_path / "files"
    _sha_dir(src, "aaa111bbb222", "doc.txt", "some document body " * 20)
    rag = BrainIndex(brain_index_dir(pid))
    rag.build_from_directory(
        src, name="rebuild-embedder-test", embedder="embedder-a",
        chunk_size=80, overlap=10, extensions=[".txt"],
    )
    manifest_path = brain_index_dir(pid) / "manifest.json"
    before = json.loads(manifest_path.read_text())
    assert before["embedding_model"]["name"] == "embedder-a"
    assert before["rag_settings"]["embedder"] == "embedder-a"

    # Patch settings to request a different embedder — no rebuild yet.
    r = client.post(f"/api/projects/{pid}/brain/settings", json={"embedder": "embedder-b"})
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    mid = json.loads(manifest_path.read_text())
    assert mid["rag_settings"]["embedder"] == "embedder-b"
    # Vectors not yet rebuilt — embedding_model.name still the old one.
    assert mid["embedding_model"]["name"] == "embedder-a"

    # Caller rebuilds WITHOUT explicitly naming the embedder (as the settings
    # endpoint's docstring instructs: "caller must rebuild vectors afterwards").
    r = client.post(f"/api/projects/{pid}/brain/rebuild-vectors", json={})
    done = wait_build(client, pid, r)
    assert done["ok"] is True and done["mode"] == "re-embed", done
    assert done["embedder"] == "embedder-b"

    after = json.loads(manifest_path.read_text())
    assert after["embedding_model"]["name"] == "embedder-b"
    assert after["rag_settings"]["embedder"] == "embedder-b"

    # The store's own default: the settings' embedder wins over the vectors' one; an explicit name wins over both.
    assert rag.rebuild_vectors()["embedder"] == "embedder-b"
    assert rag.rebuild_vectors(embedder="embedder-c")["embedder"] == "embedder-c"
