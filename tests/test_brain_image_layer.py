"""Smart Brain image layer: extraction with provenance, incremental embedding, ranking, removal, Knowledge check.

The image/text model is replaced by a deterministic fake (colour histogram as the "visual" vector, a word->colour map
for text) so the logic runs without downloading CLIP; the real model is exercised live (RESULTS.md)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from finetune_studio.smart_brain import image_embeddings as ie
from finetune_studio.smart_brain import image_layer
from finetune_studio.smart_brain.standalone_server import (
    decode_image,
    image_documents,
    rank_images,
)

COLOURS = {"red": (220, 30, 30), "green": (30, 200, 40), "blue": (30, 40, 220)}


def _png(colour: tuple[int, int, int], size: int = 160, noise: int = 0) -> bytes:
    im = Image.new("RGB", (size, size), colour)
    if noise:
        arr = np.asarray(im).astype(np.int16) + np.random.RandomState(noise).randint(-12, 12, (size, size, 3))
        im = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _vec(rgb) -> np.ndarray:
    v = np.asarray(rgb, dtype=np.float32) + 1.0
    return v / np.linalg.norm(v)


class FakeEmbedder:
    dim = 3
    model_id = ie.DEFAULT_IMAGE_MODEL
    calls = 0

    def encode_images(self, images):
        FakeEmbedder.calls += len(images)
        return np.stack([_vec(np.asarray(im).reshape(-1, 3).mean(axis=0)) for im in images])

    def encode_texts(self, texts):
        single = isinstance(texts, str)
        rows = [_vec(next((c for w, c in COLOURS.items() if w in t.lower()), (128, 128, 128)))
                for t in ([texts] if single else texts)]
        return rows[0] if single else np.stack(rows)

    def metadata(self):
        return {"model": self.model_id, "dimension": 3}


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    FakeEmbedder.calls = 0
    monkeypatch.setattr(ie, "cached_image_embedder", lambda model_id=None, device="auto": FakeEmbedder())
    monkeypatch.setattr("finetune_studio.smart_brain.model_cache.cached_image_embedder",
                        lambda model_id=None, device="auto": FakeEmbedder())


def _asset(doc: str, colour: str, *, locator: str = "image", context: str = "", description: str = "",
           sha: str = "s1", noise: int = 0) -> ie.ImageAsset:
    return ie.ImageAsset(document_id=doc, filename=f"{doc}.png", source=f"internal/{doc}.png", locator=locator,
                         mime_type="image/png", width=160, height=160, data=_png(COLOURS[colour], noise=noise),
                         figure=1, context=context, description=description, source_sha=sha)


def _write(corpus: Path, plan: image_layer.LayerPlan) -> image_layer.LayerResult:
    result = image_layer.compute(plan)
    image_layer.write(corpus, result)
    (corpus / "manifest.json").write_text(json.dumps({"extra": {"image_search": result.meta}}), encoding="utf-8")
    return result


def test_layer_writes_vectors_thumbnails_and_provenance(tmp_path: Path) -> None:
    plan = image_layer.LayerPlan(fresh=[_asset("doc1", "red", locator="slide 2", context="SnackGrid fridge"),
                                        _asset("doc2", "blue")], model_id=ie.DEFAULT_IMAGE_MODEL)
    result = _write(tmp_path, plan)
    records, vectors, meta = image_layer.read_layer(tmp_path)
    assert vectors.shape == (2, 3) and len(records) == 2 and meta["assets"] == 2 and meta["embedded"] == 2
    rec = records[0]
    assert rec["locator"] == "slide 2" and rec["document_id"] == "doc1" and rec["context"] == "SnackGrid fridge"
    assert (tmp_path / rec["path"]).read_bytes() == plan.fresh[0].data
    thumb = Image.open(tmp_path / rec["thumbnail"])
    assert thumb.format == "JPEG" and max(thumb.size) <= ie.THUMB_SIDE
    assert result.meta["failed_count"] == 0


def test_rebuild_reuses_unchanged_pictures_and_embeds_only_new_ones(tmp_path: Path) -> None:
    first = [_asset("doc1", "red"), _asset("doc2", "blue")]
    _write(tmp_path, image_layer.LayerPlan(fresh=first, model_id=ie.DEFAULT_IMAGE_MODEL))
    assert FakeEmbedder.calls == 2
    previous = image_layer.previous_by_source(tmp_path, ie.DEFAULT_IMAGE_MODEL)
    assert set(previous) == {"doc1", "doc2"} and previous["doc1"][0] == "s1"
    # doc1 unchanged (kept without extraction), doc2 re-extracted with the same bytes, doc3 new
    plan = image_layer.LayerPlan(kept=previous["doc1"][1], fresh=[_asset("doc2", "blue", sha="s2"),
                                                                 _asset("doc3", "green")],
                                 previous_dir=tmp_path, model_id=ie.DEFAULT_IMAGE_MODEL)
    FakeEmbedder.calls = 0
    result = _write(tmp_path, plan)
    assert FakeEmbedder.calls == 1                      # only the green picture was embedded
    assert result.meta["embedded"] == 1 and result.meta["reused"] == 2 and result.meta["assets"] == 3
    records, _vectors, _ = image_layer.read_layer(tmp_path)
    assert [r["document_id"] for r in records] == ["doc1", "doc2", "doc3"]
    assert all((tmp_path / r["path"]).is_file() and (tmp_path / r["thumbnail"]).is_file() for r in records)


def test_another_model_ignores_the_previous_layer(tmp_path: Path) -> None:
    _write(tmp_path, image_layer.LayerPlan(fresh=[_asset("doc1", "red")], model_id=ie.DEFAULT_IMAGE_MODEL))
    assert image_layer.previous_by_source(tmp_path, "clip-vit-b32") == {}


def test_broken_picture_is_a_failure_not_a_crash(tmp_path: Path) -> None:
    bad = ie.ImageAsset(document_id="doc9", filename="x.png", source="x.png", locator="image", mime_type="image/png",
                        width=1, height=1, data=b"\x89PNG\r\n\x1a\nbroken")
    result = _write(tmp_path, image_layer.LayerPlan(fresh=[_asset("doc1", "red"), bad],
                                                    model_id=ie.DEFAULT_IMAGE_MODEL))
    assert result.meta["assets"] == 1 and result.meta["failed_count"] == 1
    assert result.meta["failed"][0]["document_id"] == "doc9"


def test_text_query_fuses_look_and_words_and_flags_weak() -> None:
    records = [{"asset_id": "a", "document_id": "d1", "filename": "IMG_1.jpg", "context": "", "description": ""},
               {"asset_id": "b", "document_id": "d2", "filename": "catalogue.docx",
                "context": "SnackGrid SG-6 snack fridge", "description": ""}]
    vectors = np.stack([_vec(COLOURS["red"]), _vec(COLOURS["blue"])])
    emb = FakeEmbedder()
    hits, weak = rank_images(records, vectors, emb.encode_texts("red"), query="red", top_k=2)
    assert hits[0]["asset_id"] == "a" and hits[0]["match"] == "looks" and not weak
    hits, weak = rank_images(records, vectors, emb.encode_texts("show me the SnackGrid photo"),
                             query="show me the SnackGrid photo", top_k=2)
    assert hits[0]["asset_id"] == "b" and "words" in hits[0]["match"] and hits[0]["words_score"] > 0
    # nothing looks like grey and no words match: weak
    far = np.stack([_vec((255, 0, 0)), _vec((0, 0, 255))])
    _hits, weak = rank_images(records, far, np.asarray([0.0, 1.0, 0.0], np.float32), query="zebra", top_k=2)
    assert weak


def test_image_query_and_document_grouping() -> None:
    records = [{"asset_id": "a", "document_id": "d1", "filename": "f1.pdf", "locator": "page 2"},
               {"asset_id": "b", "document_id": "d1", "filename": "f1.pdf", "locator": "page 5"},
               {"asset_id": "c", "document_id": "d2", "filename": "f2.png", "locator": "image"}]
    vectors = np.stack([_vec(COLOURS["red"]), _vec((200, 40, 40)), _vec(COLOURS["blue"])])
    q = FakeEmbedder().encode_images([decode_image(_png(COLOURS["red"], noise=3))])[0]
    hits, weak = rank_images(records, vectors, q, by_image=True, top_k=3)
    assert [h["asset_id"] for h in hits][:2] == ["a", "b"] and not weak and hits[0]["words_score"] == 0
    docs = image_documents(hits)
    assert docs[0]["document_id"] == "d1" and [i["locator"] for i in docs[0]["images"]] == ["page 2", "page 5"]
    # keep= filters records (a package's scope)
    hits, _ = rank_images(records, vectors, q, by_image=True, top_k=3, keep=lambda r: r["document_id"] == "d2")
    assert [h["asset_id"] for h in hits] == ["c"]


def test_decode_image_flattens_transparency_and_rejects_garbage() -> None:
    im = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    assert decode_image(buf.getvalue()).getpixel((5, 5)) == (255, 255, 255)
    with pytest.raises(ValueError):
        decode_image(b"not an image")


def test_remove_document_drops_its_pictures(tmp_path: Path) -> None:
    _write(tmp_path, image_layer.LayerPlan(fresh=[_asset("doc1", "red"), _asset("doc2", "blue")],
                                           model_id=ie.DEFAULT_IMAGE_MODEL))
    gone = image_layer.read_layer(tmp_path)[0][0]
    assert image_layer.remove_document(tmp_path, "doc1")
    records, vectors, meta = image_layer.read_layer(tmp_path)
    assert [r["document_id"] for r in records] == ["doc2"] and vectors.shape == (1, 3) and meta["assets"] == 1
    assert not (tmp_path / gone["path"]).exists() and not (tmp_path / gone["thumbnail"]).exists()
    assert image_layer.remove_document(tmp_path, "doc2")
    assert image_layer.read_layer(tmp_path)[1] is None and not (tmp_path / "image_assets").exists()


def test_subset_keeps_only_scoped_documents() -> None:
    records = [{"document_id": "a"}, {"document_id": "b~facts"}, {"document_id": "c"}]
    kept, vecs = image_layer.subset(records, np.eye(3, dtype=np.float32), lambda d: d in {"b", "c"})
    assert [r["document_id"] for r in kept] == ["b~facts", "c"] and vecs.shape == (2, 3)


def test_model_registry_is_pinned_and_open_licence() -> None:
    for spec in ie.IMAGE_MODELS.values():
        assert len(spec.image_revision) == 40 and (not spec.text_repo or len(spec.text_revision) == 40)
        assert "MIT" in spec.licence or "Apache" in spec.licence
    assert ie.spec_for("").id == ie.DEFAULT_IMAGE_MODEL
    with pytest.raises(ValueError):
        ie.spec_for("nope")


def test_search_without_model_on_disk_says_build_first(monkeypatch) -> None:
    monkeypatch.setattr(ie, "on_disk", lambda model_id=None: False)
    with pytest.raises(FileNotFoundError, match="build the Smart Brain with Image search on"):
        ie.load_image_embedder()


def test_shared_store_copies_only_the_pinned_files(tmp_path: Path, monkeypatch) -> None:
    """A Hub snapshot folder holds everything ever fetched for the revision; the store keeps only the patterns."""
    from finetune_studio.data import shared_models as sm

    snap = tmp_path / "snap"
    for rel in ("config.json", "model.safetensors", "pytorch_model.bin", "onnx/model.onnx", "1_Pooling/config.json",
                "2_Dense/model.safetensors", "openvino/openvino_model.bin"):
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_bytes(b"x")
    monkeypatch.setattr(sm, "IMAGE_MODELS", tmp_path / "store")
    monkeypatch.setattr(sm, "hf_cache_dir", lambda: tmp_path / "hf")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(snap))
    out = sm.register_snapshot("org/m", "a" * 40, ie._SNAPSHOT_FILES)
    kept = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert kept == ["1_Pooling/config.json", "2_Dense/model.safetensors", "META.json", "config.json",
                    "model.safetensors"]
