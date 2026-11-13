"""Image search through the studio: extraction from real office files, settings, search routes, asset serving, lint.

The CLIP model is faked (``tests.test_brain_image_layer.FakeEmbedder``); these pin the plumbing and the UI hooks."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.smart_brain import image_embeddings as ie
from finetune_studio.smart_brain import image_layer
from tests.test_brain_image_layer import COLOURS, FakeEmbedder, _png
from tests.test_project_brain import _project

TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates" / "brain.html"


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr("finetune_studio.smart_brain.model_cache.cached_image_embedder",
                        lambda model_id=None, device="auto": FakeEmbedder())
    monkeypatch.setattr(ie, "cached_image_embedder", lambda model_id=None, device="auto": FakeEmbedder())


@dataclass
class _Src:
    id: str
    file_id: str
    filename: str
    raw_path: str
    text_sha: str = "t1"
    ready: bool = True


def _docx(path: Path) -> Path:
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_paragraph("Product catalogue 2025")
    doc.add_paragraph("The SnackGrid SG-6 snack fridge, lease only.")
    doc.add_picture(io.BytesIO(_png(COLOURS["red"], 200, noise=7)), width=Inches(2))
    doc.add_paragraph("Card payments carry a 1.4% fee.")
    doc.save(path)
    return path


def _pptx(path: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Quayside leak, AQ-12"
    slide.shapes.add_picture(io.BytesIO(_png(COLOURS["blue"], 200, noise=8)), Inches(1), Inches(1.5), width=Inches(3))
    prs.save(path)
    return path


def _xlsx(path: Path) -> Path:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XlImage

    wb = Workbook()
    ws = wb.active
    ws.title = "Assets"
    ws["A1"] = "Asset tag"
    img = XlImage(io.BytesIO(_png(COLOURS["green"], 200, noise=9)))
    ws.add_image(img, "C3")
    wb.save(path)
    return path


def test_image_plan_extracts_pictures_with_provenance_context_and_reading(tmp_path: Path, monkeypatch) -> None:
    from finetune_studio.knowledge import index

    files = {"d1": _docx(tmp_path / "catalogue.docx"), "d2": _pptx(tmp_path / "ops.pptx"),
             "d3": _xlsx(tmp_path / "register.xlsx"), "d4": tmp_path / "IMG_0001.png"}
    files["d4"].write_bytes(_png(COLOURS["red"], 300))
    srcs = [_Src(k, f"f{k}", v.name, str(v)) for k, v in files.items()]
    monkeypatch.setattr(index.sources, "list_sources", lambda pid: srcs)
    readings = {"d4": "[Figure 1 · image · picture · read by Vision reader]\nA red snack fridge\n[End of figure 1]"}
    monkeypatch.setattr(index.sources, "parsed_text", lambda pid, s: readings.get(s.id, ""))
    docs = [type("D", (), {"document_id": k, "filename": v.name, "path": f"public/{v.name}"})() for k, v in files.items()]
    plan = index.image_plan("p", docs, model_id=ie.DEFAULT_IMAGE_MODEL)
    by_doc = {a.document_id: a for a in plan.fresh}
    assert set(by_doc) == {"d1", "d2", "d3", "d4"} and not plan.failed
    assert by_doc["d1"].locator.startswith("paragraph") and "SnackGrid" in by_doc["d1"].context
    assert by_doc["d2"].locator == "slide 1" and "Quayside" in by_doc["d2"].context
    assert by_doc["d3"].locator == "sheet Assets C3"
    assert by_doc["d4"].description == "A red snack fridge" and by_doc["d4"].read_by == "Vision reader"
    assert by_doc["d4"].source == "public/IMG_0001.png" and by_doc["d4"].file_id == "fd4"


def test_unchanged_source_is_not_reopened(tmp_path: Path, monkeypatch) -> None:
    from finetune_studio.knowledge import index

    png = tmp_path / "IMG_2.png"
    png.write_bytes(_png(COLOURS["blue"], 200))
    srcs = [_Src("d1", "f1", png.name, str(png))]
    monkeypatch.setattr(index.sources, "list_sources", lambda pid: srcs)
    monkeypatch.setattr(index.sources, "parsed_text", lambda pid, s: "")
    docs = [type("D", (), {"document_id": "d1", "filename": png.name, "path": png.name})()]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    result = image_layer.compute(index.image_plan("p", docs, model_id=ie.DEFAULT_IMAGE_MODEL))
    image_layer.write(corpus, result)
    (corpus / "manifest.json").write_text(json.dumps({"extra": {"image_search": result.meta}}))
    monkeypatch.setattr(index.figures if hasattr(index, "figures") else __import__(
        "finetune_studio.data.figures", fromlist=["x"]), "extract", lambda *a, **k: pytest.fail("re-extracted"))
    again = index.image_plan("p", docs, previous_dir=corpus, model_id=ie.DEFAULT_IMAGE_MODEL)
    assert not again.fresh and len(again.kept) == 1


def _corpus_with_images(root: Path) -> Path:
    from tests.test_project_brain import _write_fake_corpus

    _write_fake_corpus(root)
    np.save(root / "vectors.npy", np.zeros((2, 8), np.float32))
    (root / "vectors.idx.json").write_text(json.dumps({"abc123def456_0": 0, "abc123def456_1": 1}))
    (root / "bm25.json").write_text("{}")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.setdefault("rag_settings", {})["image_search"] = True
    plan = image_layer.LayerPlan(fresh=[
        ie.ImageAsset(document_id="abc123def456", filename="manual_notes.txt", source="manual_notes.txt",
                      locator="image", mime_type="image/png", width=160, height=160, data=_png(COLOURS["red"]),
                      context="red SnackGrid fridge", file_id="fid1")], model_id=ie.DEFAULT_IMAGE_MODEL)
    result = image_layer.compute(plan)
    image_layer.write(root, result)
    manifest.setdefault("extra", {})["image_search"] = result.meta
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def indexed(client, tmp_path: Path, monkeypatch):
    pid = _project(client)
    corpus = _corpus_with_images(tmp_path / "corpus")
    monkeypatch.setattr("finetune_studio.webui.routes.brain._corpus_dir", lambda p: corpus)
    return pid, corpus


def test_text_and_image_search_routes(client, indexed) -> None:
    pid, _corpus = indexed
    r = client.post(f"/api/projects/{pid}/brain/images/search", json={"query": "SnackGrid photo", "documents": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["layer"] and d["count"] == 1 and d["hits"][0]["match"].endswith("words")
    assert d["documents"][0]["document_id"] == "abc123def456"
    r = client.post(f"/api/projects/{pid}/brain/images/search-by-image",
                    files={"image": ("q.png", _png(COLOURS["red"], noise=2), "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["mode"] == "image" and d["hits"][0]["visual_score"] > 0.99 and d["documents"]
    r = client.post(f"/api/projects/{pid}/brain/images/search-by-image",
                    files={"image": ("q.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_asset_route_serves_original_thumbnail_and_download(client, indexed) -> None:
    pid, corpus = indexed
    rec = image_layer.read_layer(corpus)[0][0]
    full = client.get(f"/api/projects/{pid}/brain/images/{rec['asset_id']}")
    assert full.status_code == 200 and full.content == (corpus / rec["path"]).read_bytes()
    thumb = client.get(f"/api/projects/{pid}/brain/images/{rec['asset_id']}?thumb=1")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    dl = client.get(f"/api/projects/{pid}/brain/images/{rec['asset_id']}?download=1")
    assert "attachment" in dl.headers.get("content-disposition", "")
    assert client.get(f"/api/projects/{pid}/brain/images/nope").status_code == 404


def test_settings_carry_the_image_layer(client, indexed) -> None:
    pid, corpus = indexed
    s = client.get(f"/api/projects/{pid}/brain/settings").json()
    assert s["image_search"] is True and s["image_layer"]["assets"] == 1
    assert {m["id"] for m in s["image_models"]} == set(ie.IMAGE_MODELS)
    r = client.post(f"/api/projects/{pid}/brain/settings", json={"image_search": False, "image_model": "clip-vit-b32"})
    assert r.status_code == 200
    m = json.loads((corpus / "manifest.json").read_text())["rag_settings"]
    assert m["image_search"] is False and m["image_model"] == "clip-vit-b32"
    bad = client.post(f"/api/projects/{pid}/brain/settings", json={"image_model": "nope"})
    assert bad.status_code == 422


def test_search_with_images_adds_a_separate_list(client, indexed, monkeypatch) -> None:
    pid, _corpus = indexed

    class _Q:
        def search(self, *a, **k):
            return []

        def search_images(self, query, top_k=6, **k):
            return {"hits": [{"asset_id": "x", "score": 1.0, "visual_score": 0.3, "words_score": 0.0,
                              "match": "looks"}], "weak": False}

    monkeypatch.setattr("finetune_studio.smart_brain.BrainIndex.load", lambda self, **k: _Q())
    d = client.post(f"/api/projects/{pid}/brain/search", json={"query": "red fridge", "images": True}).json()
    assert d["images"][0]["asset_id"] == "x" and d["images_weak"] is False
    assert "images" not in client.post(f"/api/projects/{pid}/brain/search", json={"query": "red fridge"}).json()


def test_lint_reports_failed_and_undescribed_pictures(monkeypatch, tmp_path: Path) -> None:
    from finetune_studio.knowledge import lint

    corpus = tmp_path / "c"
    corpus.mkdir()
    plan = image_layer.LayerPlan(fresh=[ie.ImageAsset(
        document_id="s1", filename="IMG.png", source="IMG.png", locator="image", mime_type="image/png", width=1,
        height=1, data=_png(COLOURS["blue"]))], failed=[{"document_id": "s2", "filename": "x.docx",
                                                        "locator": "paragraph 3", "error": "EMF"}],
        model_id=ie.DEFAULT_IMAGE_MODEL)
    result = image_layer.compute(plan)
    image_layer.write(corpus, result)
    (corpus / "manifest.json").write_text(json.dumps({"extra": {"image_search": result.meta}}))
    monkeypatch.setattr("finetune_studio.data.fs.paths.brain_index_dir", lambda pid: corpus)
    found = {f.rule: f for f in lint._check_images("p", [])}
    assert found["image_failed"].severity == "warning" and "EMF" in found["image_failed"].detail[0]
    assert found["image_undescribed"].severity == "info" and "IMG.png" in found["image_undescribed"].detail[0]


def test_page_wires_the_controls_to_the_api() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    for needle in ('id="s-image-search"', 'id="s-image-model"', 'id="q-images"', "body.image_search",
                   "images: withImages", "'/brain/images/search'", "'/brain/images/search-by-image'",
                   "imageCards(d.images)", "?thumb=1", "?download=1", "image_extract"):
        assert needle in html, needle


def test_unknown_image_model_in_a_build_request_is_refused(client, indexed) -> None:
    pid, _corpus = indexed
    r = client.post(f"/api/projects/{pid}/brain/build", json={"image_model": "nope"})
    assert r.status_code == 422


def test_vectors_and_records_stay_aligned_after_remove(indexed) -> None:
    _pid, corpus = indexed
    from finetune_studio.smart_brain import BrainIndex

    BrainIndex(corpus).remove_source("abc123def456")
    records, vectors, _ = image_layer.read_layer(corpus)
    assert vectors is None or (len(records) == vectors.shape[0] == 0) or np.all(vectors.shape[0] == len(records))
