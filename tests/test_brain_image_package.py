"""Image search in an exported package: what ships (scope, encryption, originals), the server's /search/image,
/images, MCP rag_search_images, /ask with pictures (OpenAI / Claude image parts, text fallback), the self-test.

The CLIP encoder is replaced by the colour-histogram fake of ``tests.test_brain_image_layer`` (in the server module
too), so no model is downloaded; the bundled model folders are placeholders."""
from __future__ import annotations

import base64
import json
import tarfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.smart_brain import image_embeddings as ie
from finetune_studio.smart_brain import image_layer
from finetune_studio.smart_brain import standalone_server as srv
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package
from tests.test_brain_image_layer import COLOURS, FakeEmbedder, _png
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)


class _FakeEncoder(FakeEmbedder):
    def encode_texts(self, texts):
        return np.atleast_2d(super().encode_texts(list(texts)))


@pytest.fixture(autouse=True)
def fake_models(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(ie, "cached_image_embedder", lambda model_id=None, device="auto": FakeEmbedder())
    monkeypatch.setattr(srv, "open_image_encoder", lambda image_dir, text_dir, device: _FakeEncoder())
    image_dir, text_dir = tmp_path / "m" / "image", tmp_path / "m" / "text"
    for d in (image_dir, text_dir):
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
    monkeypatch.setattr(ie, "on_disk", lambda model_id=None: True)
    monkeypatch.setattr(ie, "model_dirs", lambda spec: (image_dir, text_dir))


def _with_images(corpus: Path) -> Path:
    """Pictures for c1 (red, a SnackGrid caption) and c3 (blue): c3 is outside a c1/c2 scope."""
    plan = image_layer.LayerPlan(fresh=[
        ie.ImageAsset(document_id="c1", filename="ledger.txt", source="ledger.txt", locator="image",
                      mime_type="image/png", width=160, height=160, data=_png(COLOURS["red"]),
                      context="SnackGrid fridge photo", file_id="F1", source_sha="S1"),
        ie.ImageAsset(document_id="c3", filename="emberfall.txt", source="emberfall.txt", locator="page 2",
                      mime_type="image/png", width=160, height=160, data=_png(COLOURS["blue"]),
                      description="a blue sign")], model_id=ie.DEFAULT_IMAGE_MODEL)
    result = image_layer.compute(plan)
    image_layer.write(corpus, result)
    m = json.loads((corpus / "manifest.json").read_text())
    m.setdefault("extra", {})["image_search"] = result.meta
    (corpus / "manifest.json").write_text(json.dumps(m))
    return corpus


def _export(corpus: Path, tmp: Path, *, encrypt: bool = False, **cfg) -> tuple[Path, Path, str | None]:
    conf = BrainExportConfig(semantic_search=False, include_models=True, include_reranker=False,
                             image_search=True, **cfg)
    res = build_package(corpus, tmp / "p.tar.gz", name="Img", encrypt=encrypt, config=conf,
                        passphrase="correct horse battery" if encrypt else None)
    with tarfile.open(res.path, "r:gz") as tf:
        names = tf.getnames()
        tf.extractall(tmp / "x", filter="data")
    (root,) = [p for p in (tmp / "x").iterdir() if p.is_dir()]
    (root / "_names.txt").write_text("\n".join(names))
    return root, res.path, res.passphrase


def _corpus(root: Path, **over) -> srv.Corpus:
    return srv.Corpus(root / "corpus", srv.resolve_config(over, {}, {})[0])


def test_plain_package_ships_vectors_thumbnails_and_the_model(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root, _archive, _ = _export(_with_images(corpus_dir), tmp_path)
    c = root / "corpus"
    records = json.loads((c / "image_assets.json").read_text())
    assert len(records) == 2 and np.load(c / "image_vectors.npy").shape == (2, 3)
    assert all("file_id" not in r and "source_sha" not in r for r in records)   # studio-only fields stay home
    assert all((c / r["thumbnail"]).is_file() for r in records)
    assert not any((c / r["path"]).exists() for r in records)                    # full pictures: only with originals
    assert (c / "image_model" / "image" / "config.json").is_file() and (c / "image_model" / "text").is_dir()
    manifest = json.loads((c / "manifest.json").read_text())
    assert manifest["search_layers"]["image"] is True and manifest["extra"]["image_search"]["assets"] == 2
    assert "Image search" in (root / "README.md").read_text()
    assert json.loads((root / "rag.config.json").read_text())["images"] is True


def test_full_pictures_ship_with_original_files(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root, _a, _ = _export(_with_images(corpus_dir), tmp_path, include_originals=True)
    records = json.loads((root / "corpus" / "image_assets.json").read_text())
    assert all((root / "corpus" / r["path"]).is_file() for r in records)


def test_scoped_package_never_carries_pictures_of_left_out_documents(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    corpus = _with_images(corpus_dir)
    res = build_package(corpus, tmp_path / "s.tar.gz", name="Img", encrypt=False, documents={"c1", "c2"},
                        config=BrainExportConfig(semantic_search=False, include_models=True, include_reranker=False,
                                                 image_search=True))
    with tarfile.open(res.path) as tf:
        tf.extractall(tmp_path / "s", filter="data")
        names = tf.getnames()
    (root,) = [p for p in (tmp_path / "s").iterdir() if p.is_dir()]
    records = json.loads((root / "corpus" / "image_assets.json").read_text())
    assert [r["document_id"] for r in records] == ["c1"]
    outside = image_layer.read_layer(corpus)[0][1]["asset_id"]
    assert not any(outside in n for n in names)
    hits = _corpus(root).search_images("a blue sign")["hits"]
    assert {h["document_id"] for h in hits} == {"c1"}


def test_encrypted_package_keeps_pictures_inside_the_container(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root, _a, pw = _export(_with_images(corpus_dir), tmp_path, encrypt=True)
    names = (root / "_names.txt").read_text()
    assert "image_assets" not in names and "image_vectors" not in names    # only corpus.enc + the public model
    assert "image_model/image/config.json" in names
    c = srv.Corpus(root / "corpus", srv.resolve_config({}, {}, {})[0], passphrase=pw or "correct horse battery")
    found = c.search_images("SnackGrid photo", thumbnails=True)
    assert found["hits"][0]["document_id"] == "c1" and found["hits"][0]["thumbnail"].startswith("data:image/jpeg")


def test_image_search_needs_models_and_an_image_layer(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    with pytest.raises(FileNotFoundError, match="turn on Image search"):
        build_package(corpus_dir, tmp_path / "a.tar.gz", encrypt=False,
                      config=BrainExportConfig(semantic_search=False, include_models=True, image_search=True))
    with pytest.raises(ValueError, match="needs Include models"):
        build_package(_with_images(corpus_dir), tmp_path / "b.tar.gz", encrypt=False,
                      config=BrainExportConfig(semantic_search=False, image_search=True))


def test_server_reports_the_layer_and_searches_by_text_and_picture(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root, _a, _ = _export(_with_images(corpus_dir), tmp_path)
    c = _corpus(root)
    assert c.layers()["image"].startswith("on (2 pictures") and c.info()["images"] == 2
    assert "images" in c.info()["mode"]
    by_words = c.search_images("show me the SnackGrid photo")
    assert by_words["hits"][0]["document_id"] == "c1" and by_words["hits"][0]["thumbnail_url"].endswith("/thumb")
    by_pic = c.search_images(image_bytes=_png(COLOURS["blue"], noise=4), documents=True)
    assert by_pic["mode"] == "image" and by_pic["hits"][0]["document_id"] == "c3"
    assert by_pic["documents"][0]["document_id"] == "c3"
    off = _corpus(root, images=False)
    assert off.layers()["image"] == "off"
    with pytest.raises(ValueError, match="turned off"):
        off.search_images("x")


def test_mcp_tool_returns_thumbnails_as_image_content(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root, _a, _ = _export(_with_images(corpus_dir), tmp_path)
    c = _corpus(root)
    respond = srv.mcp_responder(c)
    tools = {t["name"] for t in respond({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    assert "rag_search_images" in tools
    pic = base64.b64encode(_png(COLOURS["red"], noise=2)).decode()
    out = respond({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "rag_search_images", "arguments": {"image_base64": pic, "top_k": 2}}})["result"]
    assert out["isError"] is False
    text, *images = out["content"]
    assert json.loads(text["text"])["hits"][0]["document_id"] == "c1"
    assert images and images[0]["type"] == "image" and images[0]["mimeType"] == "image/jpeg"
    assert "rag_search_images" not in {t["name"] for t in srv.mcp_responder(_corpus(root, images=False))(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"})["result"]["tools"]}


def test_http_search_image_and_thumbnail_routes(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root, _a, _ = _export(_with_images(corpus_dir), tmp_path)
    server = srv._Server(("127.0.0.1", 0), srv.make_handler(_corpus(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        def post(path, body):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        d = post("/search/image", {"query": "SnackGrid", "thumbnails": True})
        assert d["hits"][0]["document_id"] == "c1" and d["hits"][0]["thumbnail"].startswith("data:")
        with urllib.request.urlopen(base + d["hits"][0]["thumbnail_url"], timeout=10) as r:
            assert r.headers["Content-Type"] == "image/jpeg" and r.read()[:3] == b"\xff\xd8\xff"
        d = post("/search", {"query": "Highmere", "images": True})
        assert "images" in d and d["hits"]
        with pytest.raises(urllib.error.HTTPError) as err:
            post("/search/image", {"image_base64": "not base64!!"})
        assert err.value.code == 400
    finally:
        server.shutdown()


def test_picture_parts_per_provider() -> None:
    pics = [(_png(COLOURS["red"]), "image/png", "x")]
    oa = srv.with_picture_parts("q", pics, "openai")
    assert oa[0] == {"type": "text", "text": "q"} and oa[1]["image_url"]["url"].startswith("data:image/png;base64,")
    an = srv.with_picture_parts("q", pics, "anthropic")
    assert an[1]["type"] == "image" and an[1]["source"]["media_type"] == "image/png"
    assert srv.picture_refused(RuntimeError("answer model at x said 400: image_url is not supported"))
    assert not srv.picture_refused(RuntimeError("answer model unavailable at x: timed out"))


def test_ask_sends_pictures_and_falls_back_to_text(corpus_dir: Path, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    root, _a, _ = _export(_with_images(corpus_dir), tmp_path)
    c = _corpus(root, answer=True, llm_base_url="http://127.0.0.1:9/v1", llm_model="m")
    seen: list = []

    def chat(messages, provider=None):
        seen.append(messages[-1]["content"])
        if isinstance(messages[-1]["content"], list) and refuse:
            raise RuntimeError("answer model at x said 400: this model does not support images")
        return "It is the red fridge [ledger.txt]."

    monkeypatch.setattr(c, "_chat", chat)
    refuse = False
    out = c.answer("Show me the SnackGrid photo", image_bytes=None)
    assert out["images_sent"] >= 1 and out["images"][0]["document_id"] == "c1"
    assert isinstance(seen[0], list) and seen[0][1]["type"] == "image_url"
    seen.clear()
    refuse = True
    out = c.answer("what is this?", image_bytes=_png(COLOURS["red"], noise=5))
    assert out["images_sent"] == 0 and "refused" in out["images_note"]
    assert isinstance(seen[0], list) and isinstance(seen[1], str) and "the picture the user attached" in seen[1]
    with pytest.raises(ValueError):
        c.answer("bad", image_bytes=b"no picture")


def test_chat_page_attaches_pictures_and_allows_data_images() -> None:
    for needle in ('id="picture"', "image_base64 = attached", "thumbnails: true", "pics(a, d.images)",
                   "/search/image"):
        assert needle in srv._CHAT_PAGE + srv._CHAT_JS, needle
    src = Path(srv.__file__).read_text()
    assert "img-src data: blob:" in src


def test_export_page_has_the_image_layer_switch() -> None:
    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/export_models.html").read_text()
    assert 'id="mcp-images"' in html and "image_search: g('mcp-images').checked && g('mcp-models').checked" in html


def test_server_file_still_ends_with_its_main_guard() -> None:
    src = Path(srv.__file__).read_text().rstrip()
    assert src.endswith('if __name__ == "__main__":\n    sys.exit(main())')
