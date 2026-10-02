"""Encrypted studio bundle (.ftsrag): round trip, no plaintext at rest, fail-closed."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest

from finetune_studio.data.rag_portable import PortableRAG
from finetune_studio.data.rag_portable import secure_bundle as sb
from finetune_studio.data.rag_portable.rag_container import (
    ContainerWriter,
    WrongPassphraseOrTampered,
)
from tests.test_rag_import_bundle import (
    _stub_embedder,
    _write_minimal_corpus,
    _write_real_payloads,
)

PASS = "correct horse battery staple"


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    _write_minimal_corpus(src, with_local_models=True)
    _write_real_payloads(src)
    return src


def test_round_trip_and_no_plaintext(tmp_path: Path, corpus: Path, monkeypatch) -> None:
    _stub_embedder(monkeypatch)
    before = {p.name for p in Path(tempfile.gettempdir()).glob("fts-rag-*")}
    out, gen = sb.export_secure_bundle(corpus, tmp_path / "b.ftsrag", PASS, log_n=10)
    assert gen is None and sb.is_secure_bundle(out)
    raw = out.read_bytes()
    assert b"hello world" not in raw and b"import-test" not in raw
    assert b"doc1" not in raw and b"sources" not in raw
    assert {p.name for p in Path(tempfile.gettempdir()).glob("fts-rag-*")} == before

    dest = tmp_path / "dest"
    sb.import_secure_bundle(out, PASS, dest)
    assert not (dest / "embedder").exists()  # public weights are never bundled
    stats = PortableRAG(dest).load().manifest
    assert stats.documents == 1 and stats.chunks == 1
    assert (dest / "sources" / "doc1.txt").read_text().startswith("hello world")
    assert not (tmp_path / "dest.importing").exists()


def test_generated_passphrase_returned_once(tmp_path: Path, corpus: Path) -> None:
    out, gen = sb.export_secure_bundle(corpus, tmp_path / "g.ftsrag", None, log_n=10)
    assert gen and len(gen) >= 16
    sb.import_secure_bundle(out, gen, tmp_path / "d")


def test_short_passphrase_rejected(tmp_path: Path, corpus: Path) -> None:
    with pytest.raises(ValueError, match="too short"):
        sb.export_secure_bundle(corpus, tmp_path / "x.ftsrag", "short", log_n=10)
    assert not (tmp_path / "x.ftsrag").exists()


def test_wrong_passphrase_and_tamper_fail_closed(tmp_path: Path, corpus: Path) -> None:
    out, _ = sb.export_secure_bundle(corpus, tmp_path / "b.ftsrag", PASS, log_n=10)
    dest = tmp_path / "dest"
    with pytest.raises(WrongPassphraseOrTampered):
        sb.import_secure_bundle(out, "wrong passphrase!!", dest)
    assert not dest.exists()
    data = bytearray(out.read_bytes())
    data[len(data) // 2] ^= 0xFF
    bad = tmp_path / "bad.ftsrag"
    bad.write_bytes(bytes(data))
    with pytest.raises(WrongPassphraseOrTampered):
        sb.import_secure_bundle(bad, PASS, dest)
    assert not dest.exists() and not (tmp_path / "dest.importing").exists()


def test_unsafe_entry_name_rejected(tmp_path: Path) -> None:
    evil = tmp_path / "evil.ftsrag"
    with open(evil, "wb") as fp:
        cw = ContainerWriter(fp, PASS, log_n=10)
        cw.add("manifest.json", b"{}")
        cw.add("../pwned.txt", b"x")
        cw.close()
    with pytest.raises(ValueError, match="unsafe path"):
        sb.import_secure_bundle(evil, PASS, tmp_path / "dest")
    assert not (tmp_path / "pwned.txt").exists() and not (tmp_path / "dest").exists()


def test_refuses_existing_without_overwrite(tmp_path: Path, corpus: Path) -> None:
    out, _ = sb.export_secure_bundle(corpus, tmp_path / "b.ftsrag", PASS, log_n=10)
    with pytest.raises(FileExistsError):
        sb.import_secure_bundle(out, PASS, corpus)


def test_routes_export_download_import(tmp_path: Path, corpus: Path, monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import finetune_studio.webui.routes.rag as rag_routes
    from finetune_studio import db

    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", tmp_path)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", tmp_path / "projects")
    _stub_embedder(monkeypatch)
    app = FastAPI()
    app.include_router(rag_routes.router)
    client = TestClient(app)
    pid = db.create_project("secbundle", "", "", "")["id"]
    dest = tmp_path / "rag_corpora" / pid
    dest.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copytree(corpus, dest)

    r = client.post(f"/{pid}/rag/bundle", json={"name": "my bundle", "passphrase": PASS})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["encrypted"] and d["passphrase"] is None and d["filename"].endswith(".ftsrag")
    assert str(tmp_path / "projects" / pid) in str(
        rag_routes._bundle_dir(pid)) and (rag_routes._bundle_dir(pid) / d["filename"]).is_file()
    blob = client.get("/api" + d["download_url"].removeprefix("/api")
                      if False else d["download_url"].replace("/projects", "", 1)).content
    assert blob[:8] == b"FTSRAGE1" and b"hello world" not in blob

    assert client.get(f"/{pid}/rag/bundle/download?file=../x.ftsrag").status_code in (400, 404)
    up = {"file": ("b.ftsrag", io.BytesIO(blob), "application/octet-stream")}
    assert client.post(f"/{pid}/rag/import?overwrite=true", files=up,
                       data={"passphrase": "nope-nope-nope"}).status_code == 400
    up = {"file": ("b.ftsrag", io.BytesIO(blob), "application/octet-stream")}
    assert client.post(f"/{pid}/rag/import?overwrite=false", files=up,
                       data={"passphrase": PASS}).status_code == 409
    up = {"file": ("b.ftsrag", io.BytesIO(blob), "application/octet-stream")}
    ok = client.post(f"/{pid}/rag/import?overwrite=true", files=up, data={"passphrase": PASS})
    assert ok.status_code == 200, ok.text
    assert ok.json()["chunks"] == 1
    assert not list((tmp_path / "projects" / pid / "rag-import-staging").glob("*"))
