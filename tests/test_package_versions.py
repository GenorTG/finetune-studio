"""Versioned packages: a stable package id, auto-bumped versions with a change report, in-place updates of a running
copy (update.sh keeps rag.config.json and the keyfile), and opening a package (encrypted, with originals) to edit."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from finetune_studio.smart_brain import package_meta as pm
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package
from tests.test_brain_encrypted_package import _load_server
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)
from tests.test_package_originals import files  # noqa: F401  (fixture)

PASS = "correct horse battery staple"
FAST = BrainExportConfig(kdf_log_n=10, semantic_search=False)
DOCS = [{"document_id": "c1", "filename": "ledger.txt", "path": "public/ledger.txt", "sha256": "a" * 64, "size": 10},
        {"document_id": "c2", "filename": "highmere.txt", "path": "public/highmere.txt", "sha256": "b" * 64, "size": 20},
        {"document_id": "c3", "filename": "emberfall.txt", "path": "internal/emberfall.txt", "sha256": "c" * 64, "size": 30}]


def test_versions_and_change_report(tmp_path: Path) -> None:
    assert pm.bump_minor(None) == "1.0.0" and pm.bump_minor("1.4.2") == "1.5.0" and pm.bump_minor("2") == "2.1.0"
    with pytest.raises(ValueError):
        pm.valid_version("one")
    first = pm.make_record(base=None, documents=DOCS[:2])
    assert first["version"] == "1.0.0" and first["parent_version"] is None and len(first["package_id"]) == 16
    new_docs = [{**DOCS[0], "document_id": "c9", "sha256": "d" * 64}, DOCS[2]]       # c1 edited, c2 removed, c3 added
    second = pm.make_record(base=first, documents=new_docs)
    assert (second["package_id"], second["version"], second["parent_version"]) == (first["package_id"], "1.1.0", "1.0.0")
    assert pm.make_record(base=first, documents=new_docs, version="v2.0.0")["version"] == "2.0.0"
    with pytest.raises(ValueError, match="not newer"):
        pm.make_record(base=first, documents=new_docs, version="0.9")
    diff = pm.changes(first["documents"], new_docs)
    assert pm.summary(diff) == {"added": ["internal/emberfall.txt"], "changed": ["public/ledger.txt"],
                                "removed": ["public/highmere.txt"]}
    md = pm.changes_markdown(second, diff)
    assert "Since version 1.0.0: 1 added, 1 changed, 1 removed" in md and "- public/highmere.txt" in md
    hidden = pm.changes_markdown(second, diff, names=False)
    assert "1 added" in hidden and "ledger" not in hidden and "emberfall" not in hidden
    assert "First version: 2 document(s)" in pm.changes_markdown(first, pm.changes(None, DOCS[:2]))
    pm.add_history(tmp_path, {**first, "origin": "exported", "name": "Pack"})
    pm.add_history(tmp_path, {**second, "origin": "exported", "name": "Pack"})
    assert pm.latest(tmp_path, first["package_id"])["version"] == "1.1.0"
    listed = pm.packages(tmp_path)
    assert listed[0]["latest"] == "1.1.0" and [v["version"] for v in listed[0]["versions"]] == ["1.1.0", "1.0.0"]


def _build(corpus: Path, out: Path, record: dict, base_docs=None, **kw):
    diff = pm.changes(base_docs, record["documents"])
    record = {**record, "changes": pm.summary(diff)}
    return build_package(corpus, out, config=kw.pop("config", FAST), passphrase=PASS, package=record,
                         changes_md=pm.changes_markdown(record, diff),
                         changes_md_public=pm.changes_markdown(record, diff, names=False), **kw)


def _extract(archive: Path, dest: Path) -> Path:
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")
    return next(dest.rglob("server.py")).parent


def test_the_package_carries_its_version_everywhere(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    record = pm.make_record(base=None, documents=DOCS)
    root = _extract(_build(corpus_dir, tmp_path / "p.tar.gz", record).path, tmp_path / "x")
    public = json.loads((root / "package.json").read_text())
    assert public == {"format": 1, "package_id": record["package_id"], "version": "1.0.0", "parent_version": None,
                      "created": record["created"], "documents": 3}
    assert "ledger" not in (root / "CHANGES.md").read_text() and "ledger" not in (root / "README.md").read_text()
    assert "version **1.0.0**" in (root / "README.md").read_text() and "update.sh" in (root / "README.md").read_text()
    srv = _load_server()
    corpus = srv.Corpus(root / "corpus", srv.resolve_config({}, {}, {})[0], PASS)
    info = corpus.info()
    assert info["package"]["version"] == "1.0.0" and [d["path"] for d in info["package"]["documents"]] == \
        ["public/ledger.txt", "public/highmere.txt", "internal/emberfall.txt"]
    assert "First version" in corpus._container.read("CHANGES.md").decode()
    plain = _extract(build_package(corpus_dir, tmp_path / "q.tar.gz", config=FAST, encrypt=False, package=record,
                                   changes_md="# full").path, tmp_path / "y")
    assert "| public/ledger.txt |" in (plain / "README.md").read_text() and (plain / "CHANGES.md").read_text() == "# full"


def _fake_venv(root: Path) -> None:
    (root / ".venv" / "bin").mkdir(parents=True)
    py = root / ".venv" / "bin" / "python"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    pip = root / ".venv" / "bin" / "pip"
    pip.write_text("#!/bin/sh\nexit 0\n")
    pip.chmod(0o755)


def test_update_script_replaces_the_package_and_keeps_local_config(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    v1 = pm.make_record(base=None, documents=DOCS[:2])
    root = _extract(_build(corpus_dir, tmp_path / "v1.tar.gz", v1).path, tmp_path / "srv")
    _fake_venv(root)
    cfg = json.loads((root / "rag.config.json").read_text())
    cfg.pop("file_inline_max_mb")                                  # an older package without this key
    (root / "rag.config.json").write_text(json.dumps({**cfg, "port": 9555, "top_k": 3}))
    keyfile = root / "passphrase.txt"
    keyfile.write_text(PASS)
    v2 = pm.make_record(base=v1, documents=DOCS)
    archive2 = _build(corpus_dir, tmp_path / "v2.tar.gz", v2, base_docs=v1["documents"],
                      config=FAST.model_copy(update={"top_k": 7})).path
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
    other = _build(corpus_dir, tmp_path / "other.tar.gz", pm.make_record(base=None, documents=DOCS)).path
    refused = subprocess.run(["bash", "update.sh", str(other), "--keyfile", str(keyfile)], cwd=root, env=env,
                             capture_output=True, text=True, timeout=120)
    assert refused.returncode == 2 and "another package" in refused.stderr
    assert json.loads((root / "package.json").read_text())["version"] == "1.0.0"          # nothing changed
    wrong = tmp_path / "wrong.txt"
    wrong.write_text("not the passphrase")
    bad = subprocess.run(["bash", "update.sh", str(archive2), "--keyfile", str(wrong)], cwd=root, env=env,
                         capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2 and "nothing was changed" in bad.stderr
    done = subprocess.run(["bash", "update.sh", str(archive2), "--keyfile", str(keyfile)], cwd=root, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "version 1.0.0 -> 1.1.0" in done.stderr
    report = json.loads(done.stdout[: done.stdout.rindex("}") + 1])
    assert report["from"] == "1.0.0" and report["to"] == "1.1.0" and "corpus" in report["replaced"]
    assert report["added_config"] == ["file_inline_max_mb"]
    kept = json.loads((root / "rag.config.json").read_text())
    assert kept["port"] == 9555 and kept["top_k"] == 3 and kept["file_inline_max_mb"] == 20   # local edits win
    assert keyfile.read_text() == PASS and (root / ".venv" / "bin" / "python").exists()
    assert json.loads((root / "package.json").read_text())["version"] == "1.1.0"
    assert "1 added" in (root / "CHANGES.md").read_text()
    assert not list(root.glob(".update-*"))
    again = subprocess.run(["bash", "update.sh", str(archive2), "--keyfile", str(keyfile)], cwd=root, env=env,
                           capture_output=True, text=True, timeout=120)
    assert again.returncode == 2 and "not newer than the installed 1.1.0" in again.stderr
    q = subprocess.run([sys.executable, "-I", "server.py", "--query", "Emberfall", "--keyfile", str(keyfile)], cwd=root,
                       env=env, capture_output=True, text=True, timeout=120)
    assert q.returncode == 0 and "emberfall" in q.stdout.lower()                      # the new corpus answers
    forced = subprocess.run(["bash", "setup.sh", "--update", str(other), "--keyfile", str(keyfile), "--force"],
                            cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert forced.returncode == 0, forced.stderr
    assert json.loads((root / "package.json").read_text())["package_id"] != v1["package_id"]


def test_open_an_encrypted_package_with_originals_and_export_its_next_version(
        client, tmp_path: Path, monkeypatch, corpus_dir: Path, files) -> None:  # noqa: F811
    from finetune_studio.data.fs import file_library as fl
    from finetune_studio.data.fs.paths import brain_index_dir
    from finetune_studio.webui.routes import settings as settings_mod

    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.chdir(tmp_path)
    docs = [{**DOCS[0], "sha256": files[0].sha256}, {**DOCS[1], "sha256": files[1].sha256}, DOCS[2]]
    v1 = pm.make_record(base=None, documents=docs)
    archive = _build(corpus_dir, tmp_path / "pkg.tar.gz", v1, originals=files,
                     config=FAST.model_copy(update={"include_originals": True, "semantic_search": True})).path
    pid = client.post("/api/projects", json={"name": "agent-test-open-package"}).json()["id"]
    with open(archive, "rb") as fp:
        wrong = client.post(f"/api/projects/{pid}/brain/import", files={"file": ("pkg.tar.gz", fp)},
                            data={"passphrase": "wrong passphrase!"})
    assert wrong.status_code == 400
    with open(archive, "rb") as fp:
        r = client.post(f"/api/projects/{pid}/brain/import", files={"file": ("pkg.tar.gz", fp)}, data={"passphrase": PASS})
    assert r.status_code == 200, r.text
    opened = r.json()["package"]
    assert opened["package_id"] == v1["package_id"] and opened["version"] == "1.0.0" and opened["restored_files"] == 2
    assert opened["vectors"] is True and r.json()["chunks"] == 3
    corpus = brain_index_dir(pid)
    assert (corpus / "chunks.parquet").is_file() and (corpus / "vectors.npy").is_file()
    assert json.loads((corpus / "manifest.json").read_text())["extra"]["package"]["version"] == "1.0.0"
    library = {f["original_name"]: f for f in fl.list_files(pid)}
    assert set(library) == {"ledger.pdf", "highmere prices.csv"}
    folders = {f["name"] for f in fl.list_folders(pid, include_auto=False)}
    assert folders == {"public"}
    raw = {f.name for f in Path(corpus).parents[1].rglob("*") if f.is_file()}
    assert raw  # the project dir exists
    hist = client.get(f"/api/projects/{pid}/brain/packages").json()
    assert hist["packages"][0]["package_id"] == v1["package_id"] and hist["packages"][0]["versions"][0]["origin"] == "opened"

    client.put(f"/api/projects/{pid}/brain/export-config", json={"scope": "studio", "values": {"kdf_log_n": 10}})
    nxt = client.post(f"/api/projects/{pid}/brain/standalone", json={
        "semantic_search": False, "passphrase": PASS, "new_version_of": v1["package_id"]})
    assert nxt.status_code == 200, nxt.text
    j = nxt.json()
    assert j["package"]["package_id"] == v1["package_id"] and j["package"]["version"] == "1.1.0"
    assert j["package"]["parent_version"] == "1.0.0" and "-v1.1.0" in j["filename"]
    assert set(j["changes"]) == {"added", "changed", "removed"}
    unknown = client.post(f"/api/projects/{pid}/brain/standalone", json={"passphrase": PASS, "new_version_of": "nope"})
    assert unknown.status_code == 404
    page = client.get(f"/projects/{pid}/export").text
    assert 'id="package-versions"' in page and "/brain/packages" in page and 'id="package-changes"' in page
    brain = client.get(f"/projects/{pid}/brain").text
    for marker in ('id="mcp-base"', 'id="mcp-version"', 'id="mcp-base-upload"', "new_version_of: document.getElementById",
                   "/brain/package-base", "open an exported package"):
        assert marker in brain


def test_upload_an_archive_as_the_base(client, tmp_path: Path, monkeypatch, corpus_dir: Path) -> None:  # noqa: F811
    monkeypatch.chdir(tmp_path)
    v1 = pm.make_record(base=None, documents=DOCS)
    archive = _build(corpus_dir, tmp_path / "base.tar.gz", v1).path
    pid = client.post("/api/projects", json={"name": "agent-test-base"}).json()["id"]
    with open(archive, "rb") as fp:
        r = client.post(f"/api/projects/{pid}/brain/package-base", files={"file": ("base.tar.gz", fp)},
                        data={"passphrase": PASS})
    assert r.status_code == 200, r.text
    assert r.json() == {"package_id": v1["package_id"], "version": "1.0.0", "parent_version": None,
                        "created": v1["created"], "documents": 3}
    assert pm.latest(pm.package_dir(pid), v1["package_id"])["origin"] == "uploaded"
    with open(archive, "rb") as fp:   # a keyword-only package opens too; its vectors come from a rebuild
        opened = client.post(f"/api/projects/{pid}/brain/import", files={"file": ("base.tar.gz", fp)},
                             data={"passphrase": PASS})
    assert opened.status_code == 200, opened.text
    assert opened.json()["package"]["vectors"] is False and "Build / Update" in opened.json()["package"]["note"]
    with open(archive, "rb") as fp:
        bad = client.post(f"/api/projects/{pid}/brain/package-base", files={"file": ("base.tar.gz", fp)},
                          data={"passphrase": "nope nope nope"})
    assert bad.status_code == 400 and "cannot read that package" in bad.json()["detail"]
