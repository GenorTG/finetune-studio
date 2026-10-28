"""The .ftsrag studio-to-studio bundle, live-checked 2026-10-28 (beta work order B2.1).

Live findings fixed here: the import form sat inside the export panel, which stays hidden until the project has an
index, so a new project (where a received bundle goes) could not import one; an imported index reported "rebuild
needed: N file(s) removed" (the exporting project's files are absent by design); the import never registered the
index in ``brain_indexes``, which the Guide reads. CLI: ``fts brain bundle`` / ``fts brain import``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.knowledge import index
from finetune_studio.smart_brain import secure_bundle as sb
from tests.knowledge_helpers import add_source, wait_build
from tests.test_brain_build_stop_progress import _encode

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
PASS = "bundle-test-pass-1"


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.smart_brain.schema import EmbeddingModelInfo

    monkeypatch.setattr("finetune_studio.smart_brain.store.get_embedder",
                        lambda name="", device="cpu": (_encode, EmbeddingModelInfo(name=name or "fake", dim=8)))


def _source_project(client) -> tuple[str, Path]:
    pid = client.post("/api/projects", json={"name": "Bundle source"}).json()["id"]
    add_source(pid, "runbook.txt", "The kchub outbox is capped at 5,000 files.\n\nAt the cap it stops accepting tags.")
    add_source(pid, "policy.txt", "Laptops use BitLocker on Windows and LUKS on Linux.")
    assert wait_build(client, pid, client.post(f"/api/projects/{pid}/brain/build", json={}))["ok"]
    return pid, index.brain_index_dir(pid)


def test_import_into_a_new_project_registers_it_and_marks_it_imported(client, tmp_path: Path) -> None:
    _src, corpus = _source_project(client)
    bundle, _ = sb.export_secure_bundle(corpus, tmp_path / "b.ftsrag", PASS, log_n=10)
    dst = client.post("/api/projects", json={"name": "Bundle target"}).json()["id"]
    assert db.list_brain_indexes(dst) == []
    with open(bundle, "rb") as f:
        r = client.post(f"/api/projects/{dst}/brain/import?overwrite=true",
                        files={"file": ("b.ftsrag", f, "application/octet-stream")}, data={"passphrase": PASS})
    assert r.status_code == 200, r.text
    assert r.json() == {"documents": 2, "chunks": 2, "imported_from": "Bundle source"}
    rows = db.list_brain_indexes(dst)
    assert len(rows) == 1 and rows[0]["store_path"] == str(index.brain_index_dir(dst))   # the Guide sees it
    st = client.get(f"/api/projects/{dst}/brain").json()["state"]
    assert st["fresh"] is True and st["removed"] == [] and len(st["imported"]) == 2
    assert st["imported_from"] == "Bundle source"
    hits = client.post(f"/api/projects/{dst}/brain/search", json={"query": "kchub outbox cap", "top_k": 1}).json()
    assert hits["hits"][0]["filename"] == "runbook.txt"


def test_a_build_in_the_importing_project_drops_the_imported_marker(client, tmp_path: Path) -> None:
    _src, corpus = _source_project(client)
    bundle, _ = sb.export_secure_bundle(corpus, tmp_path / "b.ftsrag", PASS, log_n=10)
    dst = client.post("/api/projects", json={"name": "Bundle target 2"}).json()["id"]
    with open(bundle, "rb") as f:
        assert client.post(f"/api/projects/{dst}/brain/import?overwrite=true", files={"file": ("b.ftsrag", f)},
                           data={"passphrase": PASS}).status_code == 200
    add_source(dst, "own.txt", "This project's own file: the depot opens at 06:00.")
    st = client.get(f"/api/projects/{dst}/brain").json()["state"]
    assert st["fresh"] is False and len(st["missing"]) == 1 and st["removed"] == []
    assert wait_build(client, dst, client.post(f"/api/projects/{dst}/brain/build", json={"reset": True}))["ok"]
    st = client.get(f"/api/projects/{dst}/brain").json()["state"]
    assert st["imported_from"] == "" and st["fresh"] is True and len(st["indexed"]) == 1


def test_the_import_form_is_outside_the_panels_that_need_an_index() -> None:
    html = (TEMPLATES / "brain.html").read_text(encoding="utf-8")
    panel = html.split('id="brain-import-panel"', 1)[1].split("</details>", 1)[0]
    assert 'id="bundle-import-file"' in panel and 'id="bundle-import-btn"' in panel
    assert "data-needs-index" not in html.split('id="brain-import-panel"', 1)[0].rsplit("<details", 1)[1]
    assert 'id="bundle-import-file"' not in html.split('id="brain-export-panel"', 1)[1].split("</details>", 1)[0]
    assert "window.BRAIN_HAS_INDEX && !await fts.confirm" in html          # only an existing index asks first
    assert "Imported from “" in html and "file(s) came with the bundle" in html


def test_fts_brain_bundle_and_import_round_trip(client, tmp_path: Path, monkeypatch, capsys) -> None:
    from finetune_studio.cli import _registry

    src, _ = _source_project(client)
    dst = client.post("/api/projects", json={"name": "Bundle cli target"}).json()["id"]
    (tmp_path / "pass.txt").write_text(PASS)

    def fts(*argv: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["fts", *argv])
        try:
            _registry.main()
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        out = capsys.readouterr()
        return code, out.out + out.err

    code, out = fts("brain", "bundle", "--project", src, "--passphrase-file", str(tmp_path / "pass.txt"),
                    "--name", "cli", "--out", str(tmp_path))
    assert code == 0, out
    path = json.loads(out)["path"]
    code, out = fts("brain", "import", path, "--project", dst)
    assert code != 0 and "--passphrase-file" in out
    code, out = fts("brain", "import", path, "--project", dst, "--passphrase-file", str(tmp_path / "pass.txt"))
    assert code == 0 and json.loads(out)["documents"] == 2, out
    code, out = fts("brain", "import", path, "--project", dst, "--passphrase-file", str(tmp_path / "pass.txt"))
    assert code != 0 and "--replace" in out
    (tmp_path / "bad.txt").write_text("wrong-pass-123")
    code, out = fts("brain", "import", path, "--project", dst, "--replace", "--passphrase-file", str(tmp_path / "bad.txt"))
    assert code != 0 and "wrong passphrase" in out
    assert db.list_brain_indexes(dst)
