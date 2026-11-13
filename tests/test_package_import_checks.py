"""Opening a package (walkthrough 2026-11-13 M5, m12).

M5: an opened package was recorded as ``upload-1794563690385681400.tar.gz`` (the staged temp name) in the versions
table, the base select and the next file name. m12: a truncated ``.tar.gz`` or a non-zip ``.zip`` gave HTTP 500
"bundle import failed: EOFError", and the "replace the index?" question came before the file was checked.
"""
from __future__ import annotations

from pathlib import Path

from finetune_studio.smart_brain import package_import
from finetune_studio.smart_brain import package_meta as pm
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)
from tests.test_package_versions import DOCS, PASS, _build

ROOT = Path(__file__).resolve().parents[1]


def _project(client, tmp_path: Path, monkeypatch, name: str) -> str:
    from finetune_studio.webui.routes import settings as settings_mod

    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.chdir(tmp_path)
    return client.post("/api/projects", json={"name": name}).json()["id"]


def _post(client, pid: str, path: Path, name: str, passphrase: str = PASS, overwrite: bool = False):
    with open(path, "rb") as fp:
        return client.post(f"/api/projects/{pid}/brain/import" + ("?overwrite=true" if overwrite else ""),
                           files={"file": (name, fp)}, data={"passphrase": passphrase})


def test_damaged_files_are_a_400_with_a_plain_reason(client, tmp_path: Path, monkeypatch, corpus_dir: Path) -> None:  # noqa: F811
    pid = _project(client, tmp_path, monkeypatch, "agent-test-import-damaged")
    good = _build(corpus_dir, tmp_path / "good.tar.gz", pm.make_record(base=None, documents=DOCS)).path
    cut = tmp_path / "cut.tar.gz"
    cut.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    r = _post(client, pid, cut, "cut.tar.gz", overwrite=True)
    assert r.status_code == 400, r.text
    assert "damaged" in r.json()["detail"] and "cut.tar.gz" in r.json()["detail"]
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"this is not a zip file at all" * 10)
    r = _post(client, pid, junk, "junk.zip", overwrite=True)
    assert r.status_code == 400 and "not a Smart Brain package" in r.json()["detail"], r.text


def test_an_existing_index_is_replaced_only_after_the_file_checks_out(client, tmp_path: Path, monkeypatch,
                                                                       corpus_dir: Path) -> None:  # noqa: F811
    pid = _project(client, tmp_path, monkeypatch, "agent-test-import-confirm")
    v1 = pm.make_record(base=None, documents=DOCS)
    archive = _build(corpus_dir, tmp_path / "x.tar.gz", {**v1, "name": "Quillfen staff"}).path
    assert _post(client, pid, archive, "Quillfen-staff-smart-brain-1234abcd-v1.0.0.tar.gz", overwrite=True).status_code == 200
    # Index exists now: the file is checked (and its passphrase) before the question.
    wrong = _post(client, pid, archive, "again.tar.gz", passphrase="not the passphrase")
    assert wrong.status_code == 400, wrong.text
    ask = _post(client, pid, archive, "again.tar.gz")
    assert ask.status_code == 409 and ask.json()["code"] == "index_exists"
    assert ask.json()["package"] == {"package_id": v1["package_id"], "version": "1.0.0", "name": "Quillfen staff"}
    page = client.get(f"/projects/{pid}/brain").text
    assert "e.status === 409" in page and "Checking ' + f.name" in page


def test_an_opened_package_keeps_its_name(client, tmp_path: Path, monkeypatch, corpus_dir: Path) -> None:  # noqa: F811
    pid = _project(client, tmp_path, monkeypatch, "agent-test-import-name")
    named = _build(corpus_dir, tmp_path / "a.tar.gz", {**pm.make_record(base=None, documents=DOCS),
                                                       "name": "Quillfen public bot"}).path
    assert _post(client, pid, named, "whatever.tar.gz", overwrite=True).status_code == 200
    names = {p["name"] for p in client.get(f"/api/projects/{pid}/brain/packages").json()["packages"]}
    assert names == {"Quillfen public bot"}
    # An older package without a name inside: the file name the user chose, without the export's suffix.
    old = _build(corpus_dir, tmp_path / "b.tar.gz", pm.make_record(base=None, documents=DOCS)).path
    assert _post(client, pid, old, "Highmere-docs-smart-brain-0a1b2c3d-v1.0.0.tar.gz", overwrite=True).status_code == 200
    rows = client.get(f"/api/projects/{pid}/brain/packages").json()["packages"]
    assert {p["name"] for p in rows} == {"Quillfen public bot", "Highmere-docs"}
    assert not any(str(v.get("filename") or "").startswith("upload-") for p in rows for v in p["versions"])


def test_display_name_rules() -> None:
    assert package_import._display_name({"name": "Staff"}, "x.zip") == "Staff"
    assert package_import._display_name({}, "Acme-smart-brain-0a1b2c3d-v1.2.0-with-models.tar.gz") == "Acme"
    assert package_import._display_name({}, "my package.zip") == "my package"
