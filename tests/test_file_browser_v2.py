"""File browser v2 backend: bulk actions, zip export, per-file usage,
parsed-content search, folder management. UI layer is browser-verified live."""
from __future__ import annotations

import io
import secrets
import zipfile
from pathlib import Path

import pytest

from finetune_studio.data import project_filesystem as pfs
from finetune_studio.data.fs import file_library as fl


@pytest.fixture
def fts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fts"
    (root / "projects").mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", root / "projects")
    fl._PARSED_CACHE.clear()
    return root


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Browser V2"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _upload(client, pid: str, name: str, content: bytes) -> str:
    r = client.post(
        f"/api/projects/{pid}/files/upload",
        files=[("files", (name, content, "text/markdown"))],
    )
    assert r.status_code == 200, r.text
    return r.json()["report"][0]["file_id"]


def _doc(label: str) -> bytes:
    body = "\n\n".join(
        f"Section {i} of {label}: the keeper swore on green wax in the year 18{40 + i}."
        for i in range(1, 7)
    )
    return f"# {label}\n\n{body}\n".encode()


def _bulk(client, pid, ids, action, **extra):
    return client.post(f"/api/projects/{pid}/files/bulk",
                       json={"ids": ids, "action": action, **extra})


# ── bulk actions ─────────────────────────────────────────────────────────


def test_bulk_tag_add_remove(fts_root: Path, client) -> None:
    pid = _project(client)
    a = _upload(client, pid, f"a-{secrets.token_hex(3)}.md", _doc("a"))
    b = _upload(client, pid, f"b-{secrets.token_hex(3)}.md", _doc("b"))

    r = _bulk(client, pid, [a, b], "tag-add", tags="lore, saltmere")
    assert r.status_code == 200, r.text
    assert r.json()["succeeded"] == 2
    assert "lore" in (fl.get_file(pid, a) or {})["tags"]

    r = _bulk(client, pid, [a, b], "tag-remove", tags="saltmere")
    assert r.json()["succeeded"] == 2
    assert "saltmere" not in (fl.get_file(pid, b) or {})["tags"]
    assert "lore" in (fl.get_file(pid, b) or {})["tags"]


def test_bulk_delete_restore_cycle(fts_root: Path, client) -> None:
    pid = _project(client)
    a = _upload(client, pid, f"d-{secrets.token_hex(3)}.md", _doc("d"))
    assert _bulk(client, pid, [a], "delete").json()["succeeded"] == 1
    assert (fl.get_file(pid, a) or None) is None  # hidden from live view
    assert _bulk(client, pid, [a], "restore").json()["succeeded"] == 1
    assert fl.get_file(pid, a) is not None


def test_bulk_reparse_discards_override(fts_root: Path, client) -> None:
    pid = _project(client)
    a = _upload(client, pid, f"r-{secrets.token_hex(3)}.md", _doc("r"))
    client.put(f"/api/projects/{pid}/files/{a}/parsed", json={"text": "OVERRIDE"})
    assert "OVERRIDE" in client.get(f"/api/projects/{pid}/files/{a}/parsed").json()["parsed_md"]
    assert _bulk(client, pid, [a], "reparse").json()["succeeded"] == 1
    assert "OVERRIDE" not in client.get(f"/api/projects/{pid}/files/{a}/parsed").json()["parsed_md"]


def test_bulk_move_requires_folder_and_works(fts_root: Path, client) -> None:
    pid = _project(client)
    a = _upload(client, pid, f"m-{secrets.token_hex(3)}.md", _doc("m"))
    assert _bulk(client, pid, [a], "move").json()["failed"] == 1
    folder = client.post(f"/api/projects/{pid}/folders", json={"name": "Vault"}).json()
    r = _bulk(client, pid, [a], "move", folder_id=folder["id"])
    assert r.json()["succeeded"] == 1
    files = client.get(f"/api/projects/{pid}/files?folder_id={folder['id']}").json()["files"]
    assert [f["id"] for f in files] == [a]


def test_bulk_unknown_action_400(fts_root: Path, client) -> None:
    pid = _project(client)
    assert _bulk(client, pid, ["x"], "nuke").status_code == 400


# ── zip export ───────────────────────────────────────────────────────────


def test_download_zip_contains_raw_bytes(fts_root: Path, client) -> None:
    pid = _project(client)
    doc_a = _doc("zipA")
    a = _upload(client, pid, "alpha.md", doc_a)
    b = _upload(client, pid, "beta.md", _doc("zipB"))
    r = client.post(f"/api/projects/{pid}/files/download-zip", json={"ids": [a, b]})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/zip")
    assert "Browser-V2" in r.headers["content-disposition"] or "zip" in r.headers["content-disposition"]
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert any(n.startswith("alpha") for n in names), names
    assert any(n.startswith("beta") for n in names), names
    alpha = next(n for n in names if n.startswith("alpha"))
    assert zf.read(alpha) == doc_a


def test_download_zip_empty_ids_400(fts_root: Path, client) -> None:
    pid = _project(client)
    assert client.post(f"/api/projects/{pid}/files/download-zip",
                       json={"ids": []}).status_code == 400


# ── per-file usage ───────────────────────────────────────────────────────


def test_usage_chains_file_to_runs(fts_root: Path, client) -> None:
    from finetune_studio import db

    pid = _project(client)
    name = f"u-{secrets.token_hex(3)}.md"
    fid = _upload(client, pid, name, _doc("usage"))

    src = client.post(f"/api/projects/{pid}/data-prep/sources", json={"file_id": fid}).json()["source"]
    src_id = src["id"]
    pair_id = f"qa-{src_id}-0001"
    pfs.write_qa_pair(pid, {"id": pair_id, "source_id": src_id, "question": "q", "answer": "a",
                            "status": "approved"})
    ds_dir = pfs.project_dir(pid) / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    ds_path = ds_dir / "ds.jsonl"
    ds_path.write_text('{"id": "' + pair_id + '", "messages": []}\n', encoding="utf-8")
    db.create_dataset(pid, "Usage DS", str(ds_path), source="test", qa_count=1)
    run = db.create_run(pid, "usage-run", base_model="Qwen/Qwen3-4B", data_path=str(ds_path))

    u = client.get(f"/api/projects/{pid}/files/{fid}/usage").json()
    assert u["source"]["id"] == src_id
    assert u["qa_pairs"] == 1
    assert u["datasets"] and u["datasets"][0]["name"] == "Usage DS"
    assert u["runs"] and u["runs"][0]["id"] == run["id"]
    assert u["rag"] is False


def test_usage_no_source(fts_root: Path, client) -> None:
    pid = _project(client)
    fid = _upload(client, pid, f"n-{secrets.token_hex(3)}.csv", b"a,b\n1,2\n")
    u = client.get(f"/api/projects/{pid}/files/{fid}/usage").json()
    assert u["source"] is None
    assert u["datasets"] == [] and u["runs"] == []


# ── parsed-content search ────────────────────────────────────────────────


def test_search_content_finds_snippet(fts_root: Path, client) -> None:
    pid = _project(client)
    fid = _upload(client, pid, f"s-{secrets.token_hex(3)}.md",
                  b"# Lore\n\nThe GREEN WAX SEAL of Saltmere bound the assize.\n")
    r = client.get(f"/api/projects/{pid}/files/search-content", params={"q": "green wax"})
    body = r.json()
    assert r.status_code == 200
    assert body["scanned"] >= 1
    hit = next((m for m in body["matches"] if m["file_id"] == fid), None)
    assert hit and "GREEN WAX" in hit["snippet"].upper()
    assert hit["occurrences"] == 1


def test_search_content_validation(fts_root: Path, client) -> None:
    pid = _project(client)
    assert client.get(f"/api/projects/{pid}/files/search-content",
                      params={"q": "x"}).status_code == 400


# ── folder management (data-page surface) ───────────────────────────────


def test_folder_rename_delete_survives_files(fts_root: Path, client) -> None:
    pid = _project(client)
    fid = _upload(client, pid, f"f-{secrets.token_hex(3)}.md", _doc("f"))
    folder = client.post(f"/api/projects/{pid}/folders", json={"name": "Chronicles"}).json()
    _bulk(client, pid, [fid], "move", folder_id=folder["id"])

    ren = client.patch(f"/api/projects/{pid}/folders/{folder['id']}",
                       json={"name": "Chronicles II"})
    assert ren.status_code == 200
    # folder names are sanitized (_safe_stem): spaces become underscores
    names = {f["name"] for f in client.get(f"/api/projects/{pid}/folders").json()["folders"]}
    assert "Chronicles_II" in names

    assert client.delete(f"/api/projects/{pid}/folders/{folder['id']}").status_code == 200
    # file survives, just folder-less
    alive = client.get(f"/api/projects/{pid}/files").json()["files"]
    assert any(x["id"] == fid for x in alive)


# ── UI smoke: the v2 controls exist ──────────────────────────────────────


def test_data_page_renders_v2_controls(client) -> None:
    pid = _project(client)
    html = client.get(f"/projects/{pid}/data").text
    for marker in ("fb-folders-row", "fb-page-bar", "fb-cols-btn", "fb-content-toggle",
                   "fb-usage-modal", "fb-bulk-zip", "fb-bulk-tag", "fb-bulk-reparse",
                   "fb-trash-retention", "fb-content-results"):
        assert marker in html, marker
