"""Nested folders: a folder's name is its path (``public/faq``), created with its parents, kept by uploads (API folder
paths, browser relative paths, CLI directories), the Smart Brain index, the Export scope and restored packages."""
from __future__ import annotations

import argparse
import json

import pytest
from fastapi import HTTPException

from finetune_studio.data.fs import file_library as fl
from finetune_studio.smart_brain import scope


def _project(client, name: str) -> str:
    return client.post("/api/projects", json={"name": name}).json()["id"]


def _names(pid: str) -> dict[str, str | None]:
    folders = fl.list_folders(pid, include_auto=False)
    by_id = {f["id"]: f["name"] for f in folders}
    return {f["name"]: by_id.get(f["parent_id"]) for f in folders}


def test_folder_paths_are_cleaned() -> None:
    assert fl.folder_path("public//faq/./x") == "public/faq/x"
    assert fl.folder_path(r"a\b c") == "a/b_c"
    assert fl.folder_path("../..") == ""


def test_create_makes_the_parents_and_refuses_a_duplicate(client) -> None:
    pid = _project(client, "nest-create")
    leaf = fl.create_folder(pid, "public/faq/returns")
    assert leaf["name"] == "public/faq/returns"
    assert _names(pid) == {"public": None, "public/faq": "public", "public/faq/returns": "public/faq"}
    with pytest.raises(HTTPException) as e:
        fl.create_folder(pid, "public/faq")
    assert e.value.status_code == 409
    assert fl.ensure_folder(pid, "public/faq")["id"] == next(f["id"] for f in fl.list_folders(pid) if f["name"] == "public/faq")


def test_rename_and_move_take_sub_folders_along_and_delete_keeps_files(client) -> None:
    pid = _project(client, "nest-rename")
    faq = fl.create_folder(pid, "public/faq")
    fl.create_folder(pid, "public/faq/returns")
    assert fl.rename_folder(pid, faq["id"], "help")["name"] == "public/help"
    assert set(_names(pid)) == {"public", "public/help", "public/help/returns"}
    moved = fl.rename_folder(pid, faq["id"], "archive/help")
    assert moved["name"] == "archive/help" and _names(pid)["archive/help/returns"] == "archive/help"
    with pytest.raises(HTTPException):
        fl.rename_folder(pid, faq["id"], "archive/help/returns/deeper")
    out = fl.delete_folder(pid, faq["id"])
    assert len(out["subfolders"]) == 1 and set(_names(pid)) == {"public", "archive"}


def test_one_level_folders_from_before_still_work(client) -> None:
    pid = _project(client, "nest-legacy")
    old = client.post(f"/api/projects/{pid}/folders", json={"name": "internal"}).json()
    assert old["name"] == "internal" and old["parent_id"] is None
    sub = fl.create_folder(pid, "internal/hr")
    assert sub["parent_id"] == old["id"]


def test_upload_keeps_the_tree_from_paths_folder_path_and_file_names(client) -> None:
    pid = _project(client, "nest-upload")
    files = [("files", ("returns.md", b"Returns within 30 days. " * 20, "text/markdown")),
             ("files", ("prices.md", b"Prices from 49 GBP. " * 20, "text/markdown"))]
    r = client.post(f"/api/projects/{pid}/files/upload", files=files,
                    data={"paths": ["faq/returns.md", "pricing/2025/prices.md"], "folder_path": "public"})
    report = r.json()["report"]
    assert [x["folder"] for x in report] == ["public/faq", "public/pricing/2025"]
    assert [x["filename"] for x in report] == ["returns.md", "prices.md"]
    r = client.post(f"/api/projects/{pid}/files/upload",
                    files=[("files", ("internal/hr/handbook.md", b"Holiday is 25 days. " * 20, "text/markdown"))])
    assert r.json()["report"][0]["folder"] == "internal/hr"
    listed = {f["original_name"]: f["first_folder"] for f in client.get(f"/api/projects/{pid}/files").json()["files"]}
    assert listed == {"returns.md": "public/faq", "prices.md": "public/pricing/2025", "handbook.md": "internal/hr"}
    paths = fl.folder_paths(pid)
    assert sorted(paths.values()) == ["internal/hr", "public/faq", "public/pricing/2025"]


def test_upload_into_a_chosen_folder_id(client) -> None:
    pid = _project(client, "nest-upload-id")
    faq = fl.create_folder(pid, "public/faq")
    r = client.post(f"/api/projects/{pid}/files/upload", data={"folder_id": faq["id"], "paths": ["sub/a.md"]},
                    files=[("files", ("a.md", b"Words about kettles. " * 20, "text/markdown"))])
    assert r.json()["report"][0]["folder"] == "public/faq/sub"


def test_cli_upload_keeps_a_directory_tree(client, tmp_path, capsys) -> None:
    from finetune_studio.cli.commands import files as cli_files

    pid = _project(client, "nest-cli")
    root = tmp_path / "public"
    (root / "faq").mkdir(parents=True)
    (root / "faq" / "returns.md").write_text("Returns within 30 days. " * 20)
    (root / "about.md").write_text("About the company and its hubs. " * 20)
    (root / ".hidden").write_text("x")
    cli_files._upload_files(argparse.Namespace(project=pid, paths=[str(root)], folder="site", json=True))
    out = json.loads(capsys.readouterr().out)
    assert sorted((x["filename"], x["folder"]) for x in out["report"] if x["status"] == "uploaded") == [
        ("about.md", "site/public"), ("returns.md", "site/public/faq")]


def test_scope_paths_and_prefixes_at_any_depth(client, monkeypatch, tmp_path) -> None:
    pid = _project(client, "nest-scope")
    client.post(f"/api/projects/{pid}/files/upload", data={"paths": ["public/faq/r.md", "public/a.md", "internal/x.md"]},
                files=[("files", (n, f"Text of {n} about leasing coffee machines. ".encode() * 10, "text/markdown"))
                       for n in ("r.md", "a.md", "x.md")])
    from finetune_studio.knowledge.index import documents
    docs = {d.filename: d.path for d in documents(pid)}
    assert docs == {"r.md": "public/faq/r.md", "a.md": "public/a.md", "x.md": "internal/x.md"}
    from finetune_studio.knowledge.sources import list_sources
    meta = [{"document_id": s.id, "filename": s.filename} for s in list_sources(pid)]
    monkeypatch.setattr(scope, "_documents_meta", lambda corpus: meta)
    refs = scope.documents(tmp_path, pid)
    assert sorted(d.path for d in refs) == ["internal/x.md", "public/a.md", "public/faq/r.md"]
    over = scope.overview(refs)
    assert {f["prefix"]: f["count"] for f in over["folders"]} == {"internal/": 1, "public/": 2, "public/faq/": 1}
    keep, _ = scope.resolve(scope.DocumentScope(include=["public/faq/"]), refs)
    assert len(keep) == 1


def test_index_metadata_carries_the_path_for_a_package_without_project(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scope, "_documents_meta", lambda corpus: [
        {"document_id": "d1", "filename": "r.md", "path": "public/faq/r.md"}])
    assert [d.path for d in scope.documents(tmp_path)] == ["public/faq/r.md"]


def test_restored_originals_land_in_their_nested_folder(client) -> None:
    from finetune_studio.smart_brain import package_import

    pid = _project(client, "nest-import")
    cache: dict[str, str] = {}
    fid = package_import._folder_id(pid, "public/faq", cache)
    assert _names(pid) == {"public": None, "public/faq": "public"}
    assert package_import._folder_id(pid, "public/faq", cache) == fid
