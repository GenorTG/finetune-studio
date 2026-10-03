"""Cross-project ownership + root regressions (APP-AUDIT-2026-10-02).

Four verified findings, one regression test group each:

1. Chat (``chat_v2``) must not search a RAG owned by another project.
2. File-library version/conversion data + routes must be scoped to the
   project in the URL (a foreign file id is a 404, never metadata/paths).
3. ``_persist_rag_report`` must not attach an evaluation to a run that
   belongs to a different project.
4. The RAG build/status source root must follow ``FTS_ROOT`` (the shared
   ``data.fs.paths`` helpers), not a hard-coded ``~/.finetune-studio``.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finetune_studio import db


def _mk_project(client, name: str) -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


@pytest.fixture
def fts_root(tmp_path: Path, monkeypatch) -> Path:
    """A custom FTS_ROOT (what the env var yields at import) under tmp_path."""
    root = tmp_path / "custom-fts-root"
    root.mkdir()
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", root / "projects")
    # A home dir that must NOT be consulted for project sources.
    home = tmp_path / "decoy-home"
    (home / ".finetune-studio").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return root


# ── 1. chat_v2 RAG ownership ─────────────────────────────────────────────


def test_chat_ignores_rag_owned_by_another_project(client, monkeypatch) -> None:
    pid_a = _mk_project(client, "own-chat-a")
    pid_b = _mk_project(client, "own-chat-b")
    mine = db.create_rag(project_id=pid_a, name="mine", store_path="/tmp/mine")
    foreign = db.create_rag(project_id=pid_b, name="foreign", store_path="/tmp/foreign")

    searched: list[str] = []

    def fake_search(store_path, query, top_k=5):
        searched.append(store_path)
        return [{"text": "t", "score": 0.5, "chunk_id": f"c-{store_path}"}]

    monkeypatch.setattr(
        "finetune_studio.webui.routes.chat_v2._search_rag_attachment", fake_search
    )
    r = client.post(
        f"/api/chat-v2/projects/{pid_a}/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "enabled_rag_ids": [mine["id"], foreign["id"]],
        },
    )
    assert r.status_code == 200, r.text
    assert searched == ["/tmp/mine"], f"foreign corpus was searched: {searched}"
    assert {s["rag_id"] for s in r.json().get("sources", [])} <= {mine["id"]}


# ── 2. file-library version/conversion ownership ─────────────────────────


def _upload(client, pid: str, name: str, data: bytes) -> str:
    r = client.post(
        f"/api/projects/{pid}/files/upload", files=[("files", (name, data))]
    )
    assert r.status_code == 200, r.text
    item = r.json()["report"][0]
    assert item["status"] == "uploaded", item
    return item["file_id"]


def _add_conversion(file_id: str) -> None:
    with db.cursor() as c:
        c.execute(
            "INSERT INTO file_conversions (file_id, version, format, converted_path, "
            "converted_hash, converted_size, converter, converted_at, status, error_message) "
            "VALUES (?, 1, 'md', '/secret/conv.md', 'h', 1, 'test', ?, 'ok', '')",
            (file_id, time.time()),
        )


@pytest.fixture
def two_file_projects(client, fts_root):
    pid_a = _mk_project(client, "own-files-a")
    pid_b = _mk_project(client, "own-files-b")
    fid_a = _upload(client, pid_a, "a.bin", b"alpha-bytes")
    fid_b = _upload(client, pid_b, "b.bin", b"bravo-bytes")
    _add_conversion(fid_b)
    return pid_a, pid_b, fid_a, fid_b


def test_list_versions_and_conversions_scoped_to_project(two_file_projects) -> None:
    from finetune_studio.data.fs import file_library as fl

    pid_a, pid_b, _fid_a, fid_b = two_file_projects
    assert fl.list_versions(pid_b, fid_b), "sanity: owner sees its versions"
    assert fl.list_conversions(pid_b, fid_b), "sanity: owner sees its conversions"
    assert fl.list_versions(pid_a, fid_b) == []
    assert fl.list_conversions(pid_a, fid_b) == []


@pytest.mark.parametrize(
    "suffix", ["", "/versions", "/conversions", "/raw", "/parsed", "/usage"]
)
def test_file_routes_404_for_foreign_file_id(client, two_file_projects, suffix) -> None:
    pid_a, _pid_b, _fid_a, fid_b = two_file_projects
    r = client.get(f"/api/projects/{pid_a}/files/{fid_b}{suffix}")
    assert r.status_code == 404, f"{suffix or '(meta)'}: {r.status_code} {r.text[:200]}"
    assert "/secret/" not in r.text


def test_file_routes_still_serve_own_file(client, two_file_projects) -> None:
    pid_a, _pid_b, fid_a, _fid_b = two_file_projects
    r = client.get(f"/api/projects/{pid_a}/files/{fid_a}/versions")
    assert r.status_code == 200 and len(r.json()["versions"]) == 1
    r = client.get(f"/api/projects/{pid_a}/files/{fid_a}/conversions")
    assert r.status_code == 200 and r.json()["conversions"] == []


@pytest.mark.parametrize(
    ("method", "suffix", "body"),
    [
        ("put", "/parsed", {"text": "x"}),
        ("post", "/reparse", None),
        ("patch", "/rename", {"new_name": "pwn.bin"}),
        ("post", "/purge", None),
        ("delete", "", None),
        ("post", "/restore", None),
        ("patch", "/tags", {"tags": "x", "notes": "y"}),
    ],
)
def test_mutating_file_routes_404_for_foreign_file_id(
    client, two_file_projects, method, suffix, body
) -> None:
    pid_a, pid_b, _fid_a, fid_b = two_file_projects
    kwargs = {"json": body} if body is not None else {}
    r = getattr(client, method)(f"/api/projects/{pid_a}/files/{fid_b}{suffix}", **kwargs)
    assert r.status_code == 404, f"{method} {suffix}: {r.status_code} {r.text[:200]}"
    # The owner's file is untouched.
    assert client.get(f"/api/projects/{pid_b}/files/{fid_b}").status_code == 200


# ── 3. _persist_rag_report run ownership ─────────────────────────────────


def _report() -> dict:
    return {
        "model_path": "/models/m",
        "results": [],
        "scores": {},
        "retrieval": {},
        "corpus_path": "/c",
    }


def test_persist_rag_report_does_not_link_foreign_run(client) -> None:
    from finetune_studio.webui.routes.testing import _persist_rag_report

    pid_a = _mk_project(client, "own-run-a")
    pid_b = _mk_project(client, "own-run-b")
    foreign_run = db.create_run(project_id=pid_b, name="run-b")

    bench = _persist_rag_report(
        project_id=pid_a,
        requested_run_id=foreign_run["id"],
        suite_path="/s/suite.json",
        report=_report(),
    )
    assert bench["run_id"] != foreign_run["id"]
    assert db.get_run(bench["run_id"])["project_id"] == pid_a
    assert db.list_benchmarks(foreign_run["id"]) == []


def test_persist_rag_report_links_own_run(client) -> None:
    from finetune_studio.webui.routes.testing import _persist_rag_report

    pid_a = _mk_project(client, "own-run-ok")
    run = db.create_run(project_id=pid_a, name="run-a")
    bench = _persist_rag_report(
        project_id=pid_a,
        requested_run_id=run["id"],
        suite_path="/s/suite.json",
        report=_report(),
    )
    assert bench["run_id"] == run["id"]


def test_persist_rag_report_without_project_rejects_requested_run(client) -> None:
    """No project to prove ownership against -> fail closed, never link the run."""
    from finetune_studio.webui.routes.testing import _persist_rag_report

    pid_b = _mk_project(client, "own-run-noproj")
    run = db.create_run(project_id=pid_b, name="run-b")
    with pytest.raises(ValueError, match="project_id"):
        _persist_rag_report(
            project_id="",
            requested_run_id=run["id"],
            suite_path="/s/suite.json",
            report=_report(),
        )
    assert db.list_benchmarks(run["id"]) == []


# ── 4. RAG source root honours FTS_ROOT ──────────────────────────────────


def _seed_parsed(fts_root: Path, pid: str) -> Path:
    files = fts_root / "projects" / pid / "files"
    (files / "abc123def456").mkdir(parents=True)
    (files / "abc123def456" / "parsed.txt").write_text("hello corpus", encoding="utf-8")
    return files


def test_rag_build_sources_from_fts_root(client, fts_root, tmp_path, monkeypatch) -> None:
    pid = _mk_project(client, "own-root-build")
    files = _seed_parsed(fts_root, pid)
    corpus = tmp_path / "corpus"
    monkeypatch.setattr("finetune_studio.webui.routes.rag._corpus_dir", lambda p: corpus)

    mock_rag = MagicMock()
    mock_rag.build_from_directory.return_value = {"documents": 1, "chunks": 1, "vector_dim": 8}
    with patch("finetune_studio.data.rag_portable.PortableRAG", return_value=mock_rag):
        r = client.post(f"/api/projects/{pid}/rag/build", json={"reset": True})
    assert r.status_code == 200, r.text
    assert r.json()["queued_files"] == 1
    kwargs = mock_rag.build_from_directory.call_args.kwargs
    assert Path(kwargs["source_dir"]) == files


def test_rag_build_snapshot_counts_fts_root_files(client, fts_root) -> None:
    from finetune_studio.webui.routes.rag import _rag_build_snapshot

    pid = _mk_project(client, "own-root-snap")
    _seed_parsed(fts_root, pid)
    assert _rag_build_snapshot(pid)["files_total"] == 1


def test_project_rag_files_dir_follows_fts_root(client, fts_root) -> None:
    from finetune_studio.webui.routes.project_rag import project_files_dir

    pid = _mk_project(client, "own-root-prag")
    assert project_files_dir(pid) == fts_root / "projects" / pid / "files"


def test_rag_build_still_400_when_project_has_no_files_dir(client, fts_root) -> None:
    """Resolving the shared path must not mkdir it (that would defeat the 400)."""
    pid = _mk_project(client, "own-root-nofiles")
    shutil_target = fts_root / "projects" / pid / "files"
    if shutil_target.exists():
        shutil_target.rmdir()
    r = client.post(f"/api/projects/{pid}/rag/build", json={})
    assert r.status_code == 400, r.text
    assert not shutil_target.exists()
