"""The package opens itself before it is handed over, and the Package page says what is left to do."""
from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio.knowledge import readiness as rd
from finetune_studio.smart_brain import identity as bi
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package, verify_package
from tests.test_brain_server_package import corpus_dir  # noqa: F401

FAST = BrainExportConfig(kdf_log_n=10)  # cheap scrypt for tests
PASS = "correct horse battery staple"


def _states(result: dict) -> dict[str, str]:
    return {s["key"]: s["state"] for s in result["steps"]}


def test_a_good_encrypted_package_passes_and_reports_its_identity(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", name="T", config=FAST, passphrase=PASS,
                          identity={"name": "Ada", "system_prompt": "You are Ada."})
    out = verify_package(built.path, PASS)
    assert out["ok"] is True and out["hits"] > 0 and out["identity"] is True
    assert out["layers"]["keyword"]["available"] is True
    assert out["layers"]["semantic"]["available"] is False  # no embedder bundled in this test package
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".verify-")] == []     # nothing left behind


def test_the_wrong_passphrase_fails_and_says_so(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", name="T", config=FAST, passphrase=PASS)
    out = verify_package(built.path, "wrong passphrase here")
    assert out["ok"] is False and out["detail"] and out["hits"] == 0


def test_a_damaged_archive_fails_instead_of_raising(tmp_path: Path) -> None:
    bad = tmp_path / "p.tar.gz"
    bad.write_bytes(b"not an archive")
    out = verify_package(bad, PASS)
    assert out["ok"] is False and "could not open" in out["detail"]


def test_a_zip_package_passes(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.zip", name="T", fmt="zip",
                          config=BrainExportConfig(kdf_log_n=10, archive_format="zip"), passphrase=PASS)
    assert verify_package(built.path, PASS)["ok"] is True


def test_package_self_test_uses_the_bundled_search_settings(corpus_dir: Path, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    from finetune_studio.smart_brain import server_package

    calls: list[list[str]] = []
    original = server_package.subprocess.run

    def record(args, *a, **kw):
        calls.append(list(args))
        return original(args, *a, **kw)

    monkeypatch.setattr(server_package.subprocess, "run", record)
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", name="T", config=FAST, passphrase=PASS)
    assert verify_package(built.path, PASS)["ok"] is True
    mcp = next(args for args in calls if "--mcp" in args)
    assert "--no-semantic" not in mcp and "--no-reranker" not in mcp


@pytest.fixture
def project(client) -> str:
    return client.post("/api/projects", json={"name": "ready-test"}).json()["id"]


def test_an_empty_project_is_not_ready_and_every_step_is_listed(client, project: str) -> None:
    r = client.get(f"/api/projects/{project}/brain/readiness").json()
    assert r["ready"] is False
    assert list(_states(r)) == ["files", "check", "identity", "index", "test", "package"]
    assert set(_states(r).values()) == {"todo"}


def test_identity_and_a_failed_self_test_show_up(client, project: str) -> None:
    bi.save(project, {"name": "Ada"})
    rd.record_package(project, filename="p.tar.gz", size=1, encrypted=True, identity=True,
                      verified={"ok": False, "detail": "the server would not start"})
    r = client.get(f"/api/projects/{project}/brain/readiness").json()
    steps = {s["key"]: s for s in r["steps"]}
    assert steps["identity"]["state"] == "done" and steps["identity"]["detail"] == "Ada"
    assert steps["package"]["state"] == "problem" and "would not start" in steps["package"]["detail"]


def test_a_package_older_than_the_index_needs_review(monkeypatch, client, project: str) -> None:
    rd.record_package(project, filename="p.tar.gz", size=1, encrypted=True, identity=False, verified={"ok": True, "detail": "x"})
    monkeypatch.setattr(rd, "_index_time", lambda pid: 9e12)
    assert _states(rd.readiness(project))["package"] == "review"


def test_the_package_page_shows_the_list(client, project: str) -> None:
    body = client.get(f"/projects/{project}/export").text
    assert 'id="package-readiness"' in body and "<h1>Package</h1>" in body and "Not ready yet" in body
    assert 'id="package-build-link"' in body and "Trained models" in body
    assert ">4 package<" in body
