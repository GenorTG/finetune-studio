"""Deleting a training run removes its rows AND its own output directory (``training.run_delete``, 2026-10-16).

Before: ``DELETE /api/projects/{pid}/runs/{rid}`` (and the benchmarks mirror) dropped the ``training_runs`` row
only; ``output/projects/<pid>/runs/<rid>`` (18 GB for a merged 9B) stayed on disk for ever.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.jobs import launch, store
from finetune_studio.training import run_delete


def _project(client, name: str = "Run delete") -> str:
    return client.post("/api/projects", json={"name": name}).json()["id"]


def _run_with_output(tmp_path: Path, pid: str, *, shared: bool = False) -> tuple[str, Path]:
    run = db.create_run(project_id=pid, name="trained", base_model="base", settings_obj={})
    rid = run["id"]
    out = tmp_path / "output" / "projects" / pid / ("shared-out" if shared else f"runs/{rid}")
    (out / "merged").mkdir(parents=True)
    (out / "merged" / "model.safetensors").write_bytes(b"x" * 4096)
    (out / "gguf").mkdir()
    (out / "gguf" / "model-Q4_K_M.gguf").write_bytes(b"y" * 1024)
    db.update_run(rid, output_path=str(out), status="done")
    return rid, out


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Training writes ``output/`` relative to the cwd: the test's cwd is a temp dir, never the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_delete_removes_the_rows_the_output_dir_and_the_production_pointer(client, cwd: Path) -> None:
    pid = _project(client)
    rid, out = _run_with_output(cwd, pid)
    bench = db.create_benchmark(rid, suite_name="quiz", scores={"accuracy": 1.0})
    db.create_export(project_id=pid, run_id=rid, format="gguf", quant="Q4_K_M")
    db.update_project(pid, production_run=rid)
    assert out.is_dir()

    r = client.delete(f"/api/projects/{pid}/runs/{rid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["removed_dir"] == str(out) and body["kept"] is None
    assert body["freed_bytes"] == 4096 + 1024
    assert not out.exists()
    assert (cwd / "output" / "projects" / pid).is_dir(), "only the run's own directory goes"
    assert db.get_run(rid) is None
    assert db.get_benchmark(bench["id"]) is None, "benchmark rows cascade"
    assert db.list_exports_for_run(rid) == []
    assert db.get_project(pid)["production_run"] in (None, "")
    assert client.delete(f"/api/projects/{pid}/runs/{rid}").status_code == 404


def test_the_benchmarks_mirror_route_deletes_the_same_way(client, cwd: Path) -> None:
    pid = _project(client)
    rid, out = _run_with_output(cwd, pid)
    r = client.delete(f"/api/benchmarks/projects/{pid}/runs/{rid}")
    assert r.status_code == 200, r.text
    assert r.json()["removed_dir"] == str(out) and not out.exists() and db.get_run(rid) is None
    other = _project(client, "other")
    rid2, out2 = _run_with_output(cwd, other)
    assert client.delete(f"/api/benchmarks/projects/{pid}/runs/{rid2}").status_code == 403
    assert client.delete(f"/api/projects/{pid}/runs/{rid2}").status_code == 404
    assert out2.is_dir() and db.get_run(rid2) is not None, "a foreign run keeps its files"


def test_a_shared_or_foreign_output_dir_is_kept_and_said_so(client, cwd: Path) -> None:
    pid = _project(client)
    rid, out = _run_with_output(cwd, pid, shared=True)
    r = client.delete(f"/api/projects/{pid}/runs/{rid}")
    assert r.status_code == 200, r.text
    assert r.json()["removed_dir"] is None and "shared" in r.json()["kept"]
    assert out.is_dir(), "a directory that is not named after the run may hold other runs' weights"
    assert db.get_run(rid) is None

    # A path outside output/ (``set-output`` accepts any path) is never removed, even when named after the run.
    rid2 = db.create_run(project_id=pid, name="elsewhere", settings_obj={})["id"]
    elsewhere = cwd / "models" / rid2
    elsewhere.mkdir(parents=True)
    db.update_run(rid2, output_path=str(elsewhere))
    r = client.delete(f"/api/projects/{pid}/runs/{rid2}")
    assert r.status_code == 200 and r.json()["removed_dir"] is None and "outside" in r.json()["kept"]
    assert elsewhere.is_dir()

    rid3 = db.create_run(project_id=pid, name="no output", settings_obj={})["id"]
    r = client.delete(f"/api/projects/{pid}/runs/{rid3}")
    assert r.status_code == 200 and r.json()["kept"] == "the run has no output path"


def test_a_run_whose_training_or_export_job_is_alive_is_not_deleted(client, cwd: Path, monkeypatch) -> None:
    pid = _project(client)
    rid, out = _run_with_output(cwd, pid)
    spec = store.JobSpec(id="training-test-live", kind="training", gpu=True, project_id=pid,
                         params={"run_id": rid})
    monkeypatch.setattr(launch, "list_jobs",
                        lambda kind=None, running_only=False: [{"id": spec.id, "kind": "training", "running": True}]
                        if kind == "training" else [])
    store.write_spec(spec)
    r = client.delete(f"/api/projects/{pid}/runs/{rid}")
    assert r.status_code == 409, r.text
    assert r.json()["code"] == "run_busy" and r.json()["job_id"] == spec.id and "training" in r.json()["detail"]
    assert out.is_dir() and db.get_run(rid) is not None

    export = db.create_export(project_id=pid, run_id=rid, format="gguf", quant="Q4_K_M")
    espec = store.JobSpec(id=f"export-{export['id']}", kind="export", gpu=True, project_id=pid,
                          params={"export_id": export["id"]})
    store.write_spec(espec)
    monkeypatch.setattr(launch, "list_jobs",
                        lambda kind=None, running_only=False: [{"id": espec.id, "kind": "export", "running": True}]
                        if kind == "export" else [])
    r = client.delete(f"/api/benchmarks/projects/{pid}/runs/{rid}")
    assert r.status_code == 409 and "export" in r.json()["detail"], r.text
    assert out.is_dir()


def test_run_dir_to_remove_rules(cwd: Path) -> None:
    root = cwd / "output"
    own = root / "projects" / "p1" / "runs" / "r1"
    own.mkdir(parents=True)
    assert run_delete.run_dir_to_remove("r1", str(own)) == (own.resolve(), "")
    assert run_delete.run_dir_to_remove("r1", "output/projects/p1/runs/r1") == (own.resolve(), "")
    assert run_delete.run_dir_to_remove("r2", str(own))[0] is None
    assert run_delete.run_dir_to_remove("r1", str(root))[0] is None, "never the output root itself"
    assert run_delete.run_dir_to_remove("r1", "")[0] is None
    assert run_delete.run_dir_to_remove("r1", str(root / "projects" / "p1" / "runs" / "gone"))[0] is None
    f = own / "file.bin"
    f.write_bytes(b"1")
    assert "not a directory" in run_delete.run_dir_to_remove("file.bin", str(f))[1]
