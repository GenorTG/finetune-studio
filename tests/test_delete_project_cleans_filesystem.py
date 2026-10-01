"""Regression test (2026-10-01 audit fix): DELETE /api/projects/{pid} only
dropped the DB row and left every on-disk tree behind — the file library,
QA pairs, and RAG corpus under ~/.finetune-studio/projects/<pid>/, the
project's datasets under data/projects/<pid>/, and every training run's
merged weights + GGUF exports under output/projects/<pid>/. The project
disappeared from the UI (its DB row was gone) while claiming success, but
all the actual bytes stayed on disk forever — found during a 121GB cleanup
pass where three "deleted" projects were all still fully present on disk.
"""

from __future__ import annotations

from pathlib import Path


class TestDeleteProjectCleansFilesystem:
    def test_delete_removes_all_three_on_disk_trees(
        self, client, mock_settings, monkeypatch, tmp_path,
    ):
        from finetune_studio import db
        from finetune_studio.data.fs import paths as fts_paths

        fts_root = tmp_path / "fts-home"
        monkeypatch.setattr(fts_paths, "_ROOT", fts_root)
        monkeypatch.setattr(fts_paths, "_PROJECTS", fts_root / "projects")

        pid = db.create_project(name="P", description="")["id"]

        fts_project_dir = fts_root / "projects" / pid / "files" / "raw"
        fts_project_dir.mkdir(parents=True)
        (fts_project_dir / "original.txt").write_text("raw upload")

        data_project_dir = Path(mock_settings.db_path).parent / "projects" / pid / "datasets"
        data_project_dir.mkdir(parents=True)
        (data_project_dir / "train.jsonl").write_text("{}")

        output_project_dir = Path("output") / "projects" / pid / "runs" / "rid1" / "gguf"
        output_project_dir.mkdir(parents=True)
        (output_project_dir / "model-f16.gguf").write_bytes(b"\x00" * 10)

        try:
            assert fts_root.joinpath("projects", pid).exists()
            assert data_project_dir.exists()
            assert output_project_dir.exists()

            r = client.delete(f"/api/projects/{pid}")
            assert r.status_code == 200
            assert r.json() == {"ok": True}

            assert not fts_root.joinpath("projects", pid).exists(), (
                "~/.finetune-studio/projects/<pid>/ must be removed"
            )
            assert not (Path(mock_settings.db_path).parent / "projects" / pid).exists(), (
                "data/projects/<pid>/ must be removed"
            )
            assert not (Path("output") / "projects" / pid).exists(), (
                "output/projects/<pid>/ must be removed"
            )
            assert db.get_project(pid) is None
        finally:
            import shutil
            shutil.rmtree(Path("output") / "projects" / pid, ignore_errors=True)
