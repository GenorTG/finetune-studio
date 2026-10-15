"""Tests must never write project dirs into the real ~/.finetune-studio."""
from __future__ import annotations

from pathlib import Path

from finetune_studio.data.fs import paths


def test_fts_root_is_per_test_temp_dir() -> None:
    real = Path.home() / ".finetune-studio"
    assert real not in paths.root().parents and paths.root() != real
    assert paths.project_dir("abc12345").is_relative_to(paths.root())


def test_upload_route_does_not_touch_real_root(client) -> None:
    from finetune_studio import db

    real = Path.home() / ".finetune-studio" / "projects"
    before = set(real.iterdir()) if real.exists() else set()
    pid = db.create_project(name="iso", description="")["id"]
    r = client.post(f"/api/projects/{pid}/files/upload", files={"files": ("a.txt", b"hello world", "text/plain")})
    assert r.status_code == 200
    after = set(real.iterdir()) if real.exists() else set()
    assert after == before
