"""The suite must never write ``data/projects/<id>`` (or any project dir) into the repo cwd.

``db.datasets.datasets_dir`` and ``data.fs.paths.project_roots`` resolve
``<db dir>/projects/<pid>`` through ``finetune_studio.db.datasets.settings`` — a
SEPARATE ``from config import settings`` binding that the autouse ``temp_db``
fixture used to leave pointing at the real ``data/finetune_studio.db``. Every
test that exported or registered a dataset therefore left a ``data/projects/<id>``
dir behind (~250 after a few weeks).
"""
from __future__ import annotations

from pathlib import Path

import finetune_studio.config as cfg
from finetune_studio.db import connection as conn
from finetune_studio.db import datasets as ds

REPO = Path(__file__).resolve().parent.parent


def test_datasets_dir_resolves_under_the_temp_db_not_the_repo() -> None:
    d = ds.datasets_dir("leakprobe")
    assert Path(cfg.settings.db_path).parent in d.parents
    assert REPO not in d.resolve().parents, f"datasets_dir leaked into the repo: {d}"
    assert ds.settings.db_path == conn.settings.db_path == cfg.settings.db_path


def test_no_module_binds_the_real_db_path(temp_db: str) -> None:
    from finetune_studio.data.fs import paths

    roots = paths.project_roots("leakprobe2")
    assert all(REPO not in r.resolve().parents for r in roots), roots
    assert not (REPO / "data" / "projects" / "leakprobe").exists()
    assert not (REPO / "data" / "projects" / "leakprobe2").exists()
