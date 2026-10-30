"""Shared lifecycle operations for project records and their stored data."""
from __future__ import annotations

import shutil
from pathlib import Path

from finetune_studio import db


def delete_project(pid: str) -> bool:
    """Delete a project record and every on-disk tree owned by that project.

    Returns ``False`` when the project does not exist. Model-registry refreshes
    belong to the calling adapter because the registry is a WebUI concern.
    """
    if db.get_project(pid) is None:
        return False

    from finetune_studio.config import settings
    from finetune_studio.data.fs.paths import brain_index_dir
    from finetune_studio.data.fs.paths import root as fts_root

    db.delete_project(pid)
    for project_dir in (
        fts_root() / "projects" / pid,
        Path(settings.db_path).parent / "projects" / pid,
        Path("output") / "projects" / pid,
        brain_index_dir(pid),
    ):
        shutil.rmtree(project_dir, ignore_errors=True)
    return True
