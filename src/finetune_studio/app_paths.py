"""Where the app's relative stores live, independent of the caller's working directory.

``settings.db_path`` (``data/finetune_studio.db``) and the training ``output/`` tree are
stored relative to the checkout the service runs in (its systemd WorkingDirectory).
``fts`` started from any other directory used to resolve them against *its* cwd: it
opened (and created) a second, empty database there and deleted ``<cwd>/output``.

Imports nothing heavy on purpose: :func:`anchor_db_path` must run before
``finetune_studio.db`` is imported, because that import already opens the database.
"""
from __future__ import annotations

from pathlib import Path

from finetune_studio.version import REPO_ROOT

#: The checkout (editable install); for a wheel install, the directory the process started in.
APP_ROOT: Path = REPO_ROOT if (REPO_ROOT / "pyproject.toml").is_file() else Path.cwd()


def app_path(path: str | Path) -> Path:
    """``path`` as stored by the app, resolved against :data:`APP_ROOT` when relative."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else APP_ROOT / p


def anchor_db_path() -> None:
    """Make a relative ``settings.db_path`` absolute under :data:`APP_ROOT` (``:memory:`` stays)."""
    from finetune_studio.config import settings

    if settings.db_path != ":memory:":
        settings.db_path = str(app_path(settings.db_path))
