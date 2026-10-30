"""Safe operations on the app-managed Hugging Face model cache."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from finetune_studio.webui.routes.hf_models import _LOCAL, _local_dir

log = logging.getLogger(__name__)


class HFCacheError(ValueError):
    """A cache operation could not be completed safely."""

    def __init__(self, detail: str, *, status_code: int = 409) -> None:
        super().__init__(detail)
        self.status_code = status_code


def delete_cached_model(repo_id: str) -> Path:
    """Delete one downloaded repo, refusing a model that is loaded or loading."""
    from fastapi import HTTPException

    from finetune_studio import db

    try:
        target = _local_dir(repo_id)
    except HTTPException as exc:
        raise HFCacheError(str(exc.detail), status_code=exc.status_code) from exc
    if not target.exists():
        raise HFCacheError(f"downloaded model not found: {repo_id}", status_code=404)

    active = any(
        row.get("repo_id") == repo_id
        for row in db.list_hf_downloads_in_progress(limit=10000)
    )
    if active:
        raise HFCacheError(f"cannot delete {repo_id}: its download is still running")

    # The inference worker is separate from the CLI and WebUI process. Read its
    # status through the same socket-backed engine abstraction before deleting.
    from finetune_studio.inference.remote import RemoteEngine

    engine = RemoteEngine()
    try:
        status = engine._status(fresh=True)
    except Exception as exc:  # fail closed: never delete if usage is unknowable
        raise HFCacheError(f"could not check whether a model is in use: {exc}") from exc
    for key, description in (("model_path", "loaded"), ("loading_path", "loading")):
        value = status.get(key)
        if not value:
            continue
        try:
            Path(value).resolve().relative_to(_LOCAL.resolve())
        except (OSError, ValueError):
            continue
        if Path(value).resolve() == target.resolve() or target.resolve() in Path(value).resolve().parents:
            raise HFCacheError(f"cannot delete {repo_id}: it is currently {description}; unload it first")

    shutil.rmtree(target)
    from finetune_studio.models.registry_invalidation import invalidate_model_registry

    try:
        invalidate_model_registry()
    except Exception:  # the deletion already succeeded; a rescan can recover the registry later
        log.warning("model registry invalidation after HF cache deletion failed", exc_info=True)
    return target
