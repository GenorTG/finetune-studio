"""Resolve project-scoped models for the Testing tab.

Keeps inference auto-load and the model selector aligned with on-disk
exports under each run's ``output_path`` (merged/, gguf/, …), not the
global HF discovery list.
"""

from __future__ import annotations

import os
from typing import Any

from finetune_studio.training.run_export import merged_dir_ready

# Status values written by the training engine / older callers.
_DONE_STATUSES = frozenset({"done", "completed"})

# Formats the Testing inference engine can load (HF dir or GGUF).
_INFERENCE_FORMATS = frozenset({"safetensors", "gguf"})

BASE_MODEL_CHOICE = "__base__"   # the Testing / Compare pages' "untrained base model" entry
API_READER_PREFIX = "api:"       # "api:<connection id>": a Settings → AI assistant API connection reads (no GPU)


def is_api_reader(choice: str) -> bool:
    return str(choice or "").startswith(API_READER_PREFIX)


def api_readers() -> list[dict[str, Any]]:
    """The API connections (Settings → AI assistant) a Smart Brain test can use as its reader: no GPU, no download.
    Without them a user with no GPU could not test the package's answers at all."""
    from finetune_studio.models.assistant import connection_label
    from finetune_studio.models.manager import get_manager

    out = []
    for row in get_manager().list_providers():
        if row.get("kind") != "openai_compat":
            continue
        out.append({"value": API_READER_PREFIX + str(row["id"]), "id": row["id"],
                    "label": connection_label(row), "model": row.get("model_id") or ""})
    return out


def default_reader_for_testing(models: list[dict[str, Any]], readers: list[dict[str, Any]]) -> str | None:
    """No export of this project yet: the AI assistant's API connection (the helper seat) when it is one, else the
    first API connection. Never "latest merged export" in a project that has none."""
    if models:
        return default_model_path_for_testing(models)
    from finetune_studio.models.helper import get_helper_provider_id

    seat = API_READER_PREFIX + str(get_helper_provider_id("quiz") or "")
    for r in readers:
        if r["value"] == seat:
            return seat
    return readers[0]["value"] if readers else None


def local_model_missing(path: str) -> bool:
    """True when ``path`` names a local file/dir that is not on disk.

    Hub ids (``org/name``) and empty values are not local paths, so they are never "missing".
    """
    p = (path or "").strip()
    return p.startswith(("/", "~", "./", "../")) and not os.path.exists(os.path.expanduser(p))


def unloadable_model_dir(path: str) -> bool:
    """True when ``path`` is an existing local folder no loader can open (no config.json, adapter or .gguf).

    A training run's own folder (``adapter/`` + ``checkpoint-*``) is the usual case: it has to be exported first.
    """
    p = os.path.expanduser((path or "").strip())
    if not os.path.isdir(p):
        return False
    try:
        names = os.listdir(p)
    except OSError:
        return False
    return not ({"config.json", "adapter_config.json"} & set(names) or any(n.endswith(".gguf") for n in names))


def is_run_done(run: dict[str, Any]) -> bool:
    """True when a training run is finished successfully."""
    return (run.get("status") or "").lower() in _DONE_STATUSES


def resolve_latest_merged_model(pid: str, *, list_runs_fn: Any = None) -> str | None:
    """Return path to the newest completed run's ready ``merged/`` dir.

    Skips runs with empty ``output_path`` or without weight files under
    ``merged/``. ``list_runs_fn`` defaults to ``db.list_runs`` (injectable
    for tests).
    """
    if list_runs_fn is None:
        from finetune_studio import db

        list_runs_fn = db.list_runs

    runs = list_runs_fn(pid)
    for run in runs:  # newest-first
        if not is_run_done(run):
            continue
        output_path = (run.get("output_path") or "").strip()
        if not output_path:
            continue
        if not merged_dir_ready(output_path):
            continue
        return os.path.join(output_path, "merged")
    return None


def models_for_testing_page(project_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter + sort project export rows for the Testing model selector.

    Prefer safetensors (merged/abliterated), then gguf. Newest mtime first.
    """
    usable = [
        m
        for m in project_models
        if (m.get("format") or "").lower() in _INFERENCE_FORMATS
        and (m.get("path") or "").strip()
    ]
    usable.sort(key=lambda m: float(m.get("mtime") or 0), reverse=True)
    return usable


def library_models_for_testing(project_models: list[dict[str, Any]], pid: str = "",
                               discovered: list | None = None) -> list[dict[str, Any]]:
    """Installed Model-library models (GGUF / safetensors) that are not already this project's exports.

    A Smart-Brain-only project has no trained export, and the picker listed nothing but "latest merged export":
    no reader could be tested at all (2026-10-21). Vision projectors (mmproj) are not models. Other projects'
    trained exports are not installed models (the list showed seven of another project's runs), and one model found
    twice (``models/Qwen3-0.6B`` and the Hugging Face cache's ``Qwen/Qwen3-0.6B``) is listed once.
    """
    from pathlib import Path

    if discovered is None:
        from finetune_studio.webui.app import discovered_models as discovered

    taken = {str(m.get("path") or "").rstrip("/") for m in project_models}
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for m in discovered:
        path = str(getattr(m, "path", "") or "").rstrip("/")
        fmt = str(getattr(m, "format", "") or "").lower()
        if not path or path in taken or fmt not in _INFERENCE_FORMATS or Path(path).name.lower().startswith("mmproj"):
            continue
        owner = str(getattr(m, "project_id", "") or "")
        if owner and owner != pid:
            continue
        name = str(getattr(m, "name", "") or Path(path).name)
        key = (name.rsplit("/", 1)[-1].strip().lower(), fmt)
        real = os.path.realpath(path)
        if key in seen or (real, fmt) in seen:
            continue
        seen.update({key, (real, fmt)})
        out.append({"path": path, "name": name.rsplit("/", 1)[-1] if "/" in name else name, "format": fmt,
                    "size_gb": getattr(m, "size_gb", None)})
    return sorted(out, key=lambda m: m["name"].lower())


def default_model_path_for_testing(models: list[dict[str, Any]]) -> str | None:
    """Path of the preferred default selection (latest merged safetensors)."""
    for m in models:
        path = (m.get("path") or "").rstrip("/\\")
        if (m.get("format") or "").lower() == "safetensors" and path.endswith(
            "merged"
        ):
            return m.get("path")
    return models[0]["path"] if models else None


def target_model(pid: str, override: str) -> str:
    """The model a test run will use: the explicit choice, the project's untrained base, else the latest merged export.

    Never "whatever happens to be loaded": the helper or another page's model may be resident.
    A local path that is no longer on disk is refused here, before a run is created.
    """
    if is_api_reader(override):
        from finetune_studio.models.manager import get_manager

        row = get_manager().get_provider(override[len(API_READER_PREFIX):])
        if row is None or row.get("kind") != "openai_compat":
            raise ValueError(f"{override[len(API_READER_PREFIX):]!r} is not an API connection (Settings → AI assistant)")
        return override
    if override == BASE_MODEL_CHOICE:
        from finetune_studio import db
        from finetune_studio.db.runs import backfill_project_base_model

        base = backfill_project_base_model(pid) or str((db.get_project(pid) or {}).get("base_model") or "")
        if not base.strip():
            raise ValueError("this project has no base model recorded yet; train a run or set one under project Settings")
        chosen = base.strip()
    elif override:
        chosen = override
    else:
        chosen = resolve_latest_merged_model(pid) or ""
        if not chosen:
            raise ValueError("no model to test: pick one in the Model list, or run training + merge/export first")
    if local_model_missing(chosen):
        raise ValueError(f"model not found on disk: {chosen} (re-create it, or pick another model)")
    if override and override != BASE_MODEL_CHOICE and unloadable_model_dir(chosen):
        raise ValueError(f"not a model folder: {chosen} has no config.json, adapter or .gguf file (a training run's "
                         "folder? export the run as merged or GGUF on the Export page and test that)")
    return chosen
