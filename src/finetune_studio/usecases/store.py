"""A use case in a project: its SFT dataset, its held-out suite and a manifest that ties them together.

Layout (``data.fs.paths.project_dir``):

* ``datasets/usecase-<slug>.jsonl`` — the training rows, registered as a project dataset (Training page picks it);
* ``suites/usecase-<slug>.json``    — ``{"usecase": {...}, "cases": [...]}``, the held-out questions (Testing and
  Compare list it; the scoring hook reads the ``usecase`` block);
* ``usecases/<id>.json``           — the manifest: template, instruction, labels / schema, counts, training plan, and
  every base-vs-run evaluation started for it.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from finetune_studio.usecases.rows import Prepared
from finetune_studio.usecases.templates import get_template

_LOCK = threading.Lock()
_ID = re.compile(r"^[0-9a-f]{8}$")


def usecases_dir(pid: str) -> Path:
    from finetune_studio.data.fs.paths import project_dir

    return project_dir(pid) / "usecases"


def _manifest_path(pid: str, uid: str) -> Path:
    if not _ID.match(uid or ""):
        raise LookupError("use case not found")
    return usecases_dir(pid) / f"{uid}.json"


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def suite_meta(prepared: Prepared, uid: str, name: str) -> dict[str, Any]:
    """The ``usecase`` block of the held-out suite: everything a scorer needs, nothing else."""
    return {"id": uid, "template": prepared.template_id, "name": name, "labels": prepared.labels,
            "schema": prepared.schema, "style_guide": prepared.style_guide, "teacher": prepared.teacher}


def create(pid: str, name: str, prepared: Prepared, *, teacher_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the dataset, the held-out suite and the manifest. Returns the manifest."""
    from finetune_studio import db
    from finetune_studio.data.prep.dataset_build import register_dataset_file
    from finetune_studio.db.connection import new_id
    from finetune_studio.testing.suite_import import suites_dir

    template = get_template(prepared.template_id)
    uid = new_id()
    name = name.strip() or f"{template.title} {time.strftime('%Y-%m-%d')}"
    slug = (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or template.id) + f"-{uid[:6]}"
    project = (db.get_project(pid) or {}).get("name") or pid
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prepared.train_rows)
    persisted = register_dataset_file(pid, f"usecase-{slug}.jsonl", body,
                                      f"{project} · {template.title} · {name} · {{rows}} rows", source="usecase")
    suite_path = suites_dir(pid) / f"usecase-{slug}.json"
    _write(suite_path, {"usecase": suite_meta(prepared, uid, name), "cases": prepared.heldout_cases})
    manifest = {
        "id": uid, "project_id": pid, "template": template.id, "name": name, "created_at": time.time(),
        **prepared.summary(),
        "schema": prepared.schema, "style_guide": prepared.style_guide, "teacher": prepared.teacher,
        "seed": prepared.seed, "teacher_report": teacher_report,
        "dataset": {"id": persisted.dataset.get("id"), "name": persisted.dataset.get("name"),
                    "path": str(persisted.path), "rows": persisted.rows},
        "suite": {"path": str(suite_path), "name": suite_path.name, "cases": len(prepared.heldout_cases)},
        "evaluations": [],
    }
    with _LOCK:
        _write(_manifest_path(pid, uid), manifest)
    return manifest


def get(pid: str, uid: str) -> dict[str, Any]:
    """The manifest; ``LookupError`` when the project has no such use case."""
    path = _manifest_path(pid, uid)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LookupError("use case not found") from None


def list_all(pid: str) -> list[dict[str, Any]]:
    """Every use case of the project, newest first (manifests that cannot be read are skipped)."""
    d = usecases_dir(pid)
    out = []
    for f in d.glob("*.json") if d.is_dir() else []:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda m: m.get("created_at") or 0, reverse=True)
    return out


def add_evaluation(pid: str, uid: str, record: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        manifest = get(pid, uid)
        manifest.setdefault("evaluations", []).append(record)
        _write(_manifest_path(pid, uid), manifest)
    return manifest


def delete(pid: str, uid: str) -> dict[str, Any]:
    """Remove the manifest, the held-out suite and the A/B results; the dataset and trained runs are kept."""
    import shutil

    with _LOCK:
        manifest = get(pid, uid)
        suite = Path(str((manifest.get("suite") or {}).get("path") or ""))
        if suite.name.startswith("usecase-") and suite.is_file():
            suite.unlink()
        shutil.rmtree(usecases_dir(pid) / uid, ignore_errors=True)
        _manifest_path(pid, uid).unlink(missing_ok=True)
    return {"ok": True, "kept_dataset": (manifest.get("dataset") or {}).get("id")}


def ab_path(pid: str, uid: str, group_id: str) -> Path:
    if not re.match(r"^[0-9a-f]{8}$", group_id or ""):
        raise LookupError("evaluation not found")
    return usecases_dir(pid) / uid / f"ab-{group_id}.json"


def write_ab(pid: str, uid: str, group_id: str, data: dict[str, Any]) -> None:
    _write(ab_path(pid, uid, group_id), data)


def read_ab(pid: str, uid: str, group_id: str) -> dict[str, Any] | None:
    try:
        return json.loads(ab_path(pid, uid, group_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def suite_usecase(suite_path: str) -> dict[str, Any] | None:
    """The ``usecase`` block of a suite file, or None (not a use-case suite, unreadable)."""
    if not suite_path or not suite_path.endswith(".json"):
        return None
    try:
        data = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    meta = data.get("usecase") if isinstance(data, dict) else None
    return meta if isinstance(meta, dict) and meta.get("template") else None
