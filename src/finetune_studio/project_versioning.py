"""Project snapshot creation shared by the WebUI and local CLI."""
from __future__ import annotations

from typing import Any

from finetune_studio import db


class ProjectVersionError(ValueError):
    """A version-save request that can be reported by either adapter."""

    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def save_project_version(pid: str, body: dict[str, Any]) -> dict[str, Any]:
    """Pin a project snapshot, filling the same current inputs as Quick Work."""
    if not db.get_project(pid):
        raise ProjectVersionError("project not found", status_code=404)

    parent_id = body.get("parent_version_id")
    if parent_id:
        parent = db.get_version(str(parent_id))
        if not parent or parent.get("project_id") != pid:
            raise ProjectVersionError(f"parent version {parent_id!r} not found", status_code=404)

    manifest = body.get("manifest") or {}
    if not isinstance(manifest, dict):
        raise ProjectVersionError("manifest must be an object", status_code=400)

    # Auto-fill runnable inputs when the caller does not pin them explicitly.
    if not manifest.get("datasets"):
        manifest["datasets"] = [
            {"id": dataset["id"], "name": dataset["name"], "data_path": dataset["data_path"],
             "qa_count": dataset.get("qa_count", 0)}
            for dataset in db.list_datasets(pid)
        ]
    if not manifest.get("brain_builds"):
        latest_build = db.brain_builds.latest_for_project(pid)
        if latest_build:
            manifest["brain_builds"] = [{
                "build_id": latest_build["id"], "index_id": latest_build.get("index_id", ""),
                "doc_count": latest_build.get("doc_count", 0),
                "chunk_count": latest_build.get("chunk_count", 0),
            }]
    if not manifest.get("training_runs"):
        from finetune_studio.db.runs import list_runs

        runs = [run for run in list_runs(pid)
                if run.get("status") in ("done", "completed", "completed_error")][:5]
        manifest["training_runs"] = [
            {"run_id": run["id"], "status": run.get("status", ""),
             "final_loss": run.get("final_loss"), "output_path": run.get("output_path", "")}
            for run in runs
        ]

    return db.create_version(
        pid,
        label=str(body.get("label") or ""),
        notes=str(body.get("notes") or ""),
        parent_version_id=str(parent_id) if parent_id else None,
        manifest=manifest,
    )
