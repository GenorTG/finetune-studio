"""SFT-merged runs a preference run (DPO/KTO) can start from.

The standard recipe is SFT first, then preference tuning on top of the SFT result
(Zephyr: SFT → DPO; TRL docs). The SFT run's ``merged/`` directory is a plain HF
model, so the preference run simply uses it as its base: with a PEFT adapter the
reference policy is that merged model with the adapter disabled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Routes whose merged output is a sensible preference starting point.
_START_MODES = ("sft", "tool_sft", "reasoning_distillation", "continued_pretraining", "dpo", "kto")


class StartPointError(ValueError):
    """The chosen run cannot be used as a starting model (message is user-facing)."""


def merged_dir(run: dict[str, Any]) -> Path | None:
    """``<output_path>/merged`` when it holds weight files, else ``None``."""
    out = str(run.get("output_path") or "").strip()
    if not out:
        return None
    path = Path(out).expanduser() / "merged"
    if path.is_dir() and any(p.suffix in (".safetensors", ".bin") for p in path.iterdir()):
        return path
    return None


def list_start_points(project_id: str) -> list[dict[str, Any]]:
    """Finished runs of ``project_id`` that left a merged model, newest first."""
    from finetune_studio import db

    points: list[dict[str, Any]] = []
    for run in db.list_runs(project_id):
        if run.get("status") != "done":
            continue
        path = merged_dir(run)
        if path is None:
            continue
        settings = run.get("settings") or {}
        points.append({
            "run_id": run["id"],
            "name": run.get("name") or run["id"],
            "training_mode": settings.get("training_mode", "sft"),
            "final_loss": run.get("final_loss"),
            "merged_path": str(path),
            "base_model": run.get("base_model") or "",
        })
    return points


def resolve_start_point(project_id: str, run_id: str) -> str:
    """Merged-model path of ``run_id``; ``StartPointError`` says exactly why it is unusable."""
    from finetune_studio import db

    run = db.get_run(run_id)
    if not run or run.get("project_id") != project_id:
        raise StartPointError(f"Run {run_id!r} was not found in this project.")
    if run.get("status") != "done":
        raise StartPointError(
            f"Run {run_id!r} has status {run.get('status')!r}; only finished runs can be a starting point."
        )
    path = merged_dir(run)
    if path is None:
        raise StartPointError(
            f"Run {run_id!r} has no merged model (output {run.get('output_path') or '—'}). "
            "Re-run it with 'Also save merged model' ticked, or merge it from the run's export menu."
        )
    return str(path)
