"""Read-only app state for the guide: compact JSON, never mutating.

WHAT THIS FILE DOES
===================
Each public function answers one question the guide asks before it advises:
what exists in this project (``project_overview``), which datasets and runs,
what the machine is doing (``system_status``), what training settings fit the
data (``recommend_training``) and whether a dataset will train
(``dataset_health``). They wrap the same modules the pages use — the preset
advisor, dataset health check, accel/GPU probes — instead of re-deriving numbers.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from finetune_studio import db
from finetune_studio.data.fs import qa as qa_fs

MAX_ROWS = 10
_ROUTES = ("sft", "dpo", "tool_sft", "continued_pretraining", "reasoning_distillation")

# Goal phrases → route. First match wins; order puts the most specific first.
_GOAL_ROUTES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("tool", "function call", "api call", "agent"), "tool_sft",
     "Tool-calling SFT: rows need `messages` with assistant tool_calls + tool replies and a `tools` JSON-schema array."),
    (("preference", "prefer", "chosen", "rejected", "alignment", "rlhf", "ranking", "better answer"), "dpo",
     "DPO: rows need prompt + chosen + rejected for the same prompt, reviewed by a human; at least 2 rows."),
    (("raw text", "unlabeled", "unlabelled", "corpus", "domain language", "pretrain", "continued"), "continued_pretraining",
     "Continued pretraining: one {\"text\": …} row per sample; not instruction tuning, no QA quiz."),
    (("reasoning", "chain of thought", "cot", "distill", "teacher"), "reasoning_distillation",
     "Reasoning distillation: reviewed teacher demonstrations as messages JSONL with a verifiable final answer."),
    (("cite", "citation", "changing", "up to date", "fresh", "source"), "rag",
     "RAG (not training): knowledge that changes or needs citations belongs in a RAG index on the RAG page."),
)


def _project_or_error(pid: str | None) -> tuple[dict | None, dict | None]:
    if not pid:
        return None, {"error": "no active project; open a project first (navigate to 'projects')"}
    project = db.get_project(pid)
    if not project:
        return None, {"error": f"project {pid} not found"}
    return project, None


def readiness(pid: str) -> dict[str, Any]:
    """Authoritative counts for one project; ``summary`` is quoted verbatim in replies."""
    project = db.get_project(pid)
    if not project:
        return {"error": "project not found"}
    sources = qa_fs.list_qa_sources(pid)
    pairs = qa_fs.list_qa_pairs(pid)
    counts = {status: sum(1 for pair in pairs if pair.get("status", "pending") == status)
              for status in ("pending", "approved", "rejected")}
    datasets = db.list_datasets(pid)
    rags = db.list_rags(pid)
    parsed_sources = sum(1 for source in sources if int(source.get("chunk_count") or 0) > 0)
    next_step = (
        "Review pending Q&A pairs." if counts["pending"] else
        "Export approved Q&A pairs." if counts["approved"] and not datasets else
        "Choose SFT or DPO on Training based on the data you have." if datasets else
        "Upload and parse source files."
    )
    summary = (
        f"{len(sources)} source(s), {parsed_sources} parsed; {counts['approved']} approved, "
        f"{counts['pending']} pending, and {counts['rejected']} rejected Q&A pair(s); "
        f"{len(datasets)} dataset(s); {len(rags)} RAG corpus/corpora. Next: {next_step}"
    )
    return {
        "summary": summary,
        "project": {"id": pid, "name": project.get("name"), "base_model": project.get("base_model")},
        "sources": {"count": len(sources), "parsed": parsed_sources},
        "qa_pairs": {"total": len(pairs), **counts},
        "datasets": [{"name": ds.get("name"), "rows": ds.get("qa_count", 0)} for ds in datasets],
        "rag_corpora": [{"name": rag.get("name"), "chunks": rag.get("chunk_count", 0)} for rag in rags],
        "next_step": next_step,
    }


def _short(path: str) -> str:
    return Path(path).name if path else ""


def _run_row(run: dict, production: str | None) -> dict[str, Any]:
    settings = run.get("settings") if isinstance(run.get("settings"), dict) else {}
    metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    return {
        "id": run.get("id"),
        "name": run.get("name"),
        "status": run.get("status"),
        "base_model": _short(str(run.get("base_model") or "")),
        "training_mode": settings.get("training_mode", "sft"),
        "epochs": settings.get("num_epochs"),
        "final_loss": run.get("final_loss"),
        "total_steps": metrics.get("total_steps"),
        "has_output": bool(run.get("output_path")),
        "production": bool(production) and run.get("id") == production,
        "error": (str(run.get("error"))[:160] if run.get("error") else None),
    }


def list_runs(pid: str | None) -> dict[str, Any]:
    project, err = _project_or_error(pid)
    if err:
        return err
    assert project is not None and pid is not None
    runs = db.list_runs(pid)
    production = project.get("production_run")
    return {
        "count": len(runs),
        "production_run": production or None,
        "runs": [_run_row(r, production) for r in runs[:MAX_ROWS]],
    }


def list_datasets(pid: str | None) -> dict[str, Any]:
    _, err = _project_or_error(pid)
    if err:
        return err
    assert pid is not None
    datasets = db.list_datasets(pid)
    return {
        "count": len(datasets),
        "datasets": [
            {
                "id": ds.get("id"),
                "name": ds.get("name"),
                "rows": ds.get("qa_count", 0),
                "size_kb": round((ds.get("size_bytes") or 0) / 1024, 1),
                "source": ds.get("source") or "dataset",
            }
            for ds in datasets[:MAX_ROWS]
        ],
    }


def project_overview(pid: str | None) -> dict[str, Any]:
    """Readiness plus runs, production run and RAG detail; with no project, the project list."""
    if not pid:
        projects = db.list_projects()
        return {
            "project": None,
            "projects": [{"id": p.get("id"), "name": p.get("name")} for p in projects[:MAX_ROWS]],
            "count": len(projects),
            "next_step": "Open a project, or create one on the Projects page." if projects else "Create a project on the Projects page.",
        }
    base = readiness(pid)
    if base.get("error"):
        return base
    runs = list_runs(pid)
    project = db.get_project(pid) or {}
    return {
        **base,
        "production_run": runs.get("production_run"),
        "runs": {"count": runs["count"], "latest": runs["runs"][:3]},
        "system_prompt_set": bool(project.get("system_prompt")),
    }


def dataset_health(pid: str | None, dataset_id: str | None = None, training_mode: str | None = None) -> dict[str, Any]:
    """Will this dataset train? Wraps the Training page's dataset check for the chosen route."""
    from finetune_studio.data.dataset_health import check_dataset

    _, err = _project_or_error(pid)
    if err:
        return err
    assert pid is not None
    mode = (training_mode or "sft").strip()
    if mode not in _ROUTES:
        return {"error": f"training_mode must be one of {list(_ROUTES)}"}
    datasets = db.list_datasets(pid)
    if not datasets:
        return {"error": "no datasets in this project yet; export approved Q&A pairs on the Pairs page or upload one on Training",
                "qa_audit": _qa_audit(pid)}
    ds = next((d for d in datasets if d.get("id") == dataset_id), None) if dataset_id else datasets[0]
    if ds is None:
        return {"error": f"dataset {dataset_id!r} not found", "known": [d.get("id") for d in datasets[:MAX_ROWS]]}
    path = Path(str(ds.get("data_path") or ""))
    if not path.is_file():
        return {"error": f"dataset file is missing: {path.name}"}
    report = check_dataset(path, training_mode=mode)
    return {
        "dataset": {"id": ds.get("id"), "name": ds.get("name"), "rows": ds.get("qa_count", 0)},
        "training_mode": mode,
        "examples": report.get("examples"),
        "trainable": report.get("trainable"),
        "holdout": report.get("holdout"),
        "stats": report.get("stats"),
        "issues": [
            {k: i.get(k) for k in ("code", "severity", "title", "detail", "count")}
            for i in (report.get("issues") or [])[:MAX_ROWS]
        ],
        "qa_audit": _qa_audit(pid),
    }


def _qa_audit(pid: str) -> dict[str, Any]:
    from finetune_studio.data.audit import audit_qa_pairs

    audit = audit_qa_pairs(pid)
    uncovered = sum(len(row["uncovered_chunks"]) for row in audit["source_coverage"].values())
    return {
        "status": audit["status"],
        "approved_pairs": audit["approved_pair_count"],
        "duplicates_collapsed": audit["duplicate_pairs_collapsed"],
        "sources": audit["source_count"],
        "uncovered_chunks": uncovered,
        "errors": len(audit["errors"]),
    }


def recommend_training(
    pid: str | None,
    goal: str = "",
    tier: str = "",
    base_model: str = "",
    dataset_id: str = "",
) -> dict[str, Any]:
    """Settings the preset advisor would propose for this project's data, with the reasoning."""
    from finetune_studio.training.preset_advisor import _TIER_ANCHORS, propose

    project, err = _project_or_error(pid)
    if err:
        return err
    assert project is not None and pid is not None
    goal_text = (goal or "").lower()
    route, route_why = "sft", "SFT on Q&A pairs from your documents is the default route."
    for words, candidate, why in _GOAL_ROUTES:
        if any(w in goal_text for w in words):
            route, route_why = candidate, why
            break
    if route == "rag":
        return {"route": "rag", "why": route_why, "page": "rag",
                "note": "No training settings apply; build the index with Quick index on the RAG page."}

    datasets = db.list_datasets(pid)
    ds = next((d for d in datasets if d.get("id") == dataset_id), None) if dataset_id else (datasets[0] if datasets else None)
    base = base_model or str(project.get("base_model") or "")
    pairs = int(ds.get("qa_count") or 0) if ds else 0
    chosen_tier = tier if tier in _TIER_ANCHORS else "balanced"
    adv = propose(
        tier=chosen_tier, base_model_ref=base,
        dataset_path=str(ds.get("data_path")) if ds and ds.get("data_path") else None,
        pair_count_hint=pairs or None,
    )
    alternatives = []
    for name in _TIER_ANCHORS:
        if name == chosen_tier:
            continue
        other = propose(tier=name, base_model_ref=base, pair_count_hint=adv.pair_count)
        alternatives.append({"tier": name, "epochs": other.num_epochs, "rank": other.lora_rank,
                             "optimizer_steps": other.optimizer_steps, "warnings": len(other.warnings)})
    too_few = chosen_tier != "smoke" and adv.optimizer_steps < adv.steps_floor
    why = list(adv.notes)
    if route != "sft":
        why.insert(0, route_why)
    settings: dict[str, Any] = {
        "num_epochs": adv.num_epochs, "lora_rank": adv.lora_rank, "learning_rate": adv.learning_rate,
        "batch_size": adv.batch_size, "gradient_accumulation_steps": adv.gradient_accumulation_steps,
        "warmup_steps": adv.warmup_steps,
    }
    if route == "dpo":
        settings.update({"learning_rate": "1e-6", "num_epochs": 1, "warmup_steps": 0})
        why.append("DPO overrides learning rate 1e-6, 1 epoch, no warmup (the Training page applies the same defaults).")
    return {
        "route": route,
        "route_why": route_why,
        "tier": chosen_tier,
        "dataset": {"id": ds.get("id"), "name": ds.get("name"), "rows": adv.pair_count} if ds else None,
        "base_model": _short(base) or None,
        "base_params_b": adv.base_params_b,
        "settings": settings,
        "optimizer_steps": adv.optimizer_steps,
        "steps_floor": adv.steps_floor,
        "too_few_steps": too_few,
        "warnings": adv.warnings + ([] if ds else ["No dataset in this project yet — sized for the ~500-pair reference case."]),
        "why": why,
        "alternatives": alternatives,
        "caveat": "optimizer_steps is an upper bound: the advisor ignores the validation split, so the run's real total steps are lower.",
        "next": "Use suggest_settings on the training page to prefill these values; the user presses Start.",
    }


def _helper_state() -> dict[str, Any]:
    try:
        from finetune_studio.models.helper import (
            DEFAULT_HELPER_PROVIDER_ID,
            get_configured_helper_provider,
            missing_gguf_for_provider,
        )
        helper = get_configured_helper_provider() or {}
        return {
            "configured": bool(helper),
            "file": _short(str(helper.get("model_id") or "")),
            "missing": bool(missing_gguf_for_provider(DEFAULT_HELPER_PROVIDER_ID)),
        }
    except Exception as exc:  # noqa: BLE001 - status must degrade, not fail
        return {"configured": False, "error": str(exc)[:120]}


def system_status() -> dict[str, Any]:
    """GPU/VRAM, loaded model and placement, helper, training state, RAM and disk."""
    from finetune_studio.webui.gpu_probe import vram_devices
    from finetune_studio.webui.routes.system import _accel_summary, _ram

    out: dict[str, Any] = {
        "accelerator": _accel_summary(),
        "gpus": [{"index": d.get("index"), "name": d.get("name"), "used_gb": d.get("used_gb"),
                  "total_gb": d.get("total_gb")} for d in vram_devices()],
        "ram": _ram(),
        "helper": _helper_state(),
    }
    try:
        from finetune_studio.webui.app import inference_engine, training_engine
        loaded = inference_engine.model is not None
        out["loaded_model"] = {
            "loaded": loaded,
            "name": _short(str(inference_engine.model_path or "")) if loaded else None,
            "n_ctx": getattr(inference_engine, "n_ctx", None) if loaded else None,
            "n_gpu_layers": getattr(inference_engine, "n_gpu_layers", None) if loaded else None,
            "offload": dict(getattr(inference_engine, "offload", None) or {}) if loaded else {},
        }
        out["training"] = {"status": training_engine.state.status, "step": training_engine.state.current_step,
                           "total_steps": training_engine.state.total_steps}
    except Exception as exc:  # noqa: BLE001
        out["engines_error"] = str(exc)[:120]
    try:
        usage = shutil.disk_usage(Path.cwd())
        out["disk"] = {"free_gb": round(usage.free / 1024**3, 1), "total_gb": round(usage.total / 1024**3, 1)}
    except OSError:
        out["disk"] = None
    return out
