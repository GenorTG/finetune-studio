"""Training tab — start/stop training, monitor progress."""

import asyncio
import json
import logging
import math
import re
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from finetune_studio.jobs.launch import JobRejected
from finetune_studio.training.data import (
    format_for_preference,
    format_for_unpaired,
    load_jsonl,
)
from finetune_studio.training.engine import TrainingConfig
from finetune_studio.training.monitor import training_events
from finetune_studio.training.remote import TrainingBusy
from finetune_studio.webui.app import training_engine
from finetune_studio.webui.live_sse import sse_response

log = logging.getLogger(__name__)
router = APIRouter()


def _coerce_bool(value: object) -> bool:
    """Parse JSON/FormData bool-ish values (``"1"``, ``"true"``, ``true``, …)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return bool(value)


def _optional_body_bool(body: dict, key: str, overrides: dict | None = None) -> bool | None:
    """Return coerced bool when ``key`` is present on body or overrides; else None."""
    if key in body:
        return _coerce_bool(body[key])
    if overrides is not None and key in overrides:
        return _coerce_bool(overrides[key])
    return None


def _apply_checkpoint_fields(config: TrainingConfig, body: dict) -> None:
    """Fold the training form's checkpoint / eval / early-stopping fields into ``config``.

    Absent keys keep the config's value (API callers and presets); the form sends
    a hidden ``"0"`` for unticked boxes so "off" is explicit. Bad numbers raise
    ``ValueError`` for the caller's 400.
    """
    for key in ("save_checkpoints", "early_stopping"):
        flag = _optional_body_bool(body, key)
        if flag is not None:
            setattr(config, key, flag)
    if body.get("save_limit") not in (None, ""):
        config.save_total_limit = int(body["save_limit"])
    if body.get("eval_steps") not in (None, ""):
        config.eval_steps = max(0, int(body["eval_steps"]))


_HF_REPO_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


def _resolve_model_path(model_path: str, allow_download: bool) -> tuple[str, str | None]:
    """Resolve a trainable base to a local dir; block silent multi-GB hub pulls.

    A bare ``org/repo`` model_path makes ``from_pretrained`` stream gigabytes
    mid-run (this stalled the first c327fa36 attempt). Local paths pass
    through; repo ids already in the HF hub cache pass through (the load hits
    disk); ids present in the app's ``~/.finetune-studio/hf_models`` download
    dir are rewritten to that path; anything else needs an explicit
    ``allow_download=true``. Returns ``(model_path, error_or_None)``.
    """
    p = Path(model_path).expanduser()
    if p.exists():
        return str(p), None
    repo = model_path.split(":", 1)[0]
    if model_path.startswith(("/", "./", "../", "~")) or re.match(r"^[A-Za-z]:[\\/]", model_path):
        return model_path, f"model path does not exist: {model_path}"
    if not _HF_REPO_ID_RE.match(repo):
        return model_path, None  # not a repo id — the engine surfaces its own error
    org, name = repo.split("/", 1)
    from finetune_studio.hf_env import hf_hub_cache

    hub_snap = hf_hub_cache() / f"models--{org}--{name}" / "snapshots"
    if hub_snap.is_dir() and any(hub_snap.iterdir()):
        return model_path, None
    app_dir = Path.home() / ".finetune-studio" / "hf_models" / f"{org}__{name}"
    if app_dir.is_dir():
        return str(app_dir), None
    if allow_download:
        return model_path, None
    return model_path, (
        f"model_path {model_path!r} is not local — starting would download it "
        "from the Hub mid-run (multi-GB; this stalled the first c327fa36 "
        "attempt). Pull it via Models → Explore and pick the local copy, or "
        "send allow_download=true to proceed anyway."
    )


# ── Training Presets ──────────────────────────────────────────
# Real configurations for different model sizes and use cases.
# These are NOT toy configs — they're based on LoRA best practices
# from the Unsloth/PEFT documentation and fine-tuning guides.
TRAINING_PRESETS: dict[str, dict] = {
    # ── Small models (0.5B–1.5B) ──────────────────────────────
    "nano-fast": {
        "name": "Nano — Fast iteration",
        "description": "Quick runs to check your data on a small (≤1B) model: minutes.",
        "target_models": ["0.6B", "1.5B"],
        "min_vram_gb": 6,
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 1024,
        "warmup_steps": 10,
        "weight_decay": 0.01,
        "save_steps": 50,
        "logging_steps": 5,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": False,
    },
    "nano-quality": {
        "name": "Nano — Quality",
        "description": "More epochs and a longer context for small (≤2B) models.",
        "target_models": ["0.6B", "1.5B"],
        "min_vram_gb": 8,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_steps": 20,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    # ── Medium models (3B–8B) ─────────────────────────────────
    "standard": {
        "name": "Standard — Balanced",
        "description": "Good default for 3B–9B models. Measured: a 9B, 2,500 pairs, 3 epochs ≈ 30 min on an RTX 3090.",
        "target_models": ["3B", "7B", "8B"],
        "min_vram_gb": 16,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_steps": 30,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    "standard-long": {
        "name": "Standard — Long training",
        "description": "Twice the epochs of Balanced, so about twice the time; watch for over-fitting.",
        "target_models": ["3B", "7B", "8B"],
        "min_vram_gb": 16,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-4,
        "num_epochs": 6,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    "high-rank": {
        "name": "High Rank — Maximum capacity",
        "description": "Rank 128 and batch 1: a bigger adapter for complex formats, slower than Balanced.",
        "target_models": ["3B", "7B", "8B"],
        "min_vram_gb": 20,
        "lora_rank": 128,
        "lora_alpha": 256,
        "learning_rate": 1e-4,
        "num_epochs": 4,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "warmup_steps": 40,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    # ── Large models (14B–27B) ────────────────────────────────
    "large": {
        "name": "Large — 14B/27B",
        "description": "For 14B–27B bases; needs QLoRA on a 24 GB card and takes several times longer than a 9B.",
        "target_models": ["14B", "27B"],
        "min_vram_gb": 24,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
    # ── QLoRA (4-bit base) ────────────────────────────────────
    "qlora": {
        "name": "QLoRA — Memory efficient",
        "description": "4-bit base + LoRA: the least VRAM (a 9B trains on a 24 GB card).",
        "target_models": ["3B", "7B", "8B"],
        "min_vram_gb": 6,
        "lora_rank": 64,
        "lora_alpha": 128,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_steps": 30,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "bf16": True,
        "unsloth": True,
        "merge_on_save": True,
    },
}


def _get_presets() -> list[dict]:
    """Return all presets with their IDs."""
    return [{"id": k, **v} for k, v in TRAINING_PRESETS.items()]


_PRESET_TUNING_FIELDS: dict[str, type] = {
    "lora_rank": int, "lora_alpha": int, "learning_rate": float, "num_epochs": int,
    "batch_size": int, "gradient_accumulation_steps": int, "max_seq_length": int,
    "warmup_steps": int,
}


PREFERENCE_MODES = ("dpo", "kto")
# LoRA preference defaults, measured with scripts/pref_quality_eval.py (Qwen3-0.6B, SFT-merged
# start, ~80 pairs = 54 optimizer steps; HANDOFF "Preference quality"). TRL's 1e-6 is a
# full-fine-tune value: with a LoRA adapter the policy does not move from it, and the
# reference is the adapter-disabled start, so zero movement = zero reward margin. Plain DPO at
# >= 3e-5 made the model forget facts (59/60 -> 31-48/60); adding the keep-chosen NLL term
# (RPO) at 1e-4 kept them (50-54/60) while moving behaviour. KTO lost facts at 5e-5 (51/60)
# and 1e-5 (54/60) without gaining anything on this task; 1e-5 is its least-damaging tested value.
# Long runs need far less: Zephyr's DPO-QLoRA recipe is 5e-6 over ~15k steps
# (huggingface/alignment-handbook recipes/zephyr-7b-beta/dpo/config_qlora.yaml).
PREFERENCE_DEFAULTS: dict[str, dict[str, float | int]] = {
    "dpo": {"learning_rate": 1e-4, "num_epochs": 3, "warmup_steps": 0},
    "kto": {"learning_rate": 1e-5, "num_epochs": 3, "warmup_steps": 0},
}
DEFAULT_KEEP_CHOSEN_WEIGHT = 1.0
LONG_RUN_STEPS = 300  # above this the short-run learning rate above is untested


def _apply_preset(preset_id: str, overrides: dict | None = None) -> TrainingConfig:
    """Build a TrainingConfig from a preset, with optional field overrides."""
    p = TRAINING_PRESETS.get(preset_id)
    if p is None:
        raise ValueError(f"Unknown preset: {preset_id}")
    kwargs = {k: v for k, v in p.items() if k not in ("name", "description", "target_models", "min_vram_gb")}
    if overrides:
        kwargs.update(overrides)
    return TrainingConfig(**kwargs)


@router.get("/presets")
async def list_presets():
    """Return all training presets."""
    return _get_presets()


@router.get("/recommend")
async def recommend_config(
    tier: str = "balanced",
    base_model: str = "",
    dataset: str = "",
    pairs: int | None = None,
    batch_size: int = 2,
    grad_accum: int = 4,
):
    """Propose training settings for a base model + dataset at a quality tier.

    Returns the computed config plus the arithmetic (optimizer-step math,
    dataset-size scaling, LR-by-size rules) so the user can judge it, not
    just obey it.
    """
    from finetune_studio.training.preset_advisor import propose
    try:
        adv = propose(tier=tier, base_model_ref=base_model,
                      dataset_path=dataset or None, pair_count_hint=pairs,
                      batch_size=batch_size,
                      gradient_accumulation_steps=grad_accum)
        return adv.to_dict()
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e), "code": "bad_request"})

@router.get("/status")
async def status(project_id: str | None = None):
    """Same snapshot the live SSE stream sends (one builder: ``monitor.training_snapshot``).

    ``?project_id=`` scopes it: another project's run (live or its final state) is ``idle`` for this project."""
    from finetune_studio.training.monitor import training_snapshot
    return training_snapshot(training_engine, project_id or None)

@router.get("/failure")
async def latest_failure(project_id: str):
    """The project's newest run when it failed, with a plain-language hint (OOM) from its own settings.

    ``{"failure": null}`` when the newest run is not a failure, so the Live status
    panel never shows a stale error after a later run started or succeeded.
    """
    from finetune_studio import db
    from finetune_studio.training.failure_hint import diagnose_failure
    if not db.get_project(project_id):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    runs = db.list_runs(project_id)
    run = runs[0] if runs else None
    if not run or run.get("status") not in ("failed", "error"):
        return {"failure": None}
    error = run.get("error") or ""
    hint = diagnose_failure(error, run.get("settings") or {})
    return {"failure": {
        "run_id": run["id"], "name": run.get("name") or run["id"], "error": error,
        "finished_at": run.get("finished_at"),
        "hint": hint.to_dict() if hint else None,
    }}


@router.get("/status-text")
async def status_text():
    """The dashboard card's value: what the GPU is doing NOW. A finished run is history (the runs list, the
    activity feed), not a status: the card says Idle once the run ended."""
    s = training_engine.state
    if s.status == "training":
        text = f"Training ({s.current_step}/{s.total_steps})"
    elif s.status == "loading":
        text = "Loading model…"
    elif s.status == "saving":
        text = "Saving…"
    else:
        text = "Idle"
    return PlainTextResponse(text)

@router.get("/progress")
async def progress(project_id: str | None = None):
    """SSE live training status (preferred over polling ``/status``); ``?project_id=`` scopes it like ``/status``."""
    return sse_response(training_events(training_engine, project_id or None))


@router.get("/progress-text")
async def progress_text():
    """Plain-text one-shot progress string for dashboard polling.

    Returns something short the dashboard can render without streaming:
      "Idle"            — no run
      "step 124/1000"   — running
      "step 1000/1000 · loss 1.42" — running w/ loss
      "Saving…"         — saving
      "last run: done · 54 steps · <project>" — nothing running; the last run's outcome, named
      "last run: error · <project>"            — nothing running; the last run failed
    """
    s = training_engine.state
    if s.status == "training":
        loss = f" · loss {s.loss:.2f}" if s.loss else ""
        return PlainTextResponse(f"step {s.current_step}/{s.total_steps}{loss}")
    if s.status == "loading":
        return PlainTextResponse("Loading model…")
    if s.status == "saving":
        return PlainTextResponse("Saving…")
    if s.status in ("done", "error", "stopped"):
        from finetune_studio import db
        pid = getattr(training_engine, "current_project_id", None)
        proj = db.get_project(pid) if pid else None
        who = f" · {proj['name']}" if proj else ""
        steps = f" · {s.total_steps} steps" if s.status == "done" and s.total_steps else ""
        return PlainTextResponse(f"last run: {s.status}{steps}{who}")
    return PlainTextResponse("Idle")

@router.get("/runs")
async def list_training_runs():
    """List ALL training runs (across all projects)."""
    from finetune_studio.db.runs import list_runs
    return list_runs()


@router.get("/runs/{pid}")
async def list_training_runs_for_project(pid: str):
    """List training runs for a specific project.

    The path param is named ``pid`` but it is a *project* id, not a run id.
    """
    from finetune_studio import db
    from finetune_studio.db.runs import list_runs
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    return list_runs(pid)


@router.post("/start")
async def start_training(request: Request):
    from finetune_studio import db
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"detail": "request body must be valid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"detail": "JSON object body required"}, status_code=400)

    # Extract data_path and project_id FIRST
    data_path = body.get("data_path", "")
    dataset_id = body.get("dataset_id", "")
    project_id = body.get("project_id", "")
    if dataset_id and not data_path:
        ds = db.get_dataset(dataset_id)
        if not ds or ds.get("project_id") != project_id:
            return JSONResponse({"detail": f"dataset {dataset_id!r} not found in this project"}, status_code=404)
        data_path = ds["data_path"]
    if not data_path:
        return JSONResponse({"detail": "No data_path / dataset_id provided"}, status_code=400)

    preset_id = body.get("preset_id")
    overrides = body.get("overrides") or {}
    # Project training form omits unchecked boxes; default false (not true).
    merge_flag = _optional_body_bool(body, "merge_on_save", overrides)
    unsloth_flag = _optional_body_bool(body, "unsloth", overrides)
    try:
        if preset_id:
            try:
                # Top-level tuning fields (what the project form sends) win over
                # the preset, same as ``overrides``; they were silently ignored.
                top = {k: cast(body[k]) for k, cast in _PRESET_TUNING_FIELDS.items()
                       if body.get(k) not in (None, "")}
                config = _apply_preset(preset_id, {**top, **overrides})
            except (ValueError, TypeError) as e:
                return JSONResponse({"detail": str(e)}, status_code=400)
            if not config.model_path and body.get("model_path"):
                config.model_path = body["model_path"]
            if merge_flag is not None:
                config.merge_on_save = merge_flag
            if unsloth_flag is not None:
                config.unsloth = unsloth_flag
        else:
            config = TrainingConfig(
                model_path=body.get("model_path", ""),
                output_dir=body.get("output_dir", "output"),
                lora_rank=int(body.get("lora_rank", 64)),
                lora_alpha=int(body.get("lora_alpha", 128)),
                learning_rate=float(body.get("learning_rate", 2e-4)),
                num_epochs=int(body.get("num_epochs", 3)),
                batch_size=int(body.get("batch_size", 2)),
                gradient_accumulation_steps=int(body.get("gradient_accumulation_steps", 4)),
                max_seq_length=int(body.get("max_seq_length", 2048)),
                warmup_steps=int(body.get("warmup_steps", 30)),
                weight_decay=float(body.get("weight_decay", 0.01)),
                save_steps=int(body.get("save_steps", 100)),
                logging_steps=int(body.get("logging_steps", 10)),
                bf16=_coerce_bool(body.get("bf16", True)),
                # Omitted → standard TRL (no Unsloth status wording) for stock Qwen3-4B flow.
                unsloth=False if unsloth_flag is None else unsloth_flag,
                merge_on_save=False if merge_flag is None else merge_flag,
                export_gguf=_coerce_bool(body.get("export_gguf", False)),
                gguf_quants=body.get("gguf_quants", ["f16", "q8_0", "q4_k_m", "q5_k_m"]),
                data_path=data_path,
                project_id=project_id,
                abliterate=_coerce_bool(body.get("abliterate", False)),
                abliteration_strength=float(body.get("abliteration_strength", 1.0)),
            )
        _apply_checkpoint_fields(config, body)
    except (ValueError, TypeError, OverflowError) as e:
        return JSONResponse({"detail": f"invalid training parameter: {e}"}, status_code=400)
    if config.save_total_limit < 1:
        return JSONResponse({"detail": "save_limit must be >= 1"}, status_code=400)
    if config.early_stopping and config.eval_steps <= 0:
        return JSONResponse({"detail": "Early stopping needs evaluation: set "
                             "'Eval every N steps' above 0."}, status_code=400)
    for _name in ("num_epochs", "batch_size", "gradient_accumulation_steps", "max_seq_length", "lora_rank"):
        if getattr(config, _name, 1) < 1:
            return JSONResponse({"detail": f"{_name} must be >= 1"}, status_code=400)
    merge_on_save = config.merge_on_save

    start_run_id = str(body.get("start_run_id") or "").strip()
    if start_run_id:
        from finetune_studio.training.start_points import (
            StartPointError,
            resolve_start_point,
        )
        try:
            config.model_path = resolve_start_point(project_id, start_run_id)
        except StartPointError as e:
            return JSONResponse({"detail": str(e)}, status_code=400)
    if not config.model_path:
        return JSONResponse({"detail": "No model_path provided"}, status_code=400)
    config.model_path, dl_err = _resolve_model_path(
        config.model_path, _optional_body_bool(body, "allow_download") or False
    )
    if dl_err:
        return JSONResponse({"detail": dl_err}, status_code=400)
    training_data = load_jsonl(data_path)
    if not training_data:
        return JSONResponse({"detail": "Training dataset is empty"}, status_code=400)
    training_mode = str(body.get("training_mode") or "sft").strip().lower()
    supported_training_modes = {
        "sft", "dpo", "kto", "tool_sft", "continued_pretraining", "reasoning_distillation",
    }
    if training_mode not in supported_training_modes:
        return JSONResponse({
            "detail": "training_mode must be one of: " + ", ".join(sorted(supported_training_modes)),
        }, status_code=400)
    raw_replay = body.get("replay_percent")
    if raw_replay not in (None, ""):
        try:
            replay_percent = float(raw_replay)
        except (TypeError, ValueError):
            return JSONResponse({"detail": "replay_percent must be a number"}, status_code=400)
        if not 0 <= replay_percent <= 50:
            return JSONResponse({"detail": "replay_percent must be between 0 and 50"}, status_code=400)
        if replay_percent and training_mode not in ("sft", "tool_sft"):
            return JSONResponse({"detail": "replay mixes in chat answers, so it is for SFT runs only"},
                                status_code=400)
        config.replay_fraction = replay_percent / 100
    if training_mode in PREFERENCE_MODES:
        # LoRA preference tuning needs its own optimizer scale (see PREFERENCE_DEFAULTS);
        # the SFT preset/form defaults would wreck the base model. Explicit values and
        # selected presets stay user-controlled.
        if not preset_id:
            for field, value in PREFERENCE_DEFAULTS[training_mode].items():
                if body.get(field) in (None, ""):
                    setattr(config, field, value)
        config.unsloth = False
        try:
            beta = float(body.get("preference_beta") or config.preference_beta)
        except (TypeError, ValueError):
            return JSONResponse({"detail": "preference_beta must be a number"}, status_code=400)
        if not 0 < beta <= 1:
            return JSONResponse({"detail": "preference_beta must be in (0, 1]"}, status_code=400)
        config.preference_beta = beta
        if training_mode == "dpo":
            raw_weight = body.get("preference_sft_weight")
            try:
                sft_weight = DEFAULT_KEEP_CHOSEN_WEIGHT if raw_weight in (None, "") else float(raw_weight)
            except (TypeError, ValueError):
                return JSONResponse({"detail": "preference_sft_weight must be a number"}, status_code=400)
            if not 0 <= sft_weight <= 10:
                return JSONResponse({"detail": "preference_sft_weight must be between 0 and 10"}, status_code=400)
            config.preference_sft_weight = sft_weight
        if training_mode == "dpo":
            try:
                format_for_preference(training_data)
            except (TypeError, ValueError) as e:
                return JSONResponse({
                    "detail": f"Invalid preference dataset: {e}. Each row needs prompt, chosen, and rejected.",
                }, status_code=400)
            if len(training_data) < 2:
                return JSONResponse({
                    "detail": "DPO needs at least 2 preference pairs so one can be held out for validation.",
                }, status_code=400)
        else:
            try:
                unpaired = format_for_unpaired(training_data)
            except (TypeError, ValueError) as e:
                return JSONResponse({
                    "detail": f"Invalid KTO dataset: {e}. Each row needs prompt, completion and a true/false "
                             "label (preference rows prompt/chosen/rejected are converted automatically).",
                }, status_code=400)
            if len(unpaired) < 4:
                return JSONResponse({
                    "detail": "KTO needs at least 4 examples so a validation holdout still has both kinds.",
                }, status_code=400)
            if config.batch_size < 2:
                return JSONResponse({
                    "detail": "KTO needs batch_size >= 2: its KL term is estimated from other prompts in the "
                             "same batch (TRL: 'Actual (not effective) batch size must be > 1').",
                }, status_code=400)
    elif training_mode == "continued_pretraining":
        from finetune_studio.training.data import format_for_continued_pretraining
        try:
            format_for_continued_pretraining(training_data)
        except (TypeError, ValueError) as e:
            return JSONResponse({"detail": f"Invalid continued-pretraining dataset: {e}"}, status_code=400)
    elif training_mode in {"tool_sft", "reasoning_distillation"}:
        from finetune_studio.training.data import format_for_sft
        if training_mode == "tool_sft" and any(
            not isinstance(row, dict) or not isinstance(row.get("messages"), list)
            or not isinstance(row.get("tools"), list) or not row["tools"]
            or any(not isinstance(tool, dict) or not isinstance(tool.get("function"), dict)
                   for tool in row["tools"])
            for row in training_data
        ):
            return JSONResponse({
                "detail": "Every tool-calling SFT row needs messages and its tools function JSON-schema list.",
            }, status_code=400)
        if training_mode == "tool_sft" and not any(
            isinstance(message, dict) and isinstance(message.get("tool_calls"), list) and message["tool_calls"]
            for row in training_data for message in row["messages"]
        ):
            return JSONResponse({
                "detail": "Tool-calling SFT needs at least one assistant tool_calls example; plain conversations belong in SFT.",
            }, status_code=400)
        if not format_for_sft(training_data):
            return JSONResponse({"detail": f"Invalid {training_mode} dataset: no supported conversation rows"}, status_code=400)
    system_prompt = body.get("system_prompt", "")
    system_prompt_mode = body.get("system_prompt_mode", "bake")
    # The project training form sends only the mode; bake/runtime without a
    # prompt used to silently train with none (E2E-26). Use the project's.
    if not system_prompt and system_prompt_mode != "none" and project_id:
        system_prompt = (db.get_project(project_id) or {}).get("system_prompt", "") or ""

    if training_engine.state.status in ("training", "loading", "saving"):
        return JSONResponse({"detail": "another training run is in progress"}, status_code=409)
    # Test runs use the inference worker's model without a GPU lease: training would unload it mid-run.
    from finetune_studio.webui.testing_jobs import active_jobs as testing_active
    on_card = [j for j in testing_active() if j["gpu"]]
    if on_card:
        return JSONResponse({"detail": f"a test {on_card[0]['kind']} is using the GPU (benchmark "
                                       f"{on_card[0]['benchmark_id']}); wait for it or cancel it on the Testing page",
                             "code": "gpu_busy"}, status_code=409)

    # Create a run record
    run = db.create_run(
        project_id=project_id,
        name=f"Run · {Path(data_path).name}",
        base_model=config.model_path,
        data_path=data_path,
        settings_obj={
            "training_mode": training_mode,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "max_seq_length": config.max_seq_length,
            "merge_on_save": merge_on_save,
            "unsloth": config.unsloth,
            "system_prompt": system_prompt,
            "system_prompt_mode": system_prompt_mode,
            "dataset_rows": len(training_data),
            "validation_split": 0.1,
            "preference_beta": config.preference_beta if training_mode in PREFERENCE_MODES else None,
            "preference_sft_weight": config.preference_sft_weight if training_mode == "dpo" else None,
            "replay_percent": round(config.replay_fraction * 100, 1) or None,
            "start_run_id": start_run_id or None,
        },
        system_prompt=system_prompt,
        system_prompt_mode=system_prompt_mode,
    )
    run_id = run["id"]
    # A bare "output" default is shared by every project and run, so each new
    # run silently overwrote the previous run's adapter/merged model (E2E-25).
    if (config.output_dir or "output").rstrip("/") == "output":
        config.output_dir = (
            f"output/projects/{project_id}/runs/{run_id}" if project_id else f"output/runs/{run_id}"
        )
    # Persist immediately so Export / Training UI can locate the run even when
    # later status callbacks omit output_path or use a composite engine run id.
    db.update_run(
        run_id,
        name=f"Run {run_id} · {Path(data_path).stem}",   # no ".jsonl" in pickers ("…approved_core.jsonl/merged")
        output_path=config.output_dir,
    )
    # Training and any resident helper/test model share the same GPU. Unload
    # BOTH engines before the worker imports/loads the trainable base —
    # unloading only the global InferenceEngine used to leave a
    # ModelManager-loaded helper (e.g. data-prep's) resident through the
    # whole run, competing for VRAM and making a valid 4B run fail at step
    # zero for no reason visible from the training config itself.
    try:
        from finetune_studio.models.llama_loader import unload_all_models
        # Freeing CUDA memory takes seconds: not on the event loop.
        await asyncio.to_thread(unload_all_models)
    except Exception:
        log.exception("Failed to unload resident models before training")
    try:
        config.training_mode = training_mode
        if training_mode != "sft":
            # The Unsloth worker currently formats chat-only SFT; the explicit
            # routes use the standard TRL lifecycle so their data semantics are
            # not silently lost.
            config.unsloth = False
        # The run is its own job process (it persists this run's row itself and outlives a web restart).
        training_engine.start(config, training_data, system_prompt, run_id=run_id,
                              output_dir=config.output_dir, project_id=project_id)
    except (TrainingBusy, JobRejected) as e:
        db.update_run(run_id, status="error")
        code = getattr(e, "code", "") or "training_busy"
        return JSONResponse({"detail": f"start refused: {e}", "code": code},
                            status_code=getattr(e, "status", 409))
    except Exception as e:
        log.exception("training start failed")
        db.update_run(run_id, status="error")
        return JSONResponse({"detail": f"start failed: {e}"}, status_code=500)
    warnings: list[str] = []
    if training_mode in PREFERENCE_MODES:
        steps = math.ceil(len(training_data) * 0.9 / max(1, config.batch_size * config.gradient_accumulation_steps))
        steps *= config.num_epochs
        if steps > LONG_RUN_STEPS and config.learning_rate >= 5e-5:
            warnings.append(
                f"About {steps} optimizer steps at learning rate {config.learning_rate:g}: that rate was only "
                f"measured on ~50-step runs. Long preference runs usually need 5e-6 to 2e-5; watch the "
                "held-out accuracy and the quiz for forgotten facts."
            )
    return {"status": "started", "steps": training_engine.state.total_steps, "run_id": run_id,
            **({"warnings": warnings} if warnings else {})}

@router.get("/start-points/{pid}")
async def training_start_points(pid: str):
    """Finished runs of a project with a merged model: valid starting points for DPO/KTO."""
    from finetune_studio import db
    from finetune_studio.training.start_points import list_start_points
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    return {"start_points": list_start_points(pid)}


@router.post("/stop")
async def stop_training():
    training_engine.stop()
    return {"status": "stopping"}


@router.get("/runs/{run_id}/auto-suites")
async def list_auto_suites(run_id: str):
    """List all auto-generated suites for a training run."""
    from finetune_studio.db.connection import cursor
    with cursor() as c:
        rows = c.execute("SELECT id, suite_name, suite_path, case_count, categories_json, created_at FROM auto_suites WHERE run_id = ? ORDER BY created_at DESC", (run_id,)).fetchall()
    return [{"id": r[0], "suite_name": r[1], "suite_path": r[2], "case_count": r[3], "categories": json.loads(r[4]) if r[4] else {}, "created_at": r[5]} for r in rows]


@router.post("/runs/{run_id}/auto-suites/generate")
async def trigger_auto_suite(run_id: str, request: Request):
    """Trigger auto-generation of a benchmark suite from training data.

    Coverage rule: the suite tests **every** dataset row by default (N rows
    → N cases, no cap). ``{"sample_size": K}`` in the body is the only way
    to shrink it — a deterministic uniform sample, labeled ``-sampledKofN``
    in the suite name and ``coverage="sampled"`` in the response, so a
    sampled verdict can never be misread as full coverage.

    JSONL→suite conversion is deterministic (no LLM). The response still
    names the configured helper so UI / future LLM-assisted generation
    paths share one explicit default with no silent model fallback.
    """
    from finetune_studio import db
    from finetune_studio.models.helper import (
        DEFAULT_HELPER_LABEL,
        get_configured_helper_provider,
        get_helper_provider_id,
    )
    sample_size: int | None = None
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — empty/absent body = full coverage
        body = {}
    if isinstance(body, dict) and body.get("sample_size") not in (None, ""):
        try:
            sample_size = int(body["sample_size"])
        except (TypeError, ValueError):
            return JSONResponse({"detail": "sample_size must be an integer"}, status_code=400)
        if sample_size <= 0:
            return JSONResponse({"detail": "sample_size must be > 0"}, status_code=400)
    run = db.get_run(run_id)
    if not run:
        return JSONResponse({"detail": "run not found"}, status_code=404)
    data_path = (run.get("data_path") or "").strip()
    if not data_path:
        return JSONResponse({"detail": "run has no data_path"}, status_code=409)
    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return JSONResponse({"detail": "run has no output_path"}, status_code=409)
    from finetune_studio.testing.generate_suite import generate_suite_from_training_data
    result = generate_suite_from_training_data(data_path, output_path, max_cases=sample_size)
    if result.get("error"):
        body = {k: v for k, v in result.items() if k != "error"}
        return JSONResponse({"detail": result["error"], "code": "no_cases", **body}, status_code=422)
    # Record in DB
    from time import time as _time

    from finetune_studio.db.connection import cursor, new_id
    suite_id = new_id()
    with cursor() as c:
        c.execute(
            "INSERT INTO auto_suites (id, run_id, project_id, suite_name, suite_path, case_count, categories_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (suite_id, run_id, run.get("project_id", ""),
             result.get("suite_name", "auto"), result.get("suite_path", ""),
             result.get("case_count", 0),
             json.dumps(result.get("categories", {})), _time()),
        )
    helper = get_configured_helper_provider()
    return {
        "ok": True,
        "suite_id": suite_id,
        "generation_mode": "deterministic",
        "helper_provider_id": get_helper_provider_id(),
        "helper_label": (helper or {}).get("label") or DEFAULT_HELPER_LABEL,
        **result,
    }
