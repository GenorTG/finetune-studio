"""Track available models and their paths.

WHAT THIS FILE DOES
==================
A simple registry that remembers what models are available:
  - On disk: scans model directories for .gguf files
  - In HuggingFace cache: checks ~/.cache/huggingface/
  - Remote: lists models from HuggingFace Hub (if HF_TOKEN is set)

KEY CONCEPTS
============
- Registry pattern: a single source of truth for "what models exist".
- Auto-discovery: scan directories instead of hardcoding paths.
- Caching: don't re-scan every time; cache the results.
"""

import json
import os
from dataclasses import dataclass


@dataclass
class ModelInfo:
    name: str
    path: str
    format: str
    size_gb: float
    architecture: str = ""
    parameters: str = ""
    modified: str = ""
    vision: bool = False
    category: str = "discovered"   # discovered | base_model | trained_export | local_helper | downloaded
    project_id: str = ""           # which project produced this (for trained_export)
    run_id: str = ""               # which training run produced this

# Non-chat model architectures to skip
_SKIP_ARCHES = {"BertModel", "BertForMaskedLM", "ClipVisionModel", "CLIPVisionModel",
                "SiglipVisionModel", "MultiModalProjector"}
# File patterns that indicate non-model files
_SKIP_GGUF_PATTERNS = ("mmproj", "projector", "vision")


def _safe_model_name(root: str, cfg: dict, project_name: str = "") -> str:
    """Extract a human-readable model name from config or directory path.

    Returns the most descriptive name available. For trained exports in generic
    dirs ("merged", "abliterated"), uses the project name for context.
    """
    # Try config.json fields
    for key in ("name", "model_name"):
        if cfg.get(key):
            return cfg[key]
    # Use model_type + architecture for a readable name
    model_type = cfg.get("model_type", "")
    arch = (cfg.get("architectures") or [""])[0]
    # Strip common suffixes from arch
    for suffix in ("ForCausalLM", "ForConditionalGeneration", "ForSequenceClassification"):
        arch = arch.replace(suffix, "")
    # Directory name — prefer parent if current is a hash
    dirname = os.path.basename(root)
    if len(dirname) == 40 and all(c in "0123456789abcdef" for c in dirname):
        # HuggingFace cache hash — use parent dir name
        dirname = os.path.basename(os.path.dirname(root))
        # e.g. "models--unsloth--gemma-4-E4B-it-unsloth-bnb-4bit" → "gemma-4-E4B-it-unsloth-bnb-4bit"
        if dirname.startswith("models--"):
            dirname = dirname.split("--", 2)[-1] if "--" in dirname else dirname
    # If dirname is a generic export dir, use project name + dirname for context
    generic_dirs = {"merged", "abliterated", "gguf", "gptq", "awq", "adapter", "checkpoint-0", "checkpoint-1"}
    if dirname.lower() in generic_dirs:
        if project_name:
            return f"{project_name} ({dirname})"
        parent = os.path.basename(os.path.dirname(root))
        if parent and parent not in ("output", "models"):
            return f"{parent}/{dirname}"
        return dirname
    # Use the directory name (which is usually descriptive) as the primary name
    if dirname and dirname not in ("export", "model", "snapshots"):
        return dirname
    if arch and model_type:
        return f"{arch} ({model_type})"
    if arch:
        return arch
    if model_type:
        return model_type
    return dirname


def _lookup_project_name(output_path: str) -> tuple[str, str]:
    """Look up (project_id, project_name) from training_runs DB by output_path."""
    try:
        import sqlite3 as _sql
        db_path = os.path.join(os.path.expanduser("~"), ".finetune-studio", "finetune_studio.db")
        if not os.path.exists(db_path):
            db_path = os.path.join(os.path.expanduser("~"), ".finetune-studio", "fts.db")
        if os.path.exists(db_path):
            with _sql.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT tr.project_id, p.name FROM training_runs tr "
                    "LEFT JOIN projects p ON p.id = tr.project_id "
                    "WHERE tr.output_path LIKE ? LIMIT 1",
                    (f"{output_path}%",),
                ).fetchone()
                if row:
                    return (row[0] or "", row[1] or "")
    except Exception:
        pass
    return ("", "")


def scan_models(directories: list) -> list:
    models = []
    for d in directories:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if not x.startswith(".") and x != "__pycache__"]
            # Detect mmproj files in this directory
            has_mmproj = any("mmproj" in f.lower() for f in files if f.endswith(".gguf"))
            for f in files:
                if f.endswith(".gguf"):
                    # Skip multimodal projectors and vision encoders
                    fl = f.lower()
                    if any(p in fl for p in _SKIP_GGUF_PATTERNS):
                        continue
                    fp = os.path.join(root, f)
                    size = os.path.getsize(fp) / (1024**3)
                    # Determine category from path
                    cat = "discovered"
                    proj_id = ""
                    run_id = ""
                    p = root.lower()
                    if "/output" in p or "output_" in p:
                        cat = "trained_export"
                    elif "shared_models" in p:
                        cat = "local_helper"
                    elif "hf_models" in p:
                        cat = "downloaded"
                    elif "huggingface/hub" in p:
                        cat = "base_model"
                    models.append(ModelInfo(
                        name=f, path=fp, format="gguf", size_gb=round(size, 2),
                        vision=has_mmproj, category=cat, project_id=proj_id, run_id=run_id,
                    ))
            has_st = any(f.endswith(".safetensors") for f in files)
            has_cfg = "config.json" in files
            if has_st and has_cfg:
                cfg_path = os.path.join(root, "config.json")
                try:
                    with open(cfg_path) as cf:
                        cfg = json.load(cf)
                    arch = cfg.get("architectures", [""])[0] if cfg.get("architectures") else ""
                    params = cfg.get("model_type", "")
                except Exception:  # noqa: BLE001
                    arch, params = "", ""
                # Skip non-chat models (embeddings, vision encoders, projectors)
                if arch in _SKIP_ARCHES:
                    dirs.clear()
                    continue
                # Skip tiny models (< 0.5GB) — likely projectors or adapters
                total = sum(
                    os.path.getsize(os.path.join(root, f))
                    for f in files if f.endswith((".safetensors", ".bin", ".pt"))
                ) / (1024**3)
                if total < 0.5:
                    dirs.clear()
                    continue
                name = _safe_model_name(root, cfg)
                # Determine category from path
                cat = "discovered"
                proj_id = ""
                run_id = ""
                p = root.lower()
                if "/output" in p or "output_" in p:
                    cat = "trained_export"
                    # Try to find project_id from training_runs DB
                    try:
                        import sqlite3 as _sql
                        db_path = os.path.join(os.path.expanduser("~"), ".finetune-studio", "finetune_studio.db")
                        if not os.path.exists(db_path):
                            db_path = os.path.join(os.path.expanduser("~"), ".finetune-studio", "fts.db")
                        if os.path.exists(db_path):
                            with _sql.connect(db_path) as conn:
                                row = conn.execute(
                                    "SELECT project_id FROM training_runs WHERE output_path = ? LIMIT 1", (root,)
                                ).fetchone()
                                if row:
                                    proj_id = row[0]
                    except Exception:
                        pass
                elif "shared_models" in p:
                    cat = "local_helper"
                elif "hf_models" in p:
                    cat = "downloaded"
                elif "huggingface/hub" in p:
                    cat = "base_model"
                models.append(ModelInfo(
                    name=name, path=root, format="safetensors",
                    size_gb=round(total, 2), architecture=arch, parameters=params,
                    category=cat, project_id=proj_id, run_id=run_id,
                ))
                dirs.clear()
    return models
