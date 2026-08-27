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

# Non-chat model architectures to skip
_SKIP_ARCHES = {"BertModel", "BertForMaskedLM", "ClipVisionModel", "CLIPVisionModel",
                "SiglipVisionModel", "MultiModalProjector"}
# File patterns that indicate non-model files
_SKIP_GGUF_PATTERNS = ("mmproj", "projector", "vision")


def _safe_model_name(root: str, cfg: dict) -> str:
    """Extract a human-readable model name from config or directory path."""
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
    # Use the directory name (which is usually descriptive) as the primary name
    # Only fall back to arch+model_type if dirname is generic or missing
    if dirname and dirname not in ("export", "model", "snapshots"):
        return dirname
    if arch and model_type:
        return f"{arch} ({model_type})"
    if arch:
        return arch
    if model_type:
        return model_type
    return dirname


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
                    models.append(ModelInfo(
                        name=f, path=fp, format="gguf", size_gb=round(size, 2),
                        vision=has_mmproj,
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
                models.append(ModelInfo(
                    name=name, path=root, format="safetensors",
                    size_gb=round(total, 2), architecture=arch, parameters=params,
                ))
                dirs.clear()
    return models
