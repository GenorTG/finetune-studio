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

def scan_models(directories: list) -> list:
    models = []
    for d in directories:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if not x.startswith(".") and x != "__pycache__"]
            for f in files:
                if f.endswith(".gguf"):
                    fp = os.path.join(root, f)
                    size = os.path.getsize(fp) / (1024**3)
                    models.append(ModelInfo(
                        name=f, path=fp, format="gguf", size_gb=round(size, 2),
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
                total = sum(
                    os.path.getsize(os.path.join(root, f))
                    for f in files if f.endswith((".safetensors", ".bin", ".pt"))
                ) / (1024**3)
                name = root.split("/")[-1]
                models.append(ModelInfo(
                    name=name, path=root, format="safetensors",
                    size_gb=round(total, 2), architecture=arch, parameters=params,
                ))
                dirs.clear()
    return models
