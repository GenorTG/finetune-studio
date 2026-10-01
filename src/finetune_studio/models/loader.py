"""Inspect a model path on disk and return display metadata.

Despite the broader docstring this module used to carry, it does not load
or wrap any inference engine — ``load_model_info`` only reads a GGUF header
size or a safetensors ``config.json`` to describe a model for the UI. The
real GGUF loader lives in ``finetune_studio.models.llama_loader``.
"""

from pathlib import Path
from typing import Any


def load_model_info(model_path: str) -> dict:
    path = Path(model_path)
    info: dict[str, Any] = {"path": str(path), "name": path.name}
    if path.is_file() and path.suffix == ".gguf":
        info["format"] = "gguf"
        info["size_gb"] = round(path.stat().st_size / (1024**3), 2)
        return info
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            info["format"] = "safetensors"
            info["architectures"] = cfg.get("architectures", [])
            info["model_type"] = cfg.get("model_type", "")
            info["hidden_size"] = cfg.get("hidden_size")
            info["num_layers"] = cfg.get("num_hidden_layers")
            info["vocab_size"] = cfg.get("vocab_size")
            safetensors_files = list(path.glob("*.safetensors"))
            info["shards"] = len(safetensors_files)
            total_size = sum(f.stat().st_size for f in safetensors_files)
            info["size_gb"] = round(total_size / (1024**3), 2)
    return info
