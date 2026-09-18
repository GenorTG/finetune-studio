from typing import Any

"""Load GGUF and HuggingFace models.

WHAT THIS FILE DOES
==================
Wraps the inference engine's loading logic with metadata tracking:
  - Model name
  - Path
  - Format (GGUF or HF)
  - Size in bytes
  - Quantization level (for GGUF)
  - Parameter count

KEY CONCEPTS
============
- Model metadata: useful for the web UI to display model info.
- Multiple model support: load several models and switch between them.
- Lazy loading: only load a model when it's actually needed.
"""

from pathlib import Path


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

def load_for_inference(model_path: str, device: str = "auto", **kwargs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(model_path)
    if path.is_file() and path.suffix == ".gguf":
        return load_gguf_inference(str(path), **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    return model, tokenizer

def load_gguf_inference(gguf_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
    from llama_cpp import Llama
    model = Llama(model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    return model, None
