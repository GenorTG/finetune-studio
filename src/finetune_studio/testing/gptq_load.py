"""Load local GPTQ checkpoints via gptqmodel (Torch backend).

Local GPTQ dirs (config.json + quantize_config.json / GPTQ metadata) must not
go through plain ``AutoModelForCausalLM`` — that path hits Marlin JIT or needs
optimum. Prefer ``GPTQModel.from_quantized(..., backend=BACKEND.GPTQ_TORCH)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def gptq_inference_missing_message() -> str:
    """Actionable error when gptqmodel / GPTQ_TORCH cannot be used for load."""
    return (
        "GPTQ checkpoint detected but gptqmodel (BACKEND.GPTQ_TORCH) is "
        "unavailable. Install with: uv pip install -e '.[gptq]' "
        "(gptqmodel), then retry inference."
    )


def is_local_gptq_checkpoint(model_path: str) -> bool:
    """True for a local dir that looks like a GPTQ export.

    Requires ``config.json`` plus either ``quantize_config.json`` or GPTQ
    markers inside ``config.json``'s ``quantization_config``.
    """
    path = Path(model_path)
    if not path.is_dir():
        return False
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return False
    if (path / "quantize_config.json").is_file():
        return True
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        return False
    markers = (
        str(qcfg.get("quant_method", "")).lower(),
        str(qcfg.get("method", "")).lower(),
        str(qcfg.get("format", "")).lower(),
        str(qcfg.get("checkpoint_format", "")).lower(),
    )
    return any("gptq" in marker for marker in markers)


def load_gptq_model_torch(
    model_path: str,
    *,
    device_map: str | dict[str, Any] | None = None,
) -> Any:
    """Load a GPTQ dir with ``GPTQModel.from_quantized`` + ``BACKEND.GPTQ_TORCH``.

    Raises ``RuntimeError`` with :func:`gptq_inference_missing_message` when
    gptqmodel or ``BACKEND.GPTQ_TORCH`` is missing.
    """
    try:
        from gptqmodel import BACKEND, GPTQModel
    except ImportError as exc:
        raise RuntimeError(gptq_inference_missing_message()) from exc

    backend = getattr(BACKEND, "GPTQ_TORCH", None)
    if backend is None:
        raise RuntimeError(gptq_inference_missing_message())

    kwargs: dict[str, Any] = {
        "backend": backend,
        "trust_remote_code": True,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    return GPTQModel.from_quantized(model_path, **kwargs)
