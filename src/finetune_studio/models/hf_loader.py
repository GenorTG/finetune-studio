"""HF causal-LM loading with the accelerator-first OOM ladder.

Single responsibility: one ``from_pretrained`` policy for training, inference
and export. Order: full offload to the chosen GPU -> (on OOM) bitsandbytes
4-bit where the backend supports it -> ``device_map="auto"`` RAM spill. A host
with no GPU loads once on the CPU in fp32 and never touches bitsandbytes.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from finetune_studio import accel
from finetune_studio.training.accel_plan import bnb_4bit_config, bnb_usable

log = logging.getLogger(__name__)


def load_causal_lm(
    model_path: str, *,
    acc: accel.Accelerator | None = None,
    force_4bit: bool = False,
    on_status: Callable[[str], None] | None = None,
) -> Any:
    """Load ``model_path``; see module docstring for the fallback order."""
    from transformers import AutoModelForCausalLM

    acc = acc or accel.get_accelerator()
    say = on_status or (lambda _m: None)
    dtype = accel.torch_dtype(acc)
    dmap = accel.device_map(acc)

    def _load(**kw: Any) -> Any:
        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kw)

    if force_4bit and bnb_usable(acc):
        say(f"Loading model in 4-bit on {acc.name}...")
        return _load(quantization_config=bnb_4bit_config(acc), device_map=dmap)

    say(f"Loading model on {acc.name}..." if acc.is_gpu else "No GPU available — loading model on CPU (fp32)...")
    try:
        return _load(torch_dtype=dtype, device_map=dmap)
    except accel.oom_errors() as exc:
        if not acc.is_gpu or not accel.is_oom_error(exc):
            raise
        log.warning("model load OOM on %s: %s", acc.torch_device, exc)
    accel.empty_cache()

    if bnb_usable(acc):
        say("GPU OOM — retrying in 4-bit...")
        try:
            return _load(quantization_config=bnb_4bit_config(acc), device_map=dmap)
        except accel.oom_errors() as exc:
            if not accel.is_oom_error(exc):
                raise
            log.warning("4-bit load OOM on %s: %s", acc.torch_device, exc)
        except (ImportError, ValueError, OSError) as exc:
            # bitsandbytes missing/unsupported for this arch — fall through to RAM spill.
            log.warning("4-bit load unavailable (%s: %s)", type(exc).__name__, exc)
        accel.empty_cache()

    say("Loading model with CPU offload (RAM+VRAM mix)...")
    return _load(torch_dtype=dtype, device_map=accel.auto_device_map(acc))


def merge_dtype(acc: accel.Accelerator | None = None) -> Any:
    """Merge/export dtype: bf16 wherever it works (incl. CPU RAM), fp16 on GPUs without bf16."""
    import torch
    acc = acc or accel.get_accelerator()
    return torch.bfloat16 if (not acc.is_gpu or acc.supports_bf16) else accel.torch_dtype(acc)


def load_merge_base(base_path: str, *, acc: accel.Accelerator | None = None) -> Any:
    """Load a 16-bit merge base on the GPU when it fits; CPU RAM only on OOM or no GPU."""
    from transformers import AutoModelForCausalLM

    acc = acc or accel.get_accelerator()
    dtype = merge_dtype(acc)

    def _load(dmap: Any) -> Any:
        return AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=dtype, device_map=dmap, trust_remote_code=True)

    if acc.is_gpu:
        try:
            return _load(accel.device_map(acc))
        except accel.oom_errors() as exc:
            if not accel.is_oom_error(exc):
                raise
            log.warning("merge base does not fit on %s (%s); merging in CPU RAM", acc.torch_device, exc)
            accel.empty_cache()
    return _load("cpu")
