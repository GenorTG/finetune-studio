"""HF causal-LM loading with the accelerator-first OOM ladder.

Single responsibility: one ``from_pretrained`` policy for training, inference
and export. Order: full offload to the chosen GPU -> (on OOM) bitsandbytes
4-bit where the backend supports it -> ``device_map="auto"`` RAM spill. A host
with no GPU loads once on the CPU in fp32 and never touches bitsandbytes.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from finetune_studio import accel
from finetune_studio.training.accel_plan import bnb_4bit_config, bnb_usable

log = logging.getLogger(__name__)

# What a LoRA step needs besides the weights: activations, the fp32 logits (a ~250k vocab makes
# them >1 GiB per sequence batch), adapter + optimizer state and the CUDA context. A 9B bf16 base
# (18 GiB) loads fine on a 24 GB card and then dies in the first forward pass (live DPO run, Qwen3.5-9B).
TRAIN_HEADROOM_GIB = 6.0


def weights_gib(model_path: str) -> float:
    """On-disk size of the weight shards in GiB (0.0 when there are none, e.g. a hub id)."""
    root = Path(model_path)
    if not root.is_dir():
        return 0.0
    shards = list(root.glob("*.safetensors")) or list(root.glob("*.bin"))
    return sum(p.stat().st_size for p in shards) / 1024**3


def training_needs_4bit(model_path: str, acc: accel.Accelerator | None = None) -> tuple[bool, str]:
    """Whether a bf16 load would leave too little VRAM to train; the second item is the user-facing reason."""
    acc = acc or accel.get_accelerator()
    size = weights_gib(model_path)
    if not (size and bnb_usable(acc)) or size + TRAIN_HEADROOM_GIB <= acc.free_gb:
        return False, ""
    return True, (
        f"Model weights are {size:.1f} GiB in bf16 but only {acc.free_gb:.1f} GiB of VRAM are free "
        f"(training needs ~{TRAIN_HEADROOM_GIB:.0f} GiB on top) — loading in 4-bit (QLoRA)."
    )


def load_causal_lm(
    model_path: str, *,
    acc: accel.Accelerator | None = None,
    force_4bit: bool = False,
    on_status: Callable[[str], None] | None = None,
) -> Any:
    """Load ``model_path``; see module docstring for the fallback order."""
    from transformers import AutoModelForCausalLM

    acc = acc or accel.get_accelerator()
    accel.activate(acc)  # bnb / PEFT / stray ``device="cuda"`` follow the current device (per thread)
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


def load_peft_adapter(base: Any, adapter_dir: str) -> Any:
    """``PeftModel.from_pretrained`` with the adapter read straight onto the base model's device.

    PEFT's default is the bare string ``"cuda"``, which safetensors resolves to ``cuda:0`` whatever
    the current device is — a context (and the adapter) on the wrong card when the model sits on
    another index, e.g. a GTX 1070 next to the 3090 the base was loaded on.
    """
    from peft import PeftModel

    device = getattr(base, "device", None)
    return PeftModel.from_pretrained(base, adapter_dir, torch_device=str(device) if device else None)


def load_merge_base(base_path: str, *, acc: accel.Accelerator | None = None) -> Any:
    """Load a 16-bit merge base on the GPU when it fits; CPU RAM only on OOM or no GPU."""
    from transformers import AutoModelForCausalLM

    acc = acc or accel.get_accelerator()
    accel.activate(acc)  # PeftModel.from_pretrained loads adapter weights on the current device
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
