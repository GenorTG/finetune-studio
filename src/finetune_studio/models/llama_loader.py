"""The one true GGUF loader — single canonical ``llama_cpp.Llama()`` builder.

Before this module existed, `LocalGGUFProvider.load()` (models/providers.py)
and `InferenceEngine._load_gguf()` (testing/inference.py) each independently
built a `Llama(**kwargs)` call. They drifted: only one had OOM-retry
(shrink context instead of ever falling back to mixed CPU/GPU offload),
only the other had mmproj/vision auto-detection and KV-cache-type support.
Every caller — data-prep's helper, chat, RAG, testing, benchmarks, the
inference tab — now goes through this one function, so a fix or a new
loader parameter lands everywhere at once instead of needing to be copied
into N places (and inevitably missing one).
"""

from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Loader parameters every call site may set. Kept in one place so a new
# parameter is one addition here instead of N additions at N call sites.
LOADER_PARAM_NAMES = (
    "n_ctx", "n_gpu_layers", "n_batch", "n_threads",
    "seed", "rope_freq_base", "rope_freq_scale",
    "flash_attn", "mmap", "mlock", "type_k", "type_v",
)


@dataclass
class LlamaLoadResult:
    llama: Any
    mmproj_path: str | None = None
    vision: bool = False
    final_n_ctx: int = 0
    warnings: list[str] = field(default_factory=list)


def load_llama_gguf(
    gguf_path: str,
    *,
    n_ctx: int = 32768,
    n_gpu_layers: int = -1,
    n_batch: int = 512,
    n_threads: int | None = None,
    seed: int | None = None,
    rope_freq_base: float = 0.0,
    rope_freq_scale: float = 0.0,
    flash_attn: bool = True,
    mmap: bool = True,
    mlock: bool = False,
    type_k: int = 0,
    type_v: int = 0,
    detect_mmproj: bool = True,
) -> LlamaLoadResult:
    """Build one `llama_cpp.Llama` instance. The single canonical loader.

    Contract (GH-AAA): NEVER mixed CPU/GPU offload. If the model + KV cache
    don't fit in VRAM at the requested n_ctx, retry with a halved n_ctx
    (floor 512) — model layers always stay fully on GPU; only the KV cache
    shrinks. Callers that want a different n_gpu_layers policy (e.g. forcing
    -1 regardless of what a client requested) must do that before calling
    this — this function loads with exactly the n_gpu_layers it's given.
    """
    from llama_cpp import Llama

    result = LlamaLoadResult(llama=None)

    chat_handler = None
    if detect_mmproj:
        gguf_dir = Path(gguf_path).parent
        base_name = Path(gguf_path).stem
        for suffix in ("-Q4_K_M", "-Q5_K_M", "-Q6_K", "-Q8_0", "-F16", "-BF16"):
            base_name = base_name.replace(suffix, "")
        mmproj_path = None
        for candidate in gguf_dir.glob("mmproj*.gguf"):
            mmproj_path = str(candidate)
            break
        if not mmproj_path:
            for candidate in gguf_dir.glob(f"*mmproj*{base_name}*.gguf"):
                mmproj_path = str(candidate)
                break
        if mmproj_path:
            try:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=False)
                result.mmproj_path = mmproj_path
                result.vision = True
                log.info("Vision enabled: mmproj=%s", Path(mmproj_path).name)
            except Exception as e:  # noqa: BLE001 — vision is optional, never blocks text loading
                log.warning("mmproj load failed (%s), running text-only", e)
                result.warnings.append(f"mmproj load failed: {e}")

    resolved_threads = n_threads if n_threads and n_threads > 0 else multiprocessing.cpu_count()

    kwargs: dict[str, Any] = {
        "model_path": gguf_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": n_batch,
        "n_threads": resolved_threads,
        "mmap": mmap,
        "verbose": False,
    }
    if chat_handler is not None:
        kwargs["chat_handler"] = chat_handler
    if flash_attn:
        kwargs["flash_attn"] = True
    if mlock:
        kwargs["use_mlock"] = True
    if seed is not None and seed >= 0:
        kwargs["seed"] = seed
    if rope_freq_base > 0:
        kwargs["rope_freq_base"] = rope_freq_base
    if rope_freq_scale > 0:
        kwargs["rope_freq_scale"] = rope_freq_scale
    if type_k > 0:
        kwargs["type_k"] = type_k
    if type_v > 0:
        kwargs["type_v"] = type_v

    log.info(
        "load_llama_gguf %s (n_ctx=%d n_gpu_layers=%d n_batch=%d n_threads=%d "
        "seed=%s rope_base=%s rope_scale=%s flash=%s mmap=%s mlock=%s type_k=%s type_v=%s)",
        gguf_path, n_ctx, n_gpu_layers, n_batch, resolved_threads,
        seed, rope_freq_base, rope_freq_scale, flash_attn, mmap, mlock, type_k, type_v,
    )

    last_err: Exception | None = None
    for attempt in range(6):
        try:
            result.llama = Llama(**kwargs)
            result.final_n_ctx = kwargs["n_ctx"]
            last_err = None
            break
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda" not in msg and "vram" not in msg:
                raise
            last_err = e
            new_ctx = max(512, kwargs["n_ctx"] // 2)
            if new_ctx == kwargs["n_ctx"]:
                break  # already at the floor; give up
            log.warning(
                "GGUF load OOM at n_ctx=%d (attempt %d/6); retrying with n_ctx=%d. "
                "Model layers stay on GPU — only KV cache shrinks.",
                kwargs["n_ctx"], attempt + 1, new_ctx,
            )
            kwargs["n_ctx"] = new_ctx
    if result.llama is None and last_err is not None:
        raise last_err
    return result


# Default context floor. helper.py's DEFAULT_HELPER_EXTRA, and every /load
# route's `body.get("n_ctx", ...)` fallback, must agree on this number —
# it used to be 16384 or 4096 in various places, silently below the 32k
# floor the agentic/helper paths actually needed.
DEFAULT_N_CTX = 32768


def resolve_loader_overrides(
    body: dict[str, Any], *, caller: str, model_path: str = "", default_ctx: bool = True,
) -> dict[str, Any]:
    """The one place every `/load` route derives its Llama kwargs from a
    request body. Enforces the GH-AAA no-mixed-offload contract (any
    n_gpu_layers other than -1 is ignored, with a warning naming the
    caller) and the 32k context floor. Was copy-pasted with slightly
    different wording into models.py, chat_v2.py, and testing.py — one of
    the three (chat_v2.py) had drifted and didn't enforce the GPU guard at
    all, silently allowing mixed offload from that one route.

    Returns a dict of exactly the keys `load_llama_gguf` / `InferenceEngine
    .load` / `ModelManager.load(extra=...)` accept, built from whatever the
    caller's body actually set (missing keys are simply absent, so a
    caller's own defaults/persisted values still apply).

    ``default_ctx=False`` for provider-backed callers (ModelManager.load):
    those already fall through to the provider's *persisted* n_ctx when the
    caller sends none, and forcing DEFAULT_N_CTX in here would silently
    overwrite that persisted value on every load instead of leaving it alone.
    """
    requested_layers = body.get("n_gpu_layers", -1)
    if requested_layers not in (None, -1):
        log.warning(
            "%s: n_gpu_layers=%s requested for %s; ignoring and loading all "
            "layers on GPU (no mixed offload).",
            caller, requested_layers, model_path or "<no model path given>",
        )
    out: dict[str, Any] = {"n_gpu_layers": -1}
    if "n_ctx" in body and body["n_ctx"] is not None:
        out["n_ctx"] = int(body["n_ctx"])
    elif default_ctx:
        out["n_ctx"] = DEFAULT_N_CTX
    for key in LOADER_PARAM_NAMES:
        if key in ("n_gpu_layers", "n_ctx"):
            continue
        if key in body and body[key] is not None:
            out[key] = body[key]
    return out


def unload_all_models() -> None:
    """Free the resident model and reset ModelManager's bookkeeping.

    There is now exactly ONE model-holding object in this process — the
    ``inference_engine`` global IS ``ModelManager().engine`` (see
    webui/app.py and models/manager.py's ``engine`` property), not a
    second independent tracker. Two calls remain here on purpose, for two
    different reasons, not two engines to coordinate:

    1. ``inference_engine.unload()`` frees the actual model/VRAM — correct
       regardless of whether it was loaded via a named provider or a raw
       path, since both go through this one object now.
    2. ``get_manager().unload()`` clears ModelManager's own
       ``_provider``/``_active_id`` bookkeeping. Without this, a model
       loaded directly via ``inference_engine.load()`` (bypassing
       ``ModelManager.load()``, as testing.py/chat_v2.py do) and then
       freed by step 1 would leave ModelManager still reporting a stale
       "active" provider in ``/api/providers`` even though nothing is
       actually loaded anymore.
    """
    try:
        from finetune_studio.webui.app import inference_engine
        if getattr(inference_engine, "model", None) is not None:
            inference_engine.unload()
    except Exception:
        log.exception("unload_all_models: failed to unload the global inference engine")
    try:
        from finetune_studio.models.manager import get_manager
        get_manager().unload()
    except Exception:
        log.exception("unload_all_models: failed to unload the ModelManager active provider")
