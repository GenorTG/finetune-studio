"""The one true GGUF loader — single canonical ``llama_cpp.Llama()`` builder.

Before this module existed, `LocalGGUFProvider.load()` (models/providers.py)
and `InferenceEngine._load_gguf()` (testing/inference.py) each independently
built a `Llama(**kwargs)` call and drifted apart. Every caller — data-prep's
helper, chat, RAG, testing, benchmarks, the inference tab — now goes through
this one function, so a fix or a new loader parameter lands everywhere at once.

Load policy (Genor 2026-10-06, extended 2026-10-08): an explicit ``n_ctx`` is NEVER shrunk to make a
load fit (agentic/tool use needs it). With no ``n_ctx`` (``AUTO_N_CTX``) the model's NATIVE context is
the default and is lowered only as far as needed for every layer to fit on the GPU, never below
``MIN_AUTO_CTX``. The GPU layer count adapts after that: ``models/gguf_fit.py`` plans how many layers
fit in free VRAM, and a load that still runs out of memory steps down until it fits — layers that do
not fit run on the CPU. Only a model that cannot load even CPU-only raises, with the numbers in the
message. Window-attention models (Gemma 3/4) load with a window-sized cache for their window layers
(``swa_full=False``): measured 2026-10-08, Gemma 4 12B needs 18.3 GB at 32k with the full-size cache
and 10.0 GB at its native 131k with the window-sized one.
"""

from __future__ import annotations

import gc
import inspect
import logging
import multiprocessing
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune_studio.accel import is_oom_message, llama_gpu_kwargs
from finetune_studio.models import llama_native_log
from finetune_studio.models.gguf_fit import (
    MIN_AUTO_CTX,
    FitPlan,
    plan_auto_ctx,
    plan_gpu_layers,
    read_shape,
    step_down,
)

log = logging.getLogger(__name__)

# GPU-offload warnings are identical on every load; log each one once per process.
_LOGGED_WARNINGS: set[str] = set()

# Loader parameters every call site may set. Kept in one place so a new
# parameter is one addition here instead of N additions at N call sites.
LOADER_PARAM_NAMES = (
    "n_ctx", "n_gpu_layers", "n_batch", "n_threads",
    "seed", "rope_freq_base", "rope_freq_scale",
    "flash_attn", "mmap", "mlock", "type_k", "type_v",
)

# llama.cpp's CUDA path aborts the whole process ("illegal memory access", SIGABRT) once a prompt fills a
# 512-token micro-batch on some Q8_0 models (live repro 2026-10-06: Qwen3-0.6B Q8_0, 0/10 ok at 512, 10/10 at
# <=384, 42/42 at 256 up to a 20k-token prompt; not memory related — it reproduces at 2.5 GB of 24 GB).
# 256 costs ~8 % prompt-processing speed on a 12B. Override with FTS_LLAMA_UBATCH once upstream is fixed.
DEFAULT_N_UBATCH = 256
# Total load attempts: the planned one, context halvings (auto context only), geometric layer step-downs,
# and always a final CPU-only attempt.
MAX_LOAD_ATTEMPTS = 8
# n_ctx value meaning "the model's native context, lowered only if it cannot fit" (see module docstring).
AUTO_N_CTX = 0


def default_n_ubatch() -> int:
    raw = os.environ.get("FTS_LLAMA_UBATCH", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_N_UBATCH


class LlamaLoadError(RuntimeError):
    """The model could not be loaded at the requested context, even with every layer on the CPU."""


@dataclass
class LlamaLoadResult:
    llama: Any
    mmproj_path: str | None = None
    vision: bool = False
    final_n_ctx: int = 0
    warnings: list[str] = field(default_factory=list)
    # Placement actually achieved: -1 = every layer on the GPU, 0 = CPU only, else the GPU layer count.
    n_gpu_layers: int = -1
    total_layers: int = 0
    offload: str = "gpu"          # "gpu" | "partial" | "cpu"
    attempts: list[dict[str, Any]] = field(default_factory=list)


def _gpu_capable() -> bool:
    """True when a GPU accelerator is selected AND the installed llama.cpp was built to use it."""
    try:
        from finetune_studio.accel import get_accelerator, llama_support
        return bool(get_accelerator().is_gpu and llama_support().gpu_offload)
    except Exception:
        log.debug("GPU capability probe failed", exc_info=True)
        return False


def _free_vram_gb() -> float | None:
    """Free memory of the accelerator llama.cpp will use; None on a CPU-only host / CPU-only build."""
    if not _gpu_capable():
        return None
    try:
        from finetune_studio.accel import mem_info_gb
        free, total = mem_info_gb()
        return free if total > 0 else None
    except Exception:   # planning is an optimisation; the retry loop still protects the load
        log.debug("free VRAM probe failed", exc_info=True)
        return None


def _with_detail(exc: Exception, detail: str) -> Exception:
    """Same exception type with the native log's reason appended (llama.cpp's own message says nothing)."""
    if not detail:
        return exc
    try:
        return type(exc)(f"{exc} — {detail}")
    except TypeError:   # an exception type with a special constructor: keep the original
        return exc


def _normalise_layers(n_gpu_layers: int | None) -> int:
    """-1 / None / the legacy 99 mean "as many as fit"; any other value is an upper bound."""
    if n_gpu_layers is None or n_gpu_layers < 0 or n_gpu_layers >= 99:
        return -1
    return int(n_gpu_layers)


def _describe_plan(plan: FitPlan | None) -> str:
    if plan is None:
        return "no GPU planning (CPU-only host or build)"
    return (f"free VRAM {plan.free_gb:.1f} GiB, all {plan.total_layers} layers need ~{plan.need_all_gb:.1f} GiB "
            f"-> {plan.n_gpu_layers if plan.n_gpu_layers >= 0 else 'all'} on GPU ({plan.reason})")


def load_llama_gguf(
    gguf_path: str,
    *,
    n_ctx: int = AUTO_N_CTX,
    n_gpu_layers: int = -1,
    n_batch: int = 512,
    n_ubatch: int | None = None,
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

    ``n_gpu_layers``: -1 = automatic (as many layers on the GPU as fit), N = at most N. An explicit
    ``n_ctx`` is kept exactly as asked; ``AUTO_N_CTX`` (0) picks the native context and lowers it only to
    fit (``result.final_n_ctx`` is what was loaded); see the module docstring for the fit policy.
    """
    from llama_cpp import Llama

    llama_native_log.install()
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
    ubatch = min(n_ubatch or default_n_ubatch(), n_batch)

    # ── context: an explicit n_ctx is kept; auto = native, lowered only to fit ──
    requested = _normalise_layers(n_gpu_layers)
    free_gb = _free_vram_gb()
    shape = read_shape(gguf_path)
    auto_ctx = n_ctx <= 0
    ctx_floor = min(shape.native_ctx, MIN_AUTO_CTX) if shape.native_ctx > 0 else MIN_AUTO_CTX
    native_ctx = shape.native_ctx
    if auto_ctx:
        n_ctx, _ctx_reason = plan_auto_ctx(gguf_path, type_k=type_k, type_v=type_v, requested=requested, free_gb=free_gb)

    kwargs: dict[str, Any] = {
        "model_path": gguf_path,
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "n_ubatch": ubatch,
        "n_threads": resolved_threads,
        "mmap": mmap,
        "verbose": False,
    }
    if chat_handler is not None:
        kwargs["chat_handler"] = chat_handler
    if shape.uses_swa and "swa_full" in inspect.signature(Llama.__init__).parameters:
        kwargs["swa_full"] = False   # window layers cache the window, not n_ctx cells (module docstring)
    # Route to the accelerator accel chose (main_gpu on multi-GPU hosts) and
    # surface a GPU-host/CPU-llama.cpp mismatch instead of degrading silently.
    gpu_extras, gpu_warnings = llama_gpu_kwargs()
    kwargs.update(gpu_extras)
    for w in gpu_warnings:
        if w not in _LOGGED_WARNINGS:
            _LOGGED_WARNINGS.add(w)
            log.warning("load_llama_gguf: %s", w)
        if w not in result.warnings:
            result.warnings.append(w)
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

    # ── placement: plan from free VRAM, then let real attempts correct the estimate ──
    plan = (plan_gpu_layers(gguf_path, n_ctx=n_ctx, type_k=type_k, type_v=type_v, requested=requested,
                            free_gb=free_gb) if free_gb is not None else None)
    layers = plan.n_gpu_layers if plan is not None else requested
    total_layers = plan.total_layers if plan is not None else shape.total_layers

    log.info(
        "load_llama_gguf %s (n_ctx=%d%s n_gpu_layers=%s n_batch=%d n_ubatch=%d n_threads=%d seed=%s "
        "rope_base=%s rope_scale=%s flash=%s mmap=%s mlock=%s type_k=%s type_v=%s) — %s",
        gguf_path, n_ctx, f" auto, native {native_ctx}" if auto_ctx else "", "auto" if requested < 0 else requested, n_batch, ubatch, resolved_threads,
        seed, rope_freq_base, rope_freq_scale, flash_attn, mmap, mlock, type_k, type_v, _describe_plan(plan),
    )

    for _attempt in range(MAX_LOAD_ATTEMPTS):
        kwargs["n_gpu_layers"] = layers
        # A CPU-only attempt must not touch the GPU at all: with 0 layers llama.cpp still reserves the
        # prompt-processing compute buffer (~1 GiB) and the KV cache on the device, which fails exactly when
        # VRAM is nearly gone (live test, 0.3 GiB free).
        kwargs["offload_kqv"] = layers != 0
        kwargs["op_offload"] = layers != 0
        position = llama_native_log.mark()
        try:
            result.llama = Llama(**kwargs)
            break
        except Exception as e:
            detail = llama_native_log.failure_detail(position)
            oom = is_oom_message(str(e)) or llama_native_log.oom_since(position)
            result.attempts.append({"n_gpu_layers": layers, "oom": oom, "error": str(e), "native": detail})
            if not oom:
                # Corrupt/unsupported file etc.: retrying with other layer counts cannot help.
                raise _with_detail(e, detail) from e
            if auto_ctx and n_ctx > ctx_floor and len(result.attempts) < MAX_LOAD_ATTEMPTS - 1:
                lowered = max(ctx_floor, n_ctx // 2 // 1024 * 1024)
                log.warning("GGUF load ran out of memory at n_ctx=%d (auto); retrying at n_ctx=%d. %s", n_ctx, lowered, detail)
                n_ctx = lowered
                kwargs["n_ctx"] = n_ctx
                gc.collect()
                continue
            nxt = step_down(layers, total_layers)
            if nxt is not None and len(result.attempts) >= MAX_LOAD_ATTEMPTS - 1:
                nxt = 0   # out of tries: the last one is always CPU-only, so a model that can load at all does
            if nxt is None:
                raise LlamaLoadError(
                    f"Out of memory loading {Path(gguf_path).name} at n_ctx={n_ctx} even with every layer on the "
                    f"CPU ({_describe_plan(plan)}). The context length was left unchanged on purpose (floor {ctx_floor}): use a "
                    f"smaller quantisation or model, or ask for a smaller n_ctx."
                    f"{' Native: ' + detail if detail else ''}"
                ) from e
            log.warning(
                "GGUF load ran out of memory with n_gpu_layers=%s at n_ctx=%d; retrying with %d layer(s) on the GPU "
                "(context unchanged). %s", "all" if layers < 0 else layers, n_ctx, nxt, detail,
            )
            layers = nxt
            gc.collect()   # drop the half-built context so its VRAM is free for the next attempt
    if result.llama is None:   # the loop only ends by break or raise; a safety net, not a code path
        raise LlamaLoadError(f"{Path(gguf_path).name} did not load")

    result.final_n_ctx = n_ctx
    if auto_ctx and native_ctx > 0 and n_ctx < native_ctx:
        msg = (f"Context lowered to {n_ctx} (the model's native {native_ctx}) so that "
               f"{'every layer fits on the GPU' if layers < 0 else 'the load fits'}.")
        result.warnings.append(msg)
        log.warning("load_llama_gguf: %s", msg)
    result.n_gpu_layers = layers
    result.total_layers = total_layers
    gpu_capable = _gpu_capable()
    if not gpu_capable or layers == 0:
        result.offload = "cpu"
    elif layers > 0 and total_layers and layers < total_layers:
        result.offload = "partial"
    if gpu_capable and result.offload != "gpu":   # a CPU-only host is not a degradation worth a warning here
        gpu = layers if layers >= 0 else total_layers
        why = ("as requested" if requested >= 0 and layers == requested
               else f"at n_ctx={n_ctx} it needs more VRAM than is free")
        msg = (f"Only {gpu}/{total_layers} layers of {Path(gguf_path).name} are on the GPU "
               f"({'none fit' if gpu == 0 and requested < 0 else 'the rest run on the CPU'}): {why}. "
               f"Generation is slower; the context was kept at {n_ctx}.")
        result.warnings.append(msg)
        log.warning("load_llama_gguf: %s", msg)
    return result


# Default context. helper.py's DEFAULT_HELPER_EXTRA, and every /load route's
# `body.get("n_ctx", ...)` fallback, must agree on this value. It used to be a fixed 32768 (and before
# that 16384 or 4096 in various places); it is now "auto" = the model's native context, lowered only to
# fit, never below MIN_AUTO_CTX (the 32k floor the agentic/helper paths actually need).
DEFAULT_N_CTX = AUTO_N_CTX


def resolve_loader_overrides(
    body: dict[str, Any], *, caller: str, model_path: str = "", default_ctx: bool = True,
) -> dict[str, Any]:
    """The one place every `/load` route derives its Llama kwargs from a
    request body. ``n_gpu_layers`` is -1 (automatic: as many layers on the GPU
    as fit) unless the caller asked for fewer — an explicit N is an upper bound,
    the legacy 99 means "all" — and the 32k context default applies when none is
    sent. ``caller`` / ``model_path`` only label log lines.

    Returns a dict of exactly the keys `load_llama_gguf` / `InferenceEngine
    .load` / `ModelManager.load(extra=...)` accept, built from whatever the
    caller's body actually set (missing keys are simply absent, so a
    caller's own defaults/persisted values still apply).

    ``default_ctx=False`` for provider-backed callers (ModelManager.load):
    those already fall through to the provider's *persisted* n_ctx when the
    caller sends none, and forcing DEFAULT_N_CTX in here would silently
    overwrite that persisted value on every load instead of leaving it alone.
    """
    log.debug("%s: loader overrides for %s from body keys %s", caller, model_path or "<no path>", sorted(body))
    out: dict[str, Any] = {"n_gpu_layers": _normalise_layers(body.get("n_gpu_layers"))}
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
    try:
        from finetune_studio.data.rag_portable.model_cache import release_rag_models
        release_rag_models("unload all models")
    except Exception:
        log.exception("unload_all_models: failed to release the cached RAG embedder/reranker")
