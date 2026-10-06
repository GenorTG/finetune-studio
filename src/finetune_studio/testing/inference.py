"""Run inference on a single prompt.

WHAT THIS FILE DOES
==================
A thin wrapper around the inference engine for testing purposes.
Loads a model, sends a prompt, returns the response with metadata.

KEY CONCEPTS
============
- Test isolation: each test loads its own model instance to avoid
  interference between tests.
- Determinism: testing often requires reproducible outputs. Set
  temperature=0 for deterministic generation.
"""

import os
import threading
import time
from collections import OrderedDict

import torch

from finetune_studio.models.llama_loader import DEFAULT_N_CTX

# Module-level cache for GGUF metadata reads (fast header-only parser is still
# cheap, but for 16 GB files we don't want to walk the header twice).
_GGUF_META_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_GGUF_META_CACHE_MAX = 32

# Auto-unload after idle (FTS_IDLE_TIMEOUT env, default 5 min; 0 disables).
DEFAULT_IDLE_TIMEOUT = 300


def idle_timeout() -> int:
    """Idle seconds before auto-unload; read live so env changes apply without a restart."""
    try:
        return max(0, int(os.environ.get("FTS_IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT)))
    except ValueError:
        return DEFAULT_IDLE_TIMEOUT


IDLE_TIMEOUT = idle_timeout()


def release_idle_memory() -> None:
    """Hand cached memory back to the OS/driver (GC, accelerator cache, glibc heap, parsed-file cache)."""
    import ctypes
    import gc

    gc.collect()
    try:
        from finetune_studio import accel
        accel.empty_cache()
    except Exception:  # noqa: BLE001,S110
        pass
    try:
        from finetune_studio.data.fs import file_library
        file_library._PARSED_CACHE.clear()
    except Exception:  # noqa: BLE001,S110
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001,S110
        pass


class InferenceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.is_gguf = False
        self.vision = False
        self.mmproj_path = None
        # Real loader params the currently-held model was actually built
        # with — not the ask. OOM-retry can shrink n_ctx below what the
        # caller requested; describe()-style callers need the truth.
        self.n_ctx: int | None = None
        self.n_gpu_layers: int | None = None
        self._gguf_template = None
        self._last_used = 0.0
        self._idle_timer = None
        self._busy = 0
        self._busy_lock = threading.Lock()
        # In-flight load bookkeeping so the activity feed can show a
        # "loading…" row while a (possibly multi-minute) load blocks — the
        # engine only exposes ``model`` once the load has fully completed.
        self._loading_path: str | None = None
        self._loading_started: float = 0.0

    def load(self, model_path, device="auto", n_ctx=DEFAULT_N_CTX, n_gpu_layers=-1, n_batch=512, mmap=True, mlock=False,
              n_threads=None, flash_attn=True, seed=None, rope_freq_base=0.0, rope_freq_scale=0.0,
              type_k=0, type_v=0, max_seq_length=None, load_in_4bit=False):
        from pathlib import Path
        self._loading_path = model_path
        self._loading_started = time.time()
        try:
            self.unload()
            # The RAG embedder/reranker cache shares this GPU: free it before the load.
            from finetune_studio.data.rag_portable.model_cache import release_rag_models
            release_rag_models("model load")
            path = Path(model_path)
            if path.is_file() and path.suffix == ".gguf":
                self._load_gguf(str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_batch=n_batch,
                                mmap=mmap, mlock=mlock, n_threads=n_threads, flash_attn=flash_attn,
                                seed=seed, rope_freq_base=rope_freq_base, rope_freq_scale=rope_freq_scale,
                                type_k=type_k, type_v=type_v)
            else:
                self._load_hf(
                    model_path,
                    device,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                )
            self.model_path = model_path
            self._last_used = time.time()
            self._start_idle_timer()
        finally:
            self._loading_path = None

    @staticmethod
    def _looks_like_qwen3(model_path: str) -> bool:
        """True when path/config indicates a Qwen3 (or Qwen3.5) checkpoint."""
        import json
        from pathlib import Path

        lowered = model_path.lower().replace("\\", "/")
        if "qwen3" in lowered:
            return True
        cfg_path = Path(model_path) / "config.json"
        if not cfg_path.is_file():
            return False
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            model_type = str(cfg.get("model_type", "")).lower()
            architectures = [str(a).lower() for a in (cfg.get("architectures") or [])]
            return model_type.startswith("qwen3") or any("qwen3" in a for a in architectures)
        except Exception:  # noqa: BLE001
            return False

    def _load_hf(self, model_path, device, max_seq_length=None, load_in_4bit=False):
        """Load HF checkpoints with plain transformers (E2E-40).

        Prefer bf16 on GPU; use bitsandbytes 4-bit only when ``load_in_4bit`` is
        True or a full-precision load OOMs. Never import Unsloth — it monkey-patches
        transformers globally and poisons all later inference in this process.
        """
        from finetune_studio.config import settings

        # max_seq_length kept for API compatibility with callers / Unsloth era.
        _ = max_seq_length if max_seq_length is not None else settings.default_max_seq_length

        from finetune_studio.hf_env import load_tokenizer
        self.tokenizer = load_tokenizer(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from finetune_studio import accel
        from finetune_studio.models.hf_loader import load_causal_lm

        accel.enable_fast_matmul()
        self.model = load_causal_lm(model_path, force_4bit=load_in_4bit)
        self.is_gguf = False

    def _load_gguf(self, gguf_path, n_ctx=DEFAULT_N_CTX, n_gpu_layers=-1, n_batch=512, mmap=True, mlock=False,
                   n_threads=None, flash_attn=True, seed=None, rope_freq_base=0.0, rope_freq_scale=0.0,
                   type_k=0, type_v=0):
        from finetune_studio.models.llama_loader import load_llama_gguf
        self.is_gguf = True

        result = load_llama_gguf(
            gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_batch=n_batch,
            n_threads=n_threads, seed=seed, rope_freq_base=rope_freq_base,
            rope_freq_scale=rope_freq_scale, flash_attn=flash_attn, mmap=mmap,
            mlock=mlock, type_k=type_k, type_v=type_v,
        )
        self.model = result.llama
        self.vision = result.vision
        self.mmproj_path = result.mmproj_path
        self.n_ctx = result.final_n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.tokenizer = None
        # Cache the GGUF's own chat template + tokens so we don't re-extract per call.
        try:
            from finetune_studio.templates.renderer import extract_template_from_gguf
            self._gguf_template = extract_template_from_gguf(gguf_path)
        except Exception:  # noqa: BLE001
            self._gguf_template = None

    def _start_idle_timer(self):
        """Start/restart the idle unload timer."""
        if self._idle_timer:
            self._idle_timer.cancel()
        timeout = idle_timeout()
        if timeout > 0 and self.model is not None:
            self._idle_timer = threading.Timer(timeout, self._auto_unload)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _auto_unload(self):
        """Unload model if idle too long."""
        if self.model is None:
            return
        timeout = idle_timeout()
        if timeout <= 0:
            return
        if self._busy:
            # Mid-generation: never pull the model out from under it.
            self._start_idle_timer()
            return
        elapsed = time.time() - self._last_used
        if elapsed >= timeout:
            print(f"[inference] Auto-unloading {self.model_path} (idle {int(elapsed)}s)")
            self.unload()

    def unload(self):
        """Unload model and free VRAM/RAM immediately.

        Previous implementation just set `self.model = None` and relied on
        Python GC to eventually call `Llama.__del__` (which frees the native
        llama.cpp context). Under load this could take seconds to minutes,
        leaving VRAM occupied and the model appearing "still loaded" to the
        next inference request or to the dashboard's VRAM chip. This version:

        1. Cancels the idle timer.
        2. Drops every reference to the model + tokenizer + gguf metadata.
        3. Explicitly `del`s the object and forces an immediate GC pass so
           `Llama.__del__` runs while we're still on the calling thread
           (the native free is deterministic this way).
        4. Releases the accelerator's cache (any vendor) so PyTorch's
           caching allocator hands memory back to the driver.

        After unload, `self.model is None` AND the Llama native context is
        gone — verified via nvidia-smi drop in VRAM within a second.
        """
        import gc

        if self._idle_timer is not None:
            try:
                self._idle_timer.cancel()
            except Exception:  # noqa: BLE001,S110
                pass
        self._idle_timer = None

        # Drop references first.
        _model = self.model
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.is_gguf = False
        self.vision = False
        self.mmproj_path = None
        self.n_ctx = None
        self.n_gpu_layers = None
        self._gguf_template = None
        self._last_used = 0.0

        # Explicit del + GC pass so Llama.__del__ runs now, not "later".
        # Without this, a busy process can hold the model in VRAM for
        # many seconds after unload returns.
        if _model is not None:
            try:
                del _model
            except Exception:  # noqa: BLE001,S110
                pass
        try:
            gc.collect()
        except Exception:  # noqa: BLE001,S110
            pass

        try:
            from finetune_studio import accel
            accel.empty_cache()
        except Exception:  # noqa: BLE001,S110
            pass

    @property
    def idle_seconds(self):
        """Seconds since last use, or 0 if no model loaded."""
        if self.model is None or self._last_used == 0:
            return 0
        return int(time.time() - self._last_used)

    def generate(self, messages, max_tokens=1024, temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, stop=None, think=False):
        if self.model is None:
            raise RuntimeError("No model loaded")
        self._last_used = time.time()
        self._start_idle_timer()
        with self._busy_lock:
            self._busy += 1
        try:
            if self.is_gguf:
                return self._generate_gguf(messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop)
            return self._generate_hf(messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop, think=think)
        finally:
            with self._busy_lock:
                self._busy -= 1
            self._last_used = time.time()

    def _generate_hf(self, messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop, think=False):
        # Qwen3 chat templates honour enable_thinking; keep the kwarg when present.
        from finetune_studio.training.formatting import render_chat_text
        text = render_chat_text(self.tokenizer, messages, generation=True, enable_thinking=think)
        from finetune_studio import accel
        accel.activate()  # request threads start on device 0; generation must run on the loaded card
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=max(temperature, 0.01),
                top_p=top_p, top_k=top_k, repetition_penalty=repeat_penalty,
                do_sample=temperature > 0, pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        # Strip Qwen3 thinking traces: <think>...</think> blocks
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # Bare </think> preambles (thinking disabled but model still emits the closer)
        response = re.sub(r"^</think>\s*", "", response).strip()
        response = response.lstrip("\n")
        # Honor stop sequences the same way the GGUF path does (llama.cpp
        # truncates at the first match) — HF has no native `stop=` kwarg for
        # `generate()`, so without this the parameter was silently ignored
        # for every non-GGUF model.
        if stop:
            cut = min(
                (idx for idx in (response.find(s) for s in stop if s) if idx != -1),
                default=-1,
            )
            if cut != -1:
                response = response[:cut]
        return response

    @staticmethod
    def estimate_memory(model_path, n_ctx=DEFAULT_N_CTX, n_gpu_layers=-1):
        """Estimate VRAM/RAM usage for a model. Returns dict with estimates in GB."""
        from pathlib import Path
        path = Path(model_path)
        result = {"weights_gb": 0.0, "kv_cache_gb": 0.0, "total_vram_gb": 0.0, "total_ram_gb": 0.0, "total_layers": 0}

        if path.is_file() and path.suffix == ".gguf":
            size_gb = path.stat().st_size / (1024**3)
            # Read GGUF metadata via fast header parser (no model load)
            meta_info = InferenceEngine.read_model_metadata(str(path))
            total_layers = meta_info["total_layers"]
            num_kv_heads = meta_info["num_kv_heads"]
            head_dim = meta_info["head_dim"]

            if total_layers == 0:
                total_layers = 32  # fallback
            if num_kv_heads == 0:
                num_kv_heads = 8
            if head_dim == 0:
                head_dim = 128

            # Weight distribution: -1 = all layers on GPU (llama.cpp native
            # "offload everything" idiom). No magic 99 / 100 fakes.
            gpu_frac = 1.0 if n_gpu_layers == -1 else min(n_gpu_layers / total_layers, 1.0)
            weights_vram = size_gb * gpu_frac
            weights_ram = size_gb * (1.0 - gpu_frac)

            # KV cache: n_ctx * 2 * n_layers * n_kv_heads * head_dim * 2 bytes (fp16)
            kv_bytes = n_ctx * 2 * total_layers * num_kv_heads * head_dim * 2
            kv_gb = kv_bytes / (1024**3)

            result.update({
                "weights_gb": round(weights_vram, 2),
                "kv_cache_gb": round(kv_gb, 2),
                "total_vram_gb": round(weights_vram + kv_gb, 2),
                "total_ram_gb": round(weights_ram, 2),
                "total_layers": total_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            })
        else:
            # Safetensors — approximate from file size
            if path.is_dir():
                total = sum(
                    f.stat().st_size for f in path.rglob("*")
                    if f.suffix in (".safetensors", ".bin", ".pt")
                ) / (1024**3)
            else:
                total = path.stat().st_size / (1024**3) if path.exists() else 0
            result["weights_gb"] = round(total, 2)
            result["total_vram_gb"] = round(total, 2)
            result["total_ram_gb"] = 0.0
        return result

    @staticmethod
    def read_model_metadata(model_path):
        """Read model metadata (layer count, etc.) without loading the full model.

        Uses an in-process cache keyed on (path, mtime) so repeated calls on the
        same file are O(1). Reads only the GGUF header (first few KB) by hand
        so we never walk the whole file just to peek at layer counts.
        """
        import json
        from pathlib import Path
        path = Path(model_path)
        try:
            st = path.stat()
            cache_key = (str(path), st.st_mtime, st.st_size)
        except OSError:
            cache_key = (str(path), 0, 0)
        cached = _GGUF_META_CACHE.get(cache_key)
        if cached is not None:
            return cached
        result = {"total_layers": 0, "num_kv_heads": 0, "head_dim": 0, "n_ctx_default": 4096}

        if path.is_file() and path.suffix == ".gguf":
            try:
                # Fast path: parse GGUF header in-process (first few KB).
                # GGUF files are little-endian; header has magic 'GGUF' (0x46554747).
                # Format (v2/v3):
                #   magic: 4 bytes ('GGUF')
                #   version: uint32
                #   tensor_count: uint64
                #   kv_count: uint64
                #   For each KV:
                #     key_length: uint64
                #     key: utf-8 bytes
                #     value_type: uint32 (0..=10 are scalar; >10 are arrays)
                #     value: depends on type
                import struct
                with open(path, "rb") as f:
                    magic = f.read(4)
                    if magic != b"GGUF":
                        raise ValueError("not a GGUF file")
                    f.read(4)  # version (v2 or v3)
                    f.read(8)  # tensor_count
                    f.read(8)  # kv_count (we don't need exact — bounded by safety cap)
                    # The actual key format is "{arch}.X" (e.g. "qwen35.block_count"),
                    # but we don't know `arch` until we find general.architecture.
                    # Collect every KV (cheap — small header), match the arch-prefixed
                    # ones after we know the architecture.
                    found_arch = None
                    for _ in range(500):  # safety cap
                        raw = f.read(8)
                        if len(raw) < 8:
                            break
                        klen = struct.unpack("<Q", raw)[0]
                        if klen == 0 or klen > 1024:
                            break
                        key = f.read(klen).decode("utf-8", errors="ignore").strip("\x00")
                        raw_t = f.read(4)
                        if len(raw_t) < 4:
                            break
                        gtype = struct.unpack("<I", raw_t)[0]
                        if gtype == 4:  # UINT32
                            v = struct.unpack("<I", f.read(4))[0]
                        elif gtype == 5:  # INT32
                            v = struct.unpack("<i", f.read(4))[0]
                        elif gtype == 8:  # STRING
                            slen = struct.unpack("<Q", f.read(8))[0]
                            v = f.read(slen).decode("utf-8", errors="ignore").strip("\x00")
                        elif gtype == 10:  # BOOL
                            v = bool(f.read(1)[0])
                        elif gtype == 0:  # UINT8
                            v = struct.unpack("<B", f.read(1))[0]
                        elif gtype == 6:  # FLOAT32
                            f.read(4); v = None
                        elif gtype == 7:  # FLOAT64
                            f.read(8); v = None
                        else:
                            # ARRAY (type 9): [4-byte elem_type][8-byte count][elements...]
                            etype = struct.unpack("<I", f.read(4))[0]
                            acount = struct.unpack("<Q", f.read(8))[0]
                            if etype == 8:  # string array: variable-size elements
                                for _ in range(acount):
                                    slen = struct.unpack("<Q", f.read(8))[0]
                                    f.read(slen)
                            else:
                                sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:8, 10:1}
                                esize = sizes.get(etype, 1)
                                f.read(acount * esize)
                            v = None
                        if key == "general.architecture" and isinstance(v, str):
                            found_arch = v
                            result["total_layers"] = 0  # reset so we know real values vs defaults
                            continue  # we'll re-fill below once we know arch
                        # Once we know the arch, accept arch-prefixed scalars.
                        if found_arch and key.startswith(found_arch + "."):
                            short = key[len(found_arch) + 1:]
                            if short == "block_count" and isinstance(v, int):
                                result["total_layers"] = int(v)
                            elif short == "attention.head_count_kv" and isinstance(v, int):
                                result["num_kv_heads"] = int(v)
                            elif short == "attention.key_length" and isinstance(v, int):
                                result["head_dim"] = int(v)
                            elif short == "context_length" and isinstance(v, int):
                                result["n_ctx_default"] = int(v)
                            if (result["total_layers"] > 0 and result["num_kv_heads"] > 0
                                    and result["head_dim"] > 0 and result["n_ctx_default"] != 4096):
                                # Got everything we need
                                break
                    # If arch-prefixed keys were never found, fall back to full reader.
                    if result["total_layers"] == 0 or result["num_kv_heads"] == 0 or result["head_dim"] == 0:
                        try:
                            from gguf.gguf_reader import GGUFReader
                            reader = GGUFReader(str(path))
                            arch_field = reader.fields.get("general.architecture")
                            if arch_field:
                                arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
                                for key_suffix, result_key, fallback in [
                                    (".block_count", "total_layers", 0),
                                    (".attention.head_count_kv", "num_kv_heads", 0),
                                    (".attention.key_length", "head_dim", 0),
                                    (".context_length", "n_ctx_default", 4096),
                                ]:
                                    field = reader.fields.get(f"{arch}{key_suffix}")
                                    if field:
                                        val = field.parts[field.data[0]]
                                        result[result_key] = int(val[0]) if len(val) else fallback
                            del reader
                        except Exception:  # noqa: BLE001,S110
                            pass
            except Exception:  # noqa: BLE001,S110
                pass


        elif path.is_dir() and (path / "config.json").exists():
            try:
                with open(path / "config.json") as f:
                    cfg = json.load(f)
                result["total_layers"] = cfg.get("num_hidden_layers", 0)
                result["num_kv_heads"] = cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 0))
                result["head_dim"] = cfg.get("hidden_size", 0) // max(cfg.get("num_attention_heads", 1), 1)
                result["n_ctx_default"] = cfg.get("max_position_embeddings", 4096)
            except Exception:  # noqa: BLE001,S110
                pass
        # Cache the result (even partial) so next call is O(1)
        _GGUF_META_CACHE[cache_key] = result
        while len(_GGUF_META_CACHE) > _GGUF_META_CACHE_MAX:
            _GGUF_META_CACHE.popitem(last=False)
        return result

    def _generate_gguf(self, messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop):
        if self.vision and self.mmproj_path:
            # Use chat_handler which supports image content
            result = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
            )
            return result["choices"][0]["message"]["content"].strip()
        # Text-only: respect the GGUF's built-in tokenizer.chat_template via the existing
        # finetune_studio.templates.renderer. Mixing templates across models is fragile —
        # we never fall back to a hardcoded prompt; the renderer itself falls
        # back to ChatML only when the GGUF has no template at all.
        from finetune_studio.templates.renderer import render_chat
        tmpl = self._gguf_template or {}
        bos = tmpl.get("bos_token", "<bos>")
        prompt = render_chat(
            template_str=tmpl.get("chat_template", ""),
            messages=messages,
            tools=None,
            bos_token=bos,
            eos_token=tmpl.get("eos_token", "<eos>"),
            add_generation_prompt=True,
        )
        # llama.cpp prepends BOS itself when tokenizing; a template that also
        # renders it yields a duplicate leading BOS (degrades Gemma-style models).
        if bos and prompt.startswith(bos):
            prompt = prompt[len(bos):]
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )
        return output["choices"][0]["text"].strip()
